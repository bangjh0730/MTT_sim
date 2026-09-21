"""
Per-(target, event) recovery extraction — the shared basis for every recovery
figure and table in evaluate/.

The unit of analysis is ONE DISTURBANCE HITTING ONE TARGET, not a fleet-average
curve. Each such pair contributes exactly one number (slots to recovery), which
is what makes overlapping disturbances tractable: a fleet-average PCRLB trace
smears two events into one blob, whereas an event whose window is interrupted by
a second event can simply be FLAGGED (clean=False) and reported separately, so a
slow value is never silently blamed on the policy when it was really event
pile-up.

Event -> affected target
-----------------------
  birth   : target k is born at slot s. k itself is the affected target — it
            starts at the ~8e4 m^2 spawn prior and must be driven down.
  failure : UAV i is physically lost at slot s. The affected target is whatever
            k UAV i was tracking at s — it loses a tracker's information from s
            onward (envs/MTTEnv.py only sums observation_info for `uav.active`).

Deaths are deliberately NOT onsets. A death removes a term from the metric
rather than degrading a target, so its "recovery" direction depends on whether
the removed target happened to be well- or poorly-tracked — see the same
argument in evaluate/event_aligned.py. Deaths DO act as censors and as
contaminating events (a fleet-shed frees a UAV, which perturbs the recovery it
lands in).

Onset for a failure is the PHYSICAL failure slot, not the slot the BS detects it
(N_FAIL_DETECT=5 slots later). The target starts degrading the moment the UAV
goes silent, and that is what "how long was this target degraded" has to count.
Both policies pay the identical detection delay, so it shifts both by the same
constant and changes no comparison.

Which UAV failed
----------------
raw_eval.npz stores failure_event_uavs for runs saved after that field was
added; older runs only recorded WHEN a failure happened, not to whom, so
_failure_onsets falls back to reconstructing it from the assignments array: the
BS clears a lost UAV's assignment exactly once, on the N_FAIL_DETECT-th silent
slot (envs/MTTEnv.py step ~line 357), so the failed UAV is the one whose
assignment goes k -> -1 at slot s + N_FAIL_DETECT - 1.

The fallback is not guesswork on faith — it was checked three ways: it finds
exactly one candidate (never two) on every failure across the saved runs; the
UAV ids it recovers agree between the marl and agentic runs of each seed, as
they must since both replay the same DisturbanceSchedule; and on fresh runs
that carry BOTH the logged field and the assignments array it reproduces the
logged (slot, uav, target) triples exactly. Prefer the logged field regardless:
it is the ground truth, and the reconstruction only exists so the runs saved
before it was added stay analysable.

The reconstruction finds no candidate in exactly one situation: the failure
dropped the active fleet below the target count, the resulting fleet-shed
(envs/birth_death.py) removed the very target that UAV was tracking, and the
assignment was therefore already cleared by remove_target before the BS ever
declared the UAV lost. There is then no surviving target to recover, so the
event is correctly dropped either way.

Recovery
--------
Slots from onset until target k's OWN PCRLB first drops below `threshold` and
stays there for MIN_RUN consecutive slots — the first sustained dip, not the
first touch, so one noisy under-threshold slot isn't mistaken for recovery.
The default threshold is the steady-state PCRLB of a target tracked by one UAV
at exactly SNR_MIN=20dB in the worst still-valid geometry (see envs/PCRLB.py):
below it, the target is being adequately sensed by definition.

An absolute threshold is used rather than each event's own pre-event baseline
because a newborn target HAS no pre-event baseline, and a single criterion keeps
birth and failure events on one comparable axis. Events whose target never
crossed the threshold at all (recovery == 0) are marked no_impact: the
disturbance did not actually degrade that target below adequate sensing — most
often a co-tracked target that kept another UAV on it. Those still contribute
their 0 to the distribution rather than being filtered out, so the CDF's
y-intercept reads as "share of disturbances that never degraded the target at
all" — a real difference between the policies (7/32 vs 3/31 compound events)
that an exclusion rule would have thrown away.
"""
# ---------------------------------------------------------------------------
# COVERAGE-ERA ANALYSIS — NOT UPDATED FOR THE OVERLOADED / RESCUE REGIME.
#
# This script was written when the simulation guaranteed |K| <= |U|: every target
# had its own UAV, a disturbance opened a COVERAGE GAP, and the thing worth
# measuring was how many slots the system took to close that gap and bring the
# target's PCRLB back down. None of those premises hold any more. With 3 UAVs and
# 8+ targets most targets are unsensed in any given slot by construction, so
# there is no coverage gap to open and no recovery to time; targets now leave by
# being RESCUED, and the objective is the average rescue delay (Eq. 19).
#
# It is kept for reference rather than deleted, but it reads npz fields that
# evaluate/utils.py no longer writes (single-id `assignments`, `death_events`)
# and will not run against new results. The replacements are:
#   * per-run mission curves ....... evaluate/plot.py::plot_eval_rescue
#   * policy comparison ............ evaluate/plot.py::plot_mode_comparison
#   * cross-run / lambda sweep ..... evaluate/compare_runs.py
#   * LLM tier cost ................ evaluate/llm_latency.py
# ---------------------------------------------------------------------------

import os
import glob
import re
from dataclasses import dataclass

import numpy as np

from config.params import N_FAIL_DETECT

# Okabe-Ito CVD-safe pair (validated: lightness/chroma/CVD/contrast all pass).
BLUE, VERM = "#0072B2", "#D55E00"
BLUE_D, VERM_D = "#004E7A", "#8F3F00"
INK, MUTED = "#222222", "#666666"

# (display label, folder under plots/). The three disturbance modes the paper
# reports; "Compound" is the one the rule-based policy was never trained for.
MODES = [("Target birth", "birth"),
         ("UAV failure", "failure"),
         ("Compound", "birth+fail")]

POLICIES = [("Rule-based", "marl", BLUE, BLUE_D),
            ("Agentic", "agentic", VERM, VERM_D)]

# Steady-state PCRLB (m^2) of a target continuously tracked by one UAV at exactly
# SNR_MIN=20dB and the worst still-valid range/geometry — see the derivation in
# envs/PCRLB.py (predict_bfim + observation_info fixed point).
SENSED_THRESHOLD = 0.7627

MIN_RUN = 2   # consecutive under-threshold slots required to count as recovered

# Horizon (slots after onset) over which an event is checked for interference by
# a second disturbance. This is deliberately a FIXED constant rather than the
# event's own realised recovery window: "was this event interrupted before it
# recovered" sounds more natural but is length-biased, because a slow recovery
# has a proportionally longer window in which to catch a second event and so is
# far likelier to be discarded. Filtering on it would preferentially delete the
# slow tail — exactly the rule-based policy's failure cases — and flatter the
# baseline. Judging every event over the same fixed horizon makes cleanliness a
# property of the event's CONTEXT, independent of its outcome, so the clean
# subset stays an unbiased sample. (Sanity check that this holds: the clean
# count now comes out IDENTICAL for both policies of a seed, as it must — both
# replay the same DisturbanceSchedule.)
#
# 150 slots trades the two pressures: long enough to clear the agentic policy's
# p90 recovery (~115 slots) with margin, so a "clean" event really did have room
# to finish, and short enough to leave a usable number of clean events (10/31 in
# the compound mode; a 250-slot horizon leaves 7).
#
# This doubles as the default post-onset window of the event-aligned traces
# (evaluate/event_aligned.py). Keep those coupled: plotting further past the
# onset than events were screened would put unscreened slots — where a second
# disturbance may well have landed — inside a figure whose whole claim is that
# it shows clean events only.
CLEAN_HORIZON = 180


@dataclass(frozen=True)
class RecoveryEvent:
    """One disturbance hitting one target."""
    mode:    str     # mode folder ("birth" / "failure" / "birth+fail")
    policy:  str     # "marl" / "agentic"
    seed:    int
    kind:    str     # "birth" / "failure"
    onset:   int     # 1-indexed slot the disturbance hit the target
    target:  int
    uav:     int     # UAV lost (failure events); -1 for births
    recovery: int    # slots from onset to sustained recovery; None if never
    censor:  str     # "" if recovered, else "death" | "episode_end"
    clean:   bool    # no other disturbance within CLEAN_HORIZON slots of onset
    blocker: str     # the interfering disturbance slots; "" when clean
    path:    str     # raw_eval.npz this came from, for consumers that need the series

    @property
    def no_impact(self) -> bool:
        """The target never went above threshold — the disturbance didn't
        actually degrade it (e.g. a co-tracked target that kept a tracker)."""
        return self.recovery == 0


def _seed_of(path: str) -> int:
    m = re.search(r"seed(\d+)", path)
    return int(m.group(1)) if m else -1


def find_runs(mode_dir: str, policy: str) -> list:
    return sorted(glob.glob(os.path.join("plots", mode_dir, "seed*", policy, "raw_eval.npz")))


def _failure_onsets(d) -> list:
    """[(slot, uav, target), ...] — one per failure that hit a surviving target.

    Prefers the logged failure_event_uavs; falls back to reconstructing the
    victim from the assignments array for runs saved before that field existed
    (see module docstring)."""
    A = d["assignments"]
    out = []

    if "failure_event_uavs" in d.files:
        pairs = zip(d["failure_event_slots"].tolist(), d["failure_event_uavs"].tolist())
    else:
        pairs = []
        for s in d["failure_slots"].tolist():
            det = s + N_FAIL_DETECT - 1          # slot the BS clears the assignment
            j = det - 1                          # -> index
            if j <= 0 or j >= A.shape[1]:
                continue
            cand = [i for i in range(A.shape[0]) if A[i, j] == -1 and A[i, j - 1] != -1]
            if len(cand) != 1:
                continue    # fleet-shed already took this UAV's target; nothing survives
            pairs.append((s, cand[0]))

    for s, i in pairs:
        if not (1 <= s <= A.shape[1]):
            continue
        k = int(A[i, s - 1])    # assignment AT the failure slot (cleared only later)
        if k >= 0:
            out.append((int(s), int(i), k))
    return out


def _recovery_slots(series: np.ndarray, threshold: float, min_run: int = MIN_RUN):
    """(slots_to_recovery, censor). series starts AT the onset slot.

    Recovery is the first index whose next min_run slots are all below
    threshold. NaN means the target is no longer alive, which censors the
    event rather than counting as recovery."""
    alive = ~np.isnan(series)
    end = int(np.argmin(alive)) if not alive.all() else len(series)   # first NaN
    live = series[:end]
    below = live < threshold
    for i in range(len(below) - min_run + 1):
        if below[i:i + min_run].all():
            return i, ""
    return None, ("death" if end < len(series) else "episode_end")


def _disturbance_slots(d) -> list:
    """Every distinct disturbance slot in the run — births, failures and deaths
    alike. Used only to decide whether an event was interfered with; deaths
    count because a fleet-shed frees a UAV and so perturbs whatever recovery is
    in flight. Deduplicated: one failure and the fleet-shed death it triggers
    share a slot, and that is one disturbance, not two."""
    return sorted(set(d["birth_event_slots"].tolist()
                      + d["failure_slots"].tolist()
                      + d["death_event_slots"].tolist()))


def load_events(mode_dir: str, policy: str, threshold: float = SENSED_THRESHOLD,
                clean_horizon: int = CLEAN_HORIZON) -> list:
    """Every (target, event) recovery record for one (mode, policy)."""
    events = []
    for f in find_runs(mode_dir, policy):
        d = np.load(f)
        if "pcrlb_per_target" not in d.files or "assignments" not in d.files:
            continue
        P    = d["pcrlb_per_target"]
        seed = _seed_of(f)

        onsets = [(int(s), "birth", int(k), -1)
                  for s, k in zip(d["birth_event_slots"].tolist(),
                                  d["birth_event_targets"].tolist())]
        onsets += [(s, "failure", k, i) for s, i, k in _failure_onsets(d)]

        others = _disturbance_slots(d)

        for slot, kind, k, uav in onsets:
            if not (1 <= slot <= P.shape[1]) or k >= P.shape[0]:
                continue
            series = P[k, slot - 1:]
            if series.size == 0 or np.isnan(series[0]):
                continue
            rec, censor = _recovery_slots(series, threshold)

            # Interference: another disturbance inside the fixed post-onset
            # horizon (see CLEAN_HORIZON on why the horizon is fixed rather than
            # the realised window). Disturbances AT the onset slot are part of
            # this same disturbance — a failure and the fleet-shed it triggers
            # share a slot — so the window opens strictly after it.
            hits = [s for s in others if slot < s <= slot + clean_horizon]
            events.append(RecoveryEvent(
                mode=mode_dir, policy=policy, seed=seed, kind=kind, onset=slot,
                target=k, uav=uav, recovery=rec, censor=censor,
                clean=not hits,
                blocker=(",".join(str(s) for s in hits) if hits else ""),
                path=f,
            ))
    return events


def load_all(threshold: float = SENSED_THRESHOLD,
             clean_horizon: int = CLEAN_HORIZON) -> dict:
    """{(mode_key, policy_key): [RecoveryEvent, ...]} for every mode x policy."""
    return {(mkey, pkey): load_events(mkey, pkey, threshold, clean_horizon)
            for _, mkey in MODES for _, pkey, _, _ in POLICIES}


def outcomes(events: list, clean_only: bool = False) -> list:
    """One entry per event — slots-to-recovery, or None if it never recovered.

    Nothing is filtered out. That is the whole point of the per-event framing:
    every (target, event) pair contributes exactly one number, so the resulting
    distribution needs no exclusion rule to defend and cannot be reshaped by
    one. No-impact events contribute 0 (the disturbance never pushed that
    target above the threshold, so it was never un-recovered), which shows up
    as a positive y-intercept on the CDF rather than as a silent deletion.
    Never-recovered events contribute None and hold the curve below 1.0.
    """
    return [e.recovery for e in events if e.clean or not clean_only]


def censored_quantile(vals: list, q: float):
    """(value, is_lower_bound) for quantile q of `vals`, which may contain None
    for never-recovered events.

    Sorting None last is correct — a censored event's true recovery time is
    longer than every observed one — but a quantile that lands ON a censored
    entry is not a number we know. It is reported as a LOWER BOUND (the largest
    observed value, is_lower_bound=True) rather than silently dropped, because
    dropping the censored events is exactly what would hide the tail: the
    rule-based policy's compound p90 is undefined precisely BECAUSE ~1 in 6 of
    its events never recover, and that is the finding, not a gap in the data.
    """
    if not vals:
        return None, False
    finite = sorted(v for v in vals if v is not None)
    if not finite:
        return None, True
    rank = q * (len(vals) - 1)          # index into the full, censored-last order
    if rank <= len(finite) - 1:
        return float(np.interp(rank, np.arange(len(finite)), finite)), False
    return float(finite[-1]), True      # quantile falls among the censored


def fmt_quantile(vals: list, q: float) -> str:
    """censored_quantile rendered for a table cell: '143' or '>232'."""
    v, bound = censored_quantile(vals, q)
    if v is None:
        return "n/a"
    return f"{'>' if bound else ''}{v:.0f}"


def partition(events: list, clean_only: bool = False) -> dict:
    """Counts behind every recovery figure, so no figure re-derives them."""
    sel = [e for e in events if e.clean or not clean_only]
    return dict(
        n_total=len(sel),
        n_no_impact=sum(1 for e in sel if e.no_impact),
        n_recovered=sum(1 for e in sel if e.recovery is not None),
        n_censored=sum(1 for e in sel if e.recovery is None),
        n_contaminated=sum(1 for e in events if not e.clean),
    )
