"""
Time-to-full-retracking: how long the FLEET AS A WHOLE takes to get every live
target back under the sensing threshold after a disturbance, and — separately —
after the moment the two policies' assignments first genuinely split apart.

This is a different unit of analysis from evaluate/recovery_events.py, which is
per-(target, event): one event contributes one number for the ONE target it hit,
and other targets' state doesn't matter. Here, one event contributes one number
for the ENTIRE fleet: "full retracking" at a slot requires EVERY currently-alive
target to be under threshold simultaneously, so a straggler target holds the
whole event's clock open even if every other target recovered instantly. This
is deliberately the harsher, fleet-wide framing.

Definitions
-----------
Recovery threshold: SENSED_THRESHOLD (0.7627 m^2), same bar used everywhere
else in evaluate/ — a target's own PCRLB below it counts as adequately sensed.

Full retracking at slot s: every target alive at s (its PCRLB series is not
NaN there) has PCRLB(s) < threshold. Vacuously true if no target is alive
(defensive; the env keeps >= 1 live target in practice, so this edge is not
expected to fire).

Event: a disturbance onset in this run — a target birth or a UAV's PHYSICAL
failure slot (not the slot the BS detects it), matching the onset convention in
evaluate/recovery_events.py. Target deaths are NOT events here (same reasoning
as there), but unlike recovery_events.py this metric does not need a "clean"
filter for them either: an unrelated target dying mid-window just removes a
term from the fleet-wide AND, it cannot corrupt the metric the way it could
corrupt a single target's own recovery curve.

Time-to-full-retracking for an event at slot t_e: search the window
[t_e, window_end] (see WINDOW_CAP below) for the first slot t_r such that full
retracking holds at t_r AND at every slot from t_r to window_end. That "stays
true" condition — not just a touch, but true all the way to the end of the
window — is what keeps one noisy dip near the end of a long window from being
counted as a real recovery earlier on. Equivalently: find the LAST slot in the
window where full retracking is False; t_r is the slot right after it (t_r =
t_e itself if it's never False in the window at all). If the window's last slot
is still False, no such t_r exists and the event is CENSORED — right-censored
at the window length, not dropped. Every figure and printed stat here reports
the censored count/fraction alongside the recovered ones rather than silently
excluding them; median/IQR/mean are computed on the recovered (finite) values
only, since a censored event's true recovery time is unknown, merely known to
exceed the window.

WINDOW_CAP is a plain module constant so it's a one-line change, not a refactor.

Window boundary — "next event": each event's search window ends at
min(t_e + WINDOW_CAP, next event onset in THIS policy's own run, episode end).
Capping at the next event (rather than always using the full WINDOW_CAP) stops
a later, unrelated disturbance from being folded into "how long did this one
take" — even though the fleet-wide metric doesn't NEED this the way the
per-target one does (see above), a second disturbance landing mid-window would
still add new information-loss dynamics that have nothing to do with the event
being measured, so the boundary is kept for both the unpaired (B1) and paired
(B2) datasets.

The DEFAULT drops that boundary (pass --bounded to keep it): the window becomes
the whole rest of the run, [t_e, T). This is only coherent alongside a change of
recovery criterion, because the bounded rule ("full retracking holds to the END
of the window") becomes "holds continuously to slot T" over a full-episode
window — which a later, legitimate disturbance breaks, pushing recovery past it
or censoring it outright, the OPPOSITE of the intent. So the full-episode
default instead defines recovery as the first SUSTAINED run of MIN_RUN
consecutive full-coverage slots (the same MIN_RUN rule as
evaluate/recovery_events.py): chatter-proof, but not requiring the fleet to
never be disturbed again. Under it, only targets that genuinely never reach
sustained full coverage before slot T remain censored (~1%, vs ~19% bounded).

Two datasets
------------
B1 (unpaired, "all events"): every disturbance event, from every matched-seed
episode (a seed counted only if BOTH policies have a saved run — "matched
seeds" per the request), across all three disturbance modes, pooled into two
plain lists: every rule-based event's time, every agentic event's time. No
pairing across policies is attempted or needed here.

B2 (paired, "first divergence only"): reuses evaluate/reassign_agreement.py's
_first_divergence — the first slot a seed's rule-based and agentic assignment
vectors genuinely split apart (with its existing lag-tolerance window, since up
to that point the Agentic Planner reasoning for a few slots is not a real
disagreement, just a delay). Both runs are IDENTICAL up to that slot by
construction (same schedule, same starting assignment), so it is a legitimate
shared t_0. From that single shared slot, time-to-full-retracking is computed
independently in each policy's own PCRLB series (each with its own window-
capping "next event", since the two policies' subsequent disturbance timelines
can differ once batteries start depleting at different rates) — giving one
matched pair per episode.

Run:  python -m evaluate.full_retrack                 # full-episode window (default)
      python -m evaluate.full_retrack --bounded       # next-event-capped window
      python -m evaluate.full_retrack --b1-style violin
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

from evaluate.recovery_events import (
    MODES, POLICIES, SENSED_THRESHOLD, MIN_RUN, INK, MUTED, find_runs,
)
from evaluate.reassign_agreement import _first_divergence, WINDOW as DIVERGENCE_WINDOW

# Slots searched after an event (or until the next event / episode end, if
# sooner) before declaring "never recovered" and right-censoring. A single
# constant, deliberately at module top so it's a one-line change.
WINDOW_CAP = 400


# -- shared event/window machinery ------------------------------------------

def _event_columns(d) -> list:
    """0-indexed columns of every disturbance onset in one run: births and
    PHYSICAL UAV failures. Sorted, deduplicated (a birth and a failure sharing
    a slot is one event boundary, not two)."""
    cols = []
    if "birth_event_slots" in d.files:
        cols += [int(s) - 1 for s in d["birth_event_slots"].tolist()]
    if "failure_slots" in d.files:
        cols += [int(s) - 1 for s in d["failure_slots"].tolist()]
    return sorted(set(cols))


def _full_retrack_mask(P: np.ndarray, threshold: float) -> np.ndarray:
    """f[c] = True iff every target alive at column c has PCRLB < threshold.
    P: (MAX_TARGETS, n_cols) slice. A target counts as alive at c iff P[k, c]
    is not NaN; a non-alive target never blocks the AND."""
    alive = ~np.isnan(P)
    ok = np.where(alive, P < threshold, True)
    return ok.all(axis=0)


def _search_recovery(f: np.ndarray):
    """f: full-retrack mask over one event's search window, in order starting
    at the event slot. Returns (time_or_None, window_len). time is slots from
    the event to the first index after which f stays True for the rest of the
    window (0 if f is True throughout); None (censored) if f's last entry is
    False."""
    L = len(f)
    if L == 0:
        return None, 0
    false_idx = np.where(~f)[0]
    if false_idx.size == 0:
        return 0, L
    if false_idx[-1] == L - 1:
        return None, L
    return int(false_idx[-1] + 1), L


def _search_recovery_sustained(f: np.ndarray, min_run: int):
    """Full-episode variant of _search_recovery.

    Instead of "stays True to the end", recovery is the first index i where the
    fleet holds full retracking for min_run consecutive slots. This is the right
    criterion when the window runs all the way to episode end: a LATER, unrelated
    disturbance legitimately breaks full coverage again, so demanding permanence
    (the bounded-window rule) would wrongly push recovery past it or censor a
    genuine recovery outright. A sustained run of min_run slots filters chatter
    without demanding the fleet never be disturbed again. Matches the MIN_RUN
    rule in evaluate/recovery_events.py.

    Censored (None) only if the fleet never sustains full coverage for min_run
    slots before the episode ends."""
    L = len(f)
    if L < min_run:
        return None, L
    for i in range(L - min_run + 1):
        if f[i:i + min_run].all():
            return i, L
    return None, L


def _window_end(event_cols: list, c0: int, T: int, cap: int,
                full_episode: bool = False) -> int:
    """Exclusive end column for the search window starting at c0.

    Bounded (default): capped by WINDOW_CAP, the next event in this same run,
    and the episode length — so a later disturbance's dynamics never fold into
    this event's recovery time.

    full_episode: the whole rest of the run [c0, T). The next-event boundary is
    dropped, so this only makes sense together with the sustained-run recovery
    criterion (_search_recovery_sustained); see its docstring for why."""
    if full_episode:
        return T
    later = [c for c in event_cols if c > c0]
    next_c = min(later) if later else T
    return min(c0 + cap, next_c, T)


def _matched_episodes(mode_dir: str) -> list:
    """[(seed_label, marl_npz_path, agentic_npz_path), ...] for one mode —
    only seeds where BOTH policies have a saved run ("matched seeds")."""
    out = []
    for f_marl in find_runs(mode_dir, "marl"):
        seed_dir = os.path.dirname(os.path.dirname(f_marl))
        f_ag = os.path.join(seed_dir, "agentic", "raw_eval.npz")
        if os.path.exists(f_ag):
            out.append((os.path.basename(seed_dir), f_marl, f_ag))
    return out


# -- B1: every event, unpaired -----------------------------------------------

def _recover(mask: np.ndarray, full_episode: bool):
    """Dispatch to the bounded ('stays True to window end') or full-episode
    ('first sustained min_run run') recovery criterion."""
    if full_episode:
        return _search_recovery_sustained(mask, MIN_RUN)
    return _search_recovery(mask)


def _events_for_run(path: str, threshold: float, cap: int,
                    full_episode: bool = False) -> list:
    """One dict per disturbance event in this single run."""
    d = np.load(path)
    if "pcrlb_per_target" not in d.files:
        return []
    P = d["pcrlb_per_target"]
    T = P.shape[1]
    cols = _event_columns(d)

    kind_of = {}
    for s in (d["birth_event_slots"].tolist() if "birth_event_slots" in d.files else []):
        kind_of[int(s) - 1] = "birth"
    for s in (d["failure_slots"].tolist() if "failure_slots" in d.files else []):
        kind_of.setdefault(int(s) - 1, "failure")

    out = []
    for c0 in cols:
        hi = _window_end(cols, c0, T, cap, full_episode)
        rec, L = _recover(_full_retrack_mask(P[:, c0:hi], threshold), full_episode)
        out.append(dict(slot=c0 + 1, kind=kind_of.get(c0, "?"), recovery=rec, window=L))
    return out


def load_b1(threshold: float = SENSED_THRESHOLD, cap: int = WINDOW_CAP,
            full_episode: bool = False) -> dict:
    """{'marl': [...], 'agentic': [...]} of per-event dicts, pooled across every
    matched-seed episode in all three disturbance modes."""
    out = {pkey: [] for _, pkey, _, _ in POLICIES}
    for _, mkey in MODES:
        for _, f_marl, f_ag in _matched_episodes(mkey):
            for pkey, f in (("marl", f_marl), ("agentic", f_ag)):
                for ev in _events_for_run(f, threshold, cap, full_episode):
                    ev["mode"] = mkey
                    out[pkey].append(ev)
    return out


# -- B2: first divergence, paired --------------------------------------------

def load_b2(threshold: float = SENSED_THRESHOLD, cap: int = WINDOW_CAP,
           divergence_window: int = DIVERGENCE_WINDOW,
           full_episode: bool = False) -> list:
    """One dict per episode that genuinely diverges: mode, seed, div_slot, and
    each policy's own (recovery, window) measured from that shared slot."""
    out = []
    for _, mkey in MODES:
        for seed, f_marl, f_ag in _matched_episodes(mkey):
            dm, da = np.load(f_marl), np.load(f_ag)
            if "assignments" not in dm.files or "assignments" not in da.files:
                continue
            am, aa = dm["assignments"], da["assignments"]
            T = min(am.shape[1], aa.shape[1])
            div = _first_divergence(am[:, :T], aa[:, :T], divergence_window)
            if div is None:
                continue

            def _one(d):
                P = d["pcrlb_per_target"]
                cols = _event_columns(d)
                hi = _window_end(cols, div, P.shape[1], cap, full_episode)
                return _recover(_full_retrack_mask(P[:, div:hi], threshold), full_episode)

            rec_m, Lm = _one(dm)
            rec_a, La = _one(da)
            out.append(dict(mode=mkey, seed=seed, div_slot=div + 1,
                            marl=rec_m, marl_window=Lm,
                            agentic=rec_a, agentic_window=La))
    return out


# -- stats --------------------------------------------------------------------

def _stats(vals_with_censor: list) -> dict:
    """vals_with_censor: list of (recovery_or_None, window_len). Splits into
    finite recoveries (for median/IQR/mean) and a censored count/fraction
    (reported separately, never folded into the finite stats)."""
    finite = [r for r, _ in vals_with_censor if r is not None]
    n = len(vals_with_censor)
    n_cens = n - len(finite)
    out = dict(n=n, n_censored=n_cens,
               frac_censored=(n_cens / n) if n else 0.0)
    if finite:
        out.update(median=float(np.median(finite)),
                   iqr=(float(np.percentile(finite, 25)), float(np.percentile(finite, 75))),
                   mean=float(np.mean(finite)))
    else:
        out.update(median=None, iqr=(None, None), mean=None)
    return out


def _use_symlog(all_finite: list, ratio: float = 15.0) -> bool:
    """Heuristic: symlog if the tail is far past the bulk of the distribution."""
    if len(all_finite) < 4:
        return False
    p75 = np.percentile(all_finite, 75)
    return p75 > 0 and (max(all_finite) / p75) > ratio


# -- Figure B1: box + strip ----------------------------------------------------

def _plot_b1(b1: dict, out_path: str, style: str = "box"):
    labels = [plabel for plabel, _, _, _ in POLICIES]
    keys   = [pkey for _, pkey, _, _ in POLICIES]
    colors = {pkey: c for _, pkey, c, _ in POLICIES}
    darks  = {pkey: dc for _, pkey, _, dc in POLICIES}

    finite = {k: [e["recovery"] for e in b1[k] if e["recovery"] is not None] for k in keys}
    cens_n = {k: sum(1 for e in b1[k] if e["recovery"] is None) for k in keys}
    all_finite = [v for k in keys for v in finite[k]]
    symlog = _use_symlog(all_finite)

    plt.rcParams.update({
        "font.size": 11, "font.family": "sans-serif",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "axes.edgecolor": "#555555",
    })
    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    x = np.arange(1, len(keys) + 1)
    jit = np.random.default_rng(0)

    if style == "violin":
        # KDE density per policy. Clip at 0 so the kernel can't smear recovery
        # times below zero (they are non-negative by construction). The median is
        # not drawn by violinplot here; a short tick is added below so it matches
        # the box style's annotation.
        parts = ax.violinplot([finite[k] for k in keys], positions=x, widths=0.7,
                              showmeans=False, showmedians=False, showextrema=False)
        for i, k in enumerate(keys):
            body = parts["bodies"][i]
            body.set(facecolor=colors[k], edgecolor=darks[k], alpha=0.30, linewidth=1.2)
        for i, k in enumerate(keys):
            if finite[k]:
                med = float(np.median(finite[k]))
                ax.hlines(med, x[i] - 0.18, x[i] + 0.18, color=darks[k], lw=2.4, zorder=6)
    else:
        bp = ax.boxplot([finite[k] for k in keys], positions=x, widths=0.5,
                        showfliers=False, patch_artist=True, zorder=3)
        for i, k in enumerate(keys):
            bp["boxes"][i].set(facecolor=colors[k], alpha=0.25, edgecolor=darks[k], linewidth=1.4)
            bp["medians"][i].set(color=darks[k], linewidth=2.0)
            for part in ("whiskers", "caps"):
                bp[part][2 * i].set(color=darks[k], linewidth=1.1)
                bp[part][2 * i + 1].set(color=darks[k], linewidth=1.1)

    # Strip overlay. Kept in violin mode too (the density hides the actual n and
    # any gaps in it), just narrower and fainter so it reads as a rug, not a
    # second chart competing with the KDE.
    strip_w, strip_a, strip_s = (0.16, 0.35, 16) if style == "violin" else (0.32, 0.55, 22)
    for i, k in enumerate(keys):
        jx = x[i] + (jit.random(len(finite[k])) - 0.5) * strip_w
        ax.scatter(jx, finite[k], s=strip_s, color=darks[k], alpha=strip_a,
                   edgecolor="white", linewidth=0.4, zorder=4)
        if finite[k]:
            ax.annotate(f"med {np.median(finite[k]):.0f}", xy=(x[i], np.median(finite[k])),
                        xytext=(28, 0), textcoords="offset points", va="center",
                        ha="left", fontsize=8.5, color=darks[k], fontweight="bold",
                        bbox=dict(boxstyle="square,pad=0.15", fc="white",
                                 ec="none", alpha=0.85), zorder=6)

    # Censored points: a distinct marker band at the top of the axis, clearly
    # outside the finite data's range. The count label sits ABOVE the marker
    # band (not beside it) so it never prints through the jittered triangles.
    data_top = max(all_finite) if all_finite else 1.0
    band_y   = data_top * (1.12 if not symlog else 2.2)
    ylim_top = data_top * (1.11 if not symlog else 3.2)
    for i, k in enumerate(keys):
        n = cens_n[k]
        if n:
            jx = x[i] + (jit.random(n) - 0.5) * 0.26
            ax.scatter(jx, [band_y] * n, marker="^", s=40, color=darks[k],
                       edgecolor="white", linewidth=0.8, zorder=5)
            ax.annotate(f"{n} censored", xy=(x[i], band_y), xytext=(0, 12),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=8.5, color=darks[k],
                        bbox=dict(boxstyle="square,pad=0.15", fc="white",
                                 ec="none", alpha=0.85))

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Time to full re-tracking (slots)")
    if symlog:
        ax.set_yscale("symlog")
        ax.set_title("Symlog y-axis: tail is >15x the 75th percentile", fontsize=9.5, color=MUTED)
    # A violin KDE smooths past the data range; clamp the bottom to 0 so it never
    # implies negative recovery times.
    ax.set_ylim(0, ylim_top)
    ax.grid(axis="y", ls=":", alpha=0.4, zorder=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return symlog


# -- Figure B2: paired slope plot ----------------------------------------------

def _plot_b2(b2: list, out_path: str):
    colors = {pkey: c for _, pkey, c, _ in POLICIES}
    darks  = {pkey: dc for _, pkey, _, dc in POLICIES}

    finite_all = [e["marl"] for e in b2 if e["marl"] is not None] + \
                 [e["agentic"] for e in b2 if e["agentic"] is not None]
    symlog = _use_symlog(finite_all)
    data_top = max(finite_all) if finite_all else 1.0
    top      = data_top * (1.15 if not symlog else 2.0)   # censored marker band
    label_y  = top * (1.0 if not symlog else 1.35)        # count label, clear of the band

    plt.rcParams.update({
        "font.size": 11, "font.family": "sans-serif",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "axes.edgecolor": "#555555",
    })
    fig, ax = plt.subplots(figsize=(6.0, 5.6))
    x0, x1 = 0, 1
    rng = np.random.default_rng(0)

    # Every censored episode is placed at the same nominal "window edge" value,
    # so without jitter multiple episodes draw exactly one line on top of
    # another — a stack of 10 can look identical to a stack of 1. A small
    # per-episode jitter (independent x nudge at each column, small y nudge
    # only in the censored band) fans them out into a visible cluster instead.
    n_censored = sum(1 for e in b2 if e["marl"] is None or e["agentic"] is None)

    for e in b2:
        censored_m, censored_a = e["marl"] is None, e["agentic"] is None
        any_censored = censored_m or censored_a
        jx0 = x0 + (rng.random() - 0.5) * 0.10
        jx1 = x1 + (rng.random() - 0.5) * 0.10
        jy  = (rng.random() - 0.5) * 0.05 * data_top if any_censored else 0.0
        ym = (top if censored_m else e["marl"]) + jy
        ya = (top if censored_a else e["agentic"]) + jy
        ax.plot([jx0, jx1], [ym, ya], color=MUTED, lw=1.1,
                ls="--" if any_censored else "-",
                alpha=0.55 if any_censored else 0.75, zorder=2)
        ax.scatter([jx0], [ym], s=30, marker="^" if censored_m else "o",
                   color=darks["marl"], edgecolor="white", linewidth=0.6, zorder=3)
        ax.scatter([jx1], [ya], s=30, marker="^" if censored_a else "o",
                   color=darks["agentic"], edgecolor="white", linewidth=0.6, zorder=3)

    if n_censored:
        ax.annotate(f"{n_censored} pair(s) never reached full re-tracking within the "
                    "window\n(censored — jittered above so the count is visible)",
                    xy=(0.5, label_y), xycoords=("axes fraction", "data"),
                    ha="center", va="bottom", fontsize=8, color=MUTED)

    ax.set_xlim(-0.3, 1.3)
    ax.set_xticks([x0, x1])
    ax.set_xticklabels(["Rule-based", "Agentic (Proposed)"])
    ax.set_ylabel("Time to full re-tracking (slots)")
    if symlog:
        ax.set_yscale("symlog")
    ax.set_ylim(0, label_y)
    ax.grid(axis="y", ls=":", alpha=0.4, zorder=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return symlog


# -- console report -------------------------------------------------------------

def _print_report(b1: dict, b2: list):
    print("== B1: every disturbance event, unpaired ==")
    print()
    hdr = f"{'Policy':11s} {'n':>4s} {'censored':>10s} {'median':>7s} {'IQR':>15s} {'mean':>7s}"
    print(hdr)
    print("-" * len(hdr))
    for plabel, pkey, _, _ in POLICIES:
        s = _stats([(e["recovery"], e["window"]) for e in b1[pkey]])
        iqr = f"[{s['iqr'][0]:.0f}, {s['iqr'][1]:.0f}]" if s["iqr"][0] is not None else "n/a"
        med = f"{s['median']:.0f}" if s["median"] is not None else "n/a"
        mean = f"{s['mean']:.1f}" if s["mean"] is not None else "n/a"
        print(f"{plabel:11s} {s['n']:>4d} {s['n_censored']:>4d} ({s['frac_censored']*100:4.1f}%) "
              f"{med:>7s} {iqr:>15s} {mean:>7s}")
    print()

    print("== B2: paired first-divergence events ==")
    print()
    print(f"episodes with a genuine divergence: {len(b2)} "
          f"(divergence window = {DIVERGENCE_WINDOW} slots)")
    for pkey in ("marl", "agentic"):
        s = _stats([(e[pkey], e[f"{pkey}_window"]) for e in b2])
        iqr = f"[{s['iqr'][0]:.0f}, {s['iqr'][1]:.0f}]" if s["iqr"][0] is not None else "n/a"
        med = f"{s['median']:.0f}" if s["median"] is not None else "n/a"
        mean = f"{s['mean']:.1f}" if s["mean"] is not None else "n/a"
        label = "Rule-based" if pkey == "marl" else "Agentic"
        print(f"  {label:11s} n={s['n']:>3d}  censored={s['n_censored']}/{s['n']}  "
              f"median={med}  IQR={iqr}  mean={mean}")
    print()

    paired = [(e["marl"], e["agentic"]) for e in b2
             if e["marl"] is not None and e["agentic"] is not None]
    n_excluded = len(b2) - len(paired)
    print(f"Wilcoxon signed-rank test on paired, BOTH-recovered episodes "
          f"(n={len(paired)}; {n_excluded} pair(s) excluded — at least one side censored):")
    if len(paired) >= 1 and any(m != a for m, a in paired):
        m_vals = [m for m, _ in paired]
        a_vals = [a for _, a in paired]
        try:
            stat, p = wilcoxon(m_vals, a_vals)
            print(f"  W={stat:.1f}  p={p:.4g}")
        except ValueError as exc:
            print(f"  could not run (scipy: {exc})")
    else:
        print("  not enough non-tied pairs to run the test")


def main():
    parser = argparse.ArgumentParser(
        description="Time-to-full-retracking: fleet-wide coverage recovery, unpaired and "
                    "paired at first policy divergence")
    parser.add_argument("--threshold", type=float, default=SENSED_THRESHOLD,
                        help="PCRLB (m^2) below which a target counts as retracked")
    parser.add_argument("--cap", type=int, default=WINDOW_CAP,
                        help="Max slots searched after an event before declaring censored")
    parser.add_argument("--divergence-window", type=int, default=DIVERGENCE_WINDOW,
                        help="Lag tolerance (slots) used by reassign_agreement's "
                             "first-divergence detection")
    parser.add_argument("--bounded", action="store_true",
                        help="Opt out of the DEFAULT full-episode window and use the "
                             "bounded one instead (cap at --cap or the next disturbance). "
                             "By default each event's recovery is measured over the whole "
                             f"rest of the episode, with the first sustained {MIN_RUN}-slot "
                             "full-coverage run counting as recovery — so a later "
                             "disturbance breaking coverage again neither corrupts nor "
                             "censors this event. --bounded reverts to 'coverage holds to "
                             "the end of a next-event-capped window', which censors far "
                             "more events (~19% vs <1%).")
    parser.add_argument("--drop-censored", action="store_true",
                        help="Discard never-recovered events entirely instead of "
                             "right-censoring them. NOTE: this is a biased view — the "
                             "discarded events are the slow tail, and dropping them "
                             "flatters whichever policy has more of them (here the "
                             "rule-based one), so the gap it shows is a floor, not the "
                             "real effect. Kept off by default for that reason.")
    parser.add_argument("--b1-style", choices=["box", "violin"], default="box",
                        help="B1 figure: box+strip (default) or violin+rug")
    parser.add_argument("-o", "--out", default="plots/paper")
    args = parser.parse_args()

    full_episode = not args.bounded
    b1 = load_b1(args.threshold, args.cap, full_episode)
    b2 = load_b2(args.threshold, args.cap, args.divergence_window, full_episode)

    tag = ""
    if full_episode:
        print(f"FULL-EPISODE mode (default): window = whole rest of episode; recovery = "
              f"first sustained {MIN_RUN}-slot full-coverage run.\n")
    else:
        tag += "_bounded"
        print("BOUNDED-WINDOW mode (--bounded): window capped at --cap / next "
              "disturbance; recovery = full coverage held to the window end.\n")
    if args.drop_censored:
        n_b1 = {k: sum(1 for e in v if e["recovery"] is None) for k, v in b1.items()}
        b1 = {k: [e for e in v if e["recovery"] is not None] for k, v in b1.items()}
        # A pair needs both endpoints to survive, so drop it if either is censored.
        n_b2 = sum(1 for e in b2 if e["marl"] is None or e["agentic"] is None)
        b2 = [e for e in b2 if e["marl"] is not None and e["agentic"] is not None]
        tag += "_nocensor"
        print("DROP-CENSORED mode (biased — discards the slow tail; see --help):")
        print(f"  B1 dropped: " + ", ".join(f"{k}={n}" for k, n in n_b1.items()))
        print(f"  B2 dropped: {n_b2} pair(s) with a censored endpoint")
        print()

    os.makedirs(args.out, exist_ok=True)
    p1 = os.path.join(args.out, f"full_retrack_b1_{args.b1_style}{tag}.png")
    p2 = os.path.join(args.out, f"full_retrack_b2_paired{tag}.png")
    sym1 = _plot_b1(b1, p1, args.b1_style)
    sym2 = _plot_b2(b2, p2)

    print(f"Saved {p1}" + ("  (symlog y)" if sym1 else ""))
    print(f"Saved {p2}" + ("  (symlog y)" if sym2 else ""))
    print()
    _print_report(b1, b2)


if __name__ == "__main__":
    main()
