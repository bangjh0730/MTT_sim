"""
Mechanism figure: annotated per-target PCRLB traces for representative compound
events, with the assignment decisions that drove them marked on the curve.

The CDF and strip figures (evaluate/recovery_time.py) establish THAT the agentic
planner recovers disturbances the rule-based policy does not, and the summary
table prices it. Neither shows WHY. This one does: each panel follows one
target's own PCRLB through one disturbance and marks every slot a UAV was
assigned to or pulled off that target, so the drop in the curve can be read
directly against the decision that caused it.

Panels are rows = event, columns = policy, both policies replaying the SAME
disturbance from the same seed (the schedule is pre-drawn per seed and shared —
see envs/schedule.py). The row is the same disturbance, but not always the same
target: which target a lost UAV was covering depends on the assignments that
policy had drifted to by then, so the two panels of a row can follow different
target ids. That is a real consequence of the policies diverging, not a mismatch
— each panel still shows what that disturbance cost that policy.

Events are chosen by rule-based recovery time, worst first (never-recovered
ranks above everything). That is deliberately NOT a neutral sample: the point of
the figure is to explain the tail, and the tail is where the rule-based policy
fails. The selection is stated in the caption rather than dressed up as typical
— the distributional claim is the CDF's job, not this figure's.

Run:  python -m evaluate.case_study
      python -m evaluate.case_study --n 4 --mode failure
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluate.recovery_events import (
    POLICIES, SENSED_THRESHOLD, MUTED, INK, load_events,
)

RC = {
    "font.size": 10, "font.family": "sans-serif",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "axes.edgecolor": "#555555",
}


def _tracker_changes(assignments: np.ndarray, target: int, lo: int, hi: int):
    """[(slot, uav, joined), ...] — every change to the set of UAVs assigned to
    `target` within [lo, hi], as 1-indexed slots. This is the policy's decision
    trail: `joined` False is a UAV leaving (reassigned elsewhere, or cleared by
    the BS when it declared that UAV lost)."""
    out = []
    for t in range(max(lo, 1), min(hi, assignments.shape[1] - 1) + 1):
        now  = {i for i in range(assignments.shape[0]) if assignments[i, t] == target}
        prev = {i for i in range(assignments.shape[0]) if assignments[i, t - 1] == target}
        for i in sorted(now - prev):
            out.append((t + 1, i, True))
        for i in sorted(prev - now):
            out.append((t + 1, i, False))
    return out


def _panel(ax, event, pre: int, post: int, thresh_y: float):
    d = np.load(event.path)
    series = d["pcrlb_per_target"][event.target]
    A = d["assignments"]
    color = next(c for _, p, c, _ in POLICIES if p == event.policy)
    dark  = next(c for _, p, _, c in POLICIES if p == event.policy)

    idx = event.onset - 1
    lo, hi = max(idx - pre, 0), min(idx + post, len(series) - 1)
    t = np.arange(lo - idx, hi - idx + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.log10(series[lo:hi + 1])

    ax.plot(t, y, color=color, lw=1.7, zorder=4)
    ax.axvline(0, color=MUTED, ls="--", lw=1, zorder=1)
    ax.axhline(thresh_y, color=MUTED, ls=":", lw=1, zorder=1)

    # Only mark recovery that actually happened inside the plotted window —
    # otherwise the marker floats past the end of its own curve and reads as a
    # stray data point.
    if event.recovery is not None and t[0] <= event.recovery <= t[-1]:
        ax.plot([event.recovery], [np.interp(event.recovery, t, y)], "o", ms=6,
                color=dark, markeredgecolor="white", markeredgewidth=1.2, zorder=6)

    # Assignment decisions. Changes are collapsed per slot (a swap is one
    # decision: a UAV leaves and another joins on the same slot) and labels are
    # staggered over three heights — reassignments bunch into the few slots
    # right after a disturbance, which on a 250-slot axis is a few pixels, so
    # two levels still overprint.
    changes = _tracker_changes(A, event.target, lo, hi)
    by_slot = {}
    for slot, uav, joined in changes:
        by_slot.setdefault(slot, []).append(f"{'+' if joined else '−'}U{uav}")
    ylo = np.nanmin(y)
    for j, (slot, tokens) in enumerate(sorted(by_slot.items())):
        x = slot - event.onset
        if not (t[0] <= x <= t[-1]):
            continue
        ax.axvline(x, color=dark, lw=0.8, alpha=0.35, zorder=2)
        ax.annotate(" ".join(tokens), xy=(x, ylo),
                    xytext=(0, 4 + 11 * (j % 3)), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7.5, color=dark,
                    bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none",
                              alpha=0.75), zorder=5)

    rec = "never" if event.recovery is None else f"{event.recovery} slots"
    ax.set_title(f"{event.policy} — T{event.target} → recovered: {rec}",
                 fontsize=9.5, color=INK)
    ax.grid(ls=":", alpha=0.35, zorder=0)


def _pair_key(e):
    return (e.seed, e.kind, e.onset)


def main():
    parser = argparse.ArgumentParser(
        description="Annotated per-target PCRLB traces for representative events")
    parser.add_argument("--mode", default="birth+fail",
                        help="Mode folder to draw case studies from")
    parser.add_argument("--n", type=int, default=3, help="Number of events (rows)")
    parser.add_argument("--pre", type=int, default=15)
    parser.add_argument("--post", type=int, default=245,
                        help="Wide enough to contain the rule-based policy's slow "
                             "recoveries (~230 slots) — a window that cut them off "
                             "would hide the very thing the panel is showing")
    parser.add_argument("-o", "--out", default="plots/paper")
    args = parser.parse_args()

    by_policy = {pkey: {_pair_key(e): e for e in load_events(args.mode, pkey)}
                 for _, pkey, _, _ in POLICIES}
    shared = sorted(set(by_policy["marl"]) & set(by_policy["agentic"]))

    # Drop events whose target DIED before either policy could recover it. Those
    # are censored, so they rank as "worst" and crowd out the whole figure, but
    # nothing about them is a policy outcome — the target ceased to exist, and a
    # panel of two flat unrecovered curves explains no mechanism.
    shared = [k for k in shared
              if not any(by_policy[p].get(k) and by_policy[p][k].censor == "death"
                         for _, p, _, _ in POLICIES)]
    if not shared:
        raise SystemExit(f"No recoverable events found under plots/{args.mode}/seed*/ "
                         f"for both policies.")

    # Worst rule-based recovery first; never-recovered sorts above every number.
    def rank(key):
        r = by_policy["marl"][key].recovery
        return (0, 0) if r is None else (1, -r)
    picks = sorted(shared, key=rank)[:args.n]

    plt.rcParams.update(RC)
    thresh_y = float(np.log10(SENSED_THRESHOLD))
    fig, axes = plt.subplots(len(picks), 2, figsize=(11.5, 2.7 * len(picks)),
                             sharex=True, sharey="row", squeeze=False)
    for row, key in enumerate(picks):
        for col, (_, pkey, _, _) in enumerate(POLICIES):
            _panel(axes[row][col], by_policy[pkey][key], args.pre, args.post, thresh_y)
        seed, kind, onset = key
        ev = by_policy["marl"][key]
        who = f"UAV {ev.uav} lost" if kind == "failure" else "target born"
        axes[row][0].set_ylabel(f"seed {seed}\n{who} @ slot {onset}\n"
                                r"$\log_{10}$ PCRLB (m$^2$)", fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("Slots since disturbance")
    fig.tight_layout()

    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, "case_study.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    print()
    print("+Un / -Un mark UAV n being assigned to / pulled off the plotted target.")
    print("Dot marks sustained recovery. Rows are the worst rule-based cases in "
          f"plots/{args.mode}/ (see module docstring).")
    print()
    for key in picks:
        for _, pkey, _, _ in POLICIES:
            e = by_policy[pkey][key]
            print(f"  seed{e.seed:<5d} {e.kind:7s} @{e.onset:<4d} {pkey:8s} "
                  f"target=T{e.target} recovery={e.recovery} clean={e.clean}")
        print()


if __name__ == "__main__":
    main()
