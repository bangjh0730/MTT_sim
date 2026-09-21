"""
Per-event recovery distributions — the two headline figures.

  recovery_cdf.png   (Fig 1) "X% of disturbances recovered within N slots",
                     one step curve per policy, faceted by disturbance mode.
  recovery_strip.png (Fig 2) every single (target, event) recovery time as one
                     point, grouped by mode x policy, with median and p90 marked.

Both are built from evaluate.recovery_events, where one disturbance hitting one
target contributes exactly one number. That framing is what makes overlapping
disturbances safe to report: a fleet-average curve smears two events together
and blames the resulting slow tail on the policy, whereas here each event stands
on its own and the ones that were interfered with are flagged rather than
silently averaged in.

Why these replace the old mean-bar chart: a mean hides tails, and the tail is
the finding — the compound mode is the one the rule-based policy never trained
for, so what matters is not its typical response but the events it cannot
resolve at all. The CDF shows those as a curve that never reaches 1.0; the strip
shows them as points parked on the censored row. A bar of means shows neither.

Run:  python -m evaluate.recovery_time
      python -m evaluate.recovery_time --clean-only     # robustness check
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

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluate.recovery_events import (
    MODES, POLICIES, SENSED_THRESHOLD, CLEAN_HORIZON, INK, MUTED,
    load_all, outcomes, partition, censored_quantile, fmt_quantile,
)

RC = {
    "font.size": 11, "font.family": "sans-serif",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "axes.edgecolor": "#555555",
}


def _cdf_xy(vals: list):
    """Step-CDF points over every event. Censored (None) events stay in the
    denominator but contribute no step, so the curve plateaus below 1.0 at
    exactly the fraction that never recovered."""
    finite = sorted(v for v in vals if v is not None)
    n = len(vals)
    x = [0.0]
    y = [sum(1 for v in finite if v == 0) / n]   # y-intercept: never-degraded share
    for v in finite:
        if v == 0:
            continue
        x.append(v)
        y.append(y[-1] + 1.0 / n)
    return np.array(x), np.array(y)


def _plot_cdf(events: dict, clean_only: bool, out_path: str, xmax: int):
    fig, axes = plt.subplots(1, len(MODES), figsize=(12.2, 3.9),
                             sharex=True, sharey=True)
    for ax, (mlabel, mkey) in zip(np.atleast_1d(axes), MODES):
        cens_notes = []   # (color, text) for every policy that plateaus below 1.0
        for plabel, pkey, color, _ in POLICIES:
            vals = outcomes(events[(mkey, pkey)], clean_only)
            if not vals:
                continue
            x, y = _cdf_xy(vals)
            # Extend the last step to the axis edge so a plateau reads as a flat
            # ceiling ("this share never recovered"), not as a curve running out.
            x = np.append(x, xmax)
            y = np.append(y, y[-1])
            med, _ = censored_quantile(vals, 0.5)
            ax.step(x, y, where="post", color=color, lw=2.0,
                    label=f"{plabel} (n={len(vals)})", zorder=3)
            if med is not None:
                ax.plot([med], [np.interp(med, x, y)], "o", ms=5, color=color,
                        markeredgecolor="white", markeredgewidth=1.2, zorder=4)
            n_cens = sum(1 for v in vals if v is None)
            if n_cens:
                cens_notes.append((color, f"{n_cens}/{len(vals)} never recovered"))
        # A curve that stops short of 1.0 is the tail finding, not a rendering
        # artefact — say so on the plot. Both policies can plateau at the SAME
        # height (identical censored events, e.g. the same target died in both
        # runs), so stack the notes rather than drawing them at one shared xy
        # where the later one hides the earlier.
        for j, (color, txt) in enumerate(cens_notes):
            ax.annotate(txt, xy=(xmax, 1.0), xytext=(-6, -6 - 12 * j),
                        textcoords="offset points", ha="right", va="top",
                        fontsize=8, color=color)
        ax.set_title(mlabel, fontsize=11)
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, 1.02)
        ax.grid(ls=":", alpha=0.4, zorder=0)
        ax.legend(frameon=False, fontsize=9, loc="lower right")
        ax.set_xlabel("Slots since disturbance")
    np.atleast_1d(axes)[0].set_ylabel("Fraction of disturbances recovered")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_strip(events: dict, clean_only: bool, out_path: str, xmax: int):
    """One point per (target, event). Never-recovered events are drawn as open
    markers on a dedicated row above the axis break rather than dropped — they
    are the tail, and dropping them is what a mean would do."""
    jit = np.random.default_rng(0)
    rows = [(mi, pi) for mi in range(len(MODES)) for pi in range(len(POLICIES))]
    ypos = {rp: len(rows) - 1 - i for i, rp in enumerate(rows)}
    cens_x = xmax * 0.94   # censored marker lane, inside the axis

    fig, ax = plt.subplots(figsize=(11.0, 5.4))
    for mi, (mlabel, mkey) in enumerate(MODES):
        for pi, (plabel, pkey, color, dark) in enumerate(POLICIES):
            vals = outcomes(events[(mkey, pkey)], clean_only)
            y0 = ypos[(mi, pi)]
            finite = [v for v in vals if v is not None]
            n_cens = len(vals) - len(finite)

            jy = y0 + (jit.random(len(finite)) - 0.5) * 0.42
            ax.scatter(finite, jy, s=34, color=color, edgecolor="white",
                       linewidth=0.7, zorder=3, alpha=0.9)
            if n_cens:
                jy = y0 + (jit.random(n_cens) - 0.5) * 0.42
                ax.scatter([cens_x] * n_cens, jy, s=42, facecolor="none",
                           edgecolor=dark, linewidth=1.4, zorder=3)

            # Both labels sit ABOVE their marks: a p90 label hung below the row
            # collides with the mode-block separator underneath it, and median
            # and p90 are far enough apart in x not to collide with each other.
            med, med_b = censored_quantile(vals, 0.5)
            p90, p90_b = censored_quantile(vals, 0.90)
            if med is not None:
                ax.plot([med, med], [y0 - 0.30, y0 + 0.30], color=dark, lw=2.6,
                        solid_capstyle="butt", zorder=5)
                ax.text(med, y0 + 0.32, f"med {'>' if med_b else ''}{med:.0f}",
                        ha="center", va="bottom", fontsize=8, color=dark)
            if p90 is not None:
                ax.plot([p90, p90], [y0 - 0.24, y0 + 0.24], color=dark, lw=1.4,
                        ls=(0, (2, 1.4)), zorder=5)
                ax.text(p90, y0 + 0.32, f"p90 {'>' if p90_b else ''}{p90:.0f}",
                        ha="center", va="bottom", fontsize=8, color=MUTED)

    ax.axvline(cens_x, color=MUTED, ls="--", lw=0.9, zorder=1)
    ax.text(cens_x, len(rows) - 0.35, "never\nrecovered", ha="center", va="bottom",
            fontsize=8, color=MUTED, linespacing=1.15)

    ax.set_yticks([ypos[(mi, pi)] for mi, pi in rows])
    ax.set_yticklabels([f"{MODES[mi][0]}\n{POLICIES[pi][0]}" for mi, pi in rows],
                       fontsize=9)
    for mi in range(1, len(MODES)):     # separator between mode blocks
        ax.axhline(len(rows) - 2 * mi - 0.5, color="#dddddd", lw=0.8, zorder=0)
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.7, len(rows) - 0.05)
    ax.set_xlabel("Slots to recovery")
    # Ticks stop before the censored lane: a number printed to the right of the
    # break would read as a recovery time, which is the one thing those markers
    # explicitly are not.
    ax.set_xticks([t for t in range(0, xmax, 50) if t < cens_x - 20])
    ax.grid(axis="x", ls=":", alpha=0.4, zorder=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _print_table(events: dict, clean_only: bool):
    header = (f"{'Mode':14s} {'Policy':11s} {'n':>3s} {'never deg':>10s} "
              f"{'recovered':>10s} {'median':>7s} {'p90':>7s} {'max':>7s}")
    print(header)
    print("-" * len(header))
    for mlabel, mkey in MODES:
        for plabel, pkey, _, _ in POLICIES:
            ev   = events[(mkey, pkey)]
            p    = partition(ev, clean_only)
            vals = outcomes(ev, clean_only)
            fin  = [v for v in vals if v is not None]
            mx   = f"{max(fin):.0f}" if fin else "n/a"
            if p["n_censored"]:
                mx = f">{mx}"
            print(f"{mlabel:14s} {plabel:11s} {p['n_total']:>3d} "
                  f"{p['n_no_impact']:>4d}/{p['n_total']:<5d} "
                  f"{p['n_recovered']:>4d}/{p['n_total']:<5d} "
                  f"{fmt_quantile(vals, 0.5):>7s} {fmt_quantile(vals, 0.9):>7s} {mx:>7s}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Per-event recovery CDF and strip plot: rule-based vs agentic")
    parser.add_argument("--threshold", type=float, default=SENSED_THRESHOLD,
                        help="PCRLB (m^2) below which a target counts as recovered/sensed")
    parser.add_argument("--clean-only", action="store_true",
                        help="Keep only events with no second disturbance within "
                             f"{CLEAN_HORIZON} slots — a robustness check, not the "
                             "headline (the per-event CDF is already overlap-safe)")
    parser.add_argument("--xmax", type=int, default=260,
                        help="Right edge of the slots axis on both figures")
    parser.add_argument("-o", "--out", default="plots/paper")
    args = parser.parse_args()

    events = load_all(args.threshold)
    os.makedirs(args.out, exist_ok=True)
    plt.rcParams.update(RC)

    tag = "_clean" if args.clean_only else ""
    cdf   = os.path.join(args.out, f"recovery_cdf{tag}.png")
    strip = os.path.join(args.out, f"recovery_strip{tag}.png")
    _plot_cdf(events, args.clean_only, cdf, args.xmax)
    _plot_strip(events, args.clean_only, strip, args.xmax)

    print(f"Recovery: target's own PCRLB < {args.threshold:g} m^2 sustained, "
          f"measured from the slot the disturbance hit it")
    if args.clean_only:
        print(f"CLEAN EVENTS ONLY (no other disturbance within {CLEAN_HORIZON} slots)")
    print()
    _print_table(events, args.clean_only)
    print(f"Saved {cdf}")
    print(f"Saved {strip}")


if __name__ == "__main__":
    main()
