"""
Publication figure: mean PCRLB, MARL vs Agentic, across disturbance modes.

Reads the per-run raw_eval.npz files (saved by main.py) laid out as
    plots/<mode>/seed<N>/<marl|agentic>/raw_eval.npz
aggregates each (mode, policy) over its seeds, and renders a clean grouped
bar chart (linear y, individual-seed dots, ±1 SD, B&W-safe texture) to
plots/paper/pcrlb_bar.{pdf,png}.

Run:  python -m evaluate.paper_pcrlb
"""
import os
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito CVD-safe pair (validated: chroma/contrast/CVD all pass).
BLUE, VERM = "#0072B2", "#D55E00"          # MARL, Agentic
BLUE_D, VERM_D = "#004E7A", "#8F3F00"      # darker shades for the seed dots
INK, MUTED = "#222222", "#666666"

# (display label, folder name under plots/). Order = left→right on the axis.
MODES = [("Birth", "birth"), ("Failure", "failure"), ("Birth + Failure", "birth+fail")]
POLICIES = [("MARL", "marl", BLUE, BLUE_D), ("Agentic", "agentic", VERM, VERM_D)]

SCALE = 1e3   # plot in units of 10^3 m^2 so ticks read 0,20,40,… not 0,20000,…


def _run_means(mode_dir: str, policy: str) -> list:
    """Per-seed mean PCRLB (in 10^3 m^2) for one mode/policy, over its seeds."""
    out = []
    for f in sorted(glob.glob(os.path.join("plots", mode_dir, "seed*", policy, "raw_eval.npz"))):
        out.append(float(np.mean(np.load(f)["pcrlb"])) / SCALE)
    return out


def main():
    # Gather stats.  data[mode][policy] = list of per-seed means.
    data = {m[1]: {p[1]: _run_means(m[1], p[1]) for p in POLICIES} for m in MODES}
    ns   = [len(data[m[1]]["agentic"]) for m in MODES]

    x     = np.arange(len(MODES))
    width = 0.36
    jit   = np.random.default_rng(0)

    plt.rcParams.update({
        "font.size": 11, "font.family": "sans-serif",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "axes.edgecolor": "#555555",
    })
    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    # Precompute stats so the layout (headroom, label heights) can be sized from
    # the actual marks — an SD whisker or a stray seed dot can sit far above the
    # bar mean. SD error bars are drawn only for n >= 3; with fewer seeds an SD is
    # statistically meaningless and would dominate the axis, and the individual
    # dots already show the spread.
    MIN_N_ERRBAR = 3
    stat = {}   # (mode_key, policy_key) -> (mean, sd_drawn, points, top_of_marks)
    for m in MODES:
        for _, pkey, _, _ in POLICIES:
            v  = data[m[1]][pkey]
            mn = float(np.mean(v))
            sd = float(np.std(v)) if len(v) >= MIN_N_ERRBAR else 0.0
            stat[(m[1], pkey)] = (mn, sd, v, max(mn + sd, max(v)))

    y_ext = max(s[3] for s in stat.values())     # highest mark anywhere
    pad   = 0.035 * y_ext

    for (plabel, pkey, face, dark), off, hatch in zip(
            POLICIES, (-width / 2, width / 2), ("///", None)):
        means = np.array([stat[(m[1], pkey)][0] for m in MODES])
        stds  = np.array([stat[(m[1], pkey)][1] for m in MODES])
        ax.bar(x + off, means, width, color=face, edgecolor="white", linewidth=0.8,
               hatch=hatch, label=plabel, zorder=3,
               yerr=stds, error_kw=dict(lw=1.1, ecolor=INK, capsize=4, zorder=4))
        # Individual seed points (jittered) so the spread is visible, not hidden.
        for xi, m in zip(x + off, MODES):
            vals = stat[(m[1], pkey)][2]
            jx = xi + (jit.random(len(vals)) - 0.5) * width * 0.55
            ax.scatter(jx, vals, s=20, color=dark, edgecolor="white",
                       linewidth=0.6, zorder=5)
        # Mean value labels, in ink, placed above whichever is higher — the error
        # bar cap or the top dot — so they never overlap a mark or float off-axis.
        for xi, m, mv in zip(x + off, MODES, means):
            ax.text(xi, stat[(m[1], pkey)][3] + pad, f"{mv:.1f}",
                    ha="center", va="bottom", fontsize=8.5, color=INK)

    # Reduction factor per mode, above the whole group (clears both bars' marks
    # and their value labels).
    for xi, m in zip(x, MODES):
        mu_marl = stat[(m[1], "marl")][0]
        mu_ag   = stat[(m[1], "agentic")][0]
        grp_top = max(stat[(m[1], "marl")][3], stat[(m[1], "agentic")][3])
        ax.text(xi, grp_top + 4 * pad, f"{mu_marl / mu_ag:.1f}×",
                ha="center", va="bottom", fontsize=10, color=MUTED, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{lbl}\n(n={n})" for (lbl, _), n in zip(MODES, ns)])
    ax.set_ylabel(r"Mean PCRLB  ($10^{3}\,\mathrm{m}^{2}$)")
    # Headroom sized to clear the highest label (factor labels sit at grp_top+4*pad);
    # no in-axes title — the paper caption serves that role.
    ax.set_ylim(0, y_ext + 7 * pad)
    ax.grid(axis="y", ls=":", alpha=0.4, zorder=0)
    ax.legend(frameon=False, loc="upper left", fontsize=10)

    os.makedirs("plots/paper", exist_ok=True)
    fig.tight_layout()
    # fig.savefig("plots/paper/pcrlb_bar.pdf", bbox_inches="tight")
    fig.savefig("plots/paper/pcrlb_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Console summary.
    print("Mode            n   MARL(10^3)  Agentic(10^3)  reduction")
    for lbl, key in MODES:
        mu_m, mu_a = np.mean(data[key]["marl"]), np.mean(data[key]["agentic"])
        print(f"  {lbl:14s} {len(data[key]['agentic'])}   {mu_m:9.1f}   {mu_a:11.1f}   {mu_m/mu_a:6.1f}x")
    # print("Saved plots/paper/pcrlb_bar.pdf and .png")


if __name__ == "__main__":
    main()
