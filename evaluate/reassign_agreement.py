"""
How long the MARL (rule-based greedy) and Agentic (LLM) policies stay in
lockstep before they first diverge: per paired run, the number of
reassignment decisions the two agree on before the first one they don't.

Both modes are run on the same seed and start from the SAME deterministic
initial assignment (see evaluate/marl.py, evaluate/agentic.py), and
births/deaths/failures land on the same slots — so their per-slot assignment
vectors are directly comparable and identical at slot 0. They then reassign
independently as events unfold. We walk forward until the first slot where
their configs genuinely diverge, and count how many MARL reassignment
decisions occurred (all agreed on, by construction of "before divergence")
up to that point.

Timing is NOT required to match. MARL applies its greedy reassignment
instantly at the event slot, whereas Agentic's Planner Agent reasons for a
few slots (and the graph applies the result from a background thread), so the
SAME resulting config lands a few slots later in Agentic. A momentary
mismatch is therefore treated as agreement as long as the lagging policy
reaches the same config within WINDOW slots; only a config neither policy
reconciles counts as the divergence point.

"One joint value per run": divergence is a single shared event (the two share
a timeline until they split), so each paired run yields one count. MARL's
config-change slots are used as the decision timeline since MARL acts
instantly at each decision point.

Reads plots/<scenario>/seed<N>/{marl,agentic}/raw_eval.npz (needs the
`assignments` field). Renders plots/paper/reassign_agreement.png: one bar per
scenario (mean count), with per-seed dots.

Run:  python -m evaluate.reassign_agreement [--window N]
"""
import os
import glob
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (display label, folder). Matches evaluate/paper_pcrlb.py's scenario order.
SCENARIOS = [("Birth", "birth"), ("Failure", "failure"), ("Birth + Failure", "birth+fail")]

BLUE, BLUE_D = "#0072B2", "#004E7A"
INK, MUTED   = "#222222", "#666666"

# Slots of tolerance when deciding whether a momentary config mismatch is a
# real divergence or just the Agentic PA lagging MARL. Typical PA latency is
# ~4-5 slots (see evaluate/llm_latency); 15 leaves margin for the
# background-thread apply.
WINDOW = 15


def _reassign_slots(a: np.ndarray) -> np.ndarray:
    """Column indices t (>=1) where assignment vector a[:,t] != a[:,t-1]."""
    return np.where(np.any(a[:, 1:] != a[:, :-1], axis=0))[0] + 1


def _first_divergence(am: np.ndarray, aa: np.ndarray, window: int):
    """First slot where the two configs differ AND neither policy reconciles to
    the other's config within the next `window` slots (so it's a genuine split,
    not the lagging policy still catching up). None if they never split."""
    T = am.shape[1]
    for t in range(1, T):
        if np.array_equal(am[:, t], aa[:, t]):
            continue
        hi = min(T, t + window + 1)
        # Does the lagging policy reach the other's current config within window?
        if (aa[:, t:hi] == am[:, t][:, None]).all(axis=0).any():
            continue   # Agentic still catching up to MARL's config
        if (am[:, t:hi] == aa[:, t][:, None]).all(axis=0).any():
            continue   # (rare) MARL catching up to Agentic's config
        return t
    return None


def _run_counts(scenario_dir: str, window: int) -> list:
    """Per paired seed: (seed_label, n_agreed, first_div_slot_or_None). n_agreed
    = MARL reassignment decisions strictly before the first divergence."""
    out = []
    for f_marl in sorted(glob.glob(os.path.join("plots", scenario_dir, "seed*", "marl", "raw_eval.npz"))):
        seed_dir = os.path.dirname(os.path.dirname(f_marl))
        f_ag = os.path.join(seed_dir, "agentic", "raw_eval.npz")
        if not os.path.exists(f_ag):
            continue
        dm, da = np.load(f_marl), np.load(f_ag)
        if "assignments" not in dm.files or "assignments" not in da.files:
            continue
        am, aa = dm["assignments"], da["assignments"]
        T = min(am.shape[1], aa.shape[1])
        am, aa = am[:, :T], aa[:, :T]
        div = _first_divergence(am, aa, window)
        marl_changes = _reassign_slots(am)
        n_agreed = len(marl_changes) if div is None else int((marl_changes < div).sum())
        out.append((os.path.basename(seed_dir), n_agreed, div))
    return out


def _plot(data: dict, out_path: str):
    """data[scenario_key] = list of (seed_label, n_agreed, first_div)."""
    x     = np.arange(len(SCENARIOS))
    width = 0.55
    jit   = np.random.default_rng(0)

    plt.rcParams.update({
        "font.size": 11, "font.family": "sans-serif",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "axes.edgecolor": "#555555",
    })
    fig, ax = plt.subplots(figsize=(7.5, 4.6))

    means, counts = [], []
    for _, skey in SCENARIOS:
        vals = [n for _, n, _ in data[skey]]
        counts.append(vals)
        means.append(float(np.mean(vals)) if vals else 0.0)
    means = np.array(means)

    y_ext = max((max(v) if v else 0) for v in counts) or 1.0
    pad   = 0.05 * y_ext
    ax.bar(x, means, width, color=BLUE, edgecolor="white", linewidth=0.8, zorder=3)
    for xi, vals in zip(x, counts):
        jx = xi + (jit.random(len(vals)) - 0.5) * width * 0.5
        ax.scatter(jx, vals, s=22, color=BLUE_D, edgecolor="white", linewidth=0.6, zorder=5)
    for xi, mv in zip(x, means):
        ax.text(xi, max([mv] + [0]) + pad, f"{mv:.1f}", ha="center", va="bottom",
                fontsize=9, color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{lbl}\n(n={len(data[key])})" for lbl, key in SCENARIOS])
    ax.set_ylabel("Agreeing reassignments before first divergence")
    ax.set_ylim(0, y_ext + 6 * pad)
    ax.grid(axis="y", ls=":", alpha=0.4, zorder=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Agreeing reassignments before first MARL/Agentic divergence, per scenario")
    parser.add_argument("--window", type=int, default=WINDOW,
                         help="Slots of tolerance for the Agentic PA delay when deciding "
                              "whether a config mismatch is a real divergence")
    args = parser.parse_args()

    data = {skey: _run_counts(skey, args.window) for _, skey in SCENARIOS}

    os.makedirs("plots/paper", exist_ok=True)
    _plot(data, "plots/paper/reassign_agreement.png")
    print(f"Saved plots/paper/reassign_agreement.png  (delay tolerance = +/-{args.window} slots)\n")

    header = f"{'Scenario':18s} {'runs':>5s} {'mean':>6s} {'median':>7s} {'min':>4s} {'max':>4s}   per-run (n_agreed @ first_div_slot)"
    print(header)
    print("-" * len(header))
    for lbl, skey in SCENARIOS:
        runs = data[skey]
        vals = [n for _, n, _ in runs]
        detail = ", ".join(f"{n}@{'-' if d is None else d}" for _, n, d in runs)
        if vals:
            print(f"{lbl:18s} {len(runs):>5d} {np.mean(vals):>6.1f} {np.median(vals):>7.0f} "
                  f"{min(vals):>4d} {max(vals):>4d}   {detail}")
        else:
            print(f"{lbl:18s} {0:>5d}      -       -    -    -")


if __name__ == "__main__":
    main()
