import os
import numpy as np

from config.params import MAP_SIZE


def plot_trajectories(
    env,
    uav_traj:  dict,
    tgt_traj:  dict,
    plots_dir: str,
    slot:      int,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(8, 8))

    uav_colors = cm.tab10(np.linspace(0, 0.9, len(uav_traj)))
    tgt_colors = cm.Set2(np.linspace(0, 0.9, len(tgt_traj)))

    for k, color in zip(tgt_traj, tgt_colors):
        pts = np.array(tgt_traj[k], dtype=float)
        finite = ~np.isnan(pts[:, 0])
        if not finite.any():
            continue   # this id never held a live target in this run
        # NaN gaps break the line, so an id reused by a later target shows as a
        # separate segment; start/end markers use the first/last live position.
        ax.plot(pts[:, 0], pts[:, 1], color=color, lw=1.5, ls="--", label=f"Target {k}")
        fpts = pts[finite]
        ax.scatter(*fpts[0],  color=color, marker="s", s=60, zorder=5)
        ax.scatter(*fpts[-1], color=color, marker="X", s=80, zorder=5)

    for i, color in zip(uav_traj, uav_colors):
        pts = np.array(uav_traj[i])
        ax.plot(pts[:, 0], pts[:, 1], color=color, lw=1.5, label=f"UAV {i}")
        ax.scatter(*pts[0],  color=color, marker="o", s=60, zorder=5)
        ax.scatter(*pts[-1], color=color, marker="^", s=80, zorder=5)

    ax.scatter(*env.bs.pos, color="black", marker="*", s=200, zorder=6, label="BS")

    legend_extra = [
        Line2D([0], [0], marker="s", color="grey", ls="none", label="Start"),
        Line2D([0], [0], marker="X", color="grey", ls="none", label="End (target)"),
        Line2D([0], [0], marker="^", color="grey", ls="none", label="End (UAV)"),
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + legend_extra, labels + [e.get_label() for e in legend_extra],
              loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0,
              fontsize=7, ncol=1)

    ax.set_xlim(0, MAP_SIZE)
    ax.set_ylim(0, MAP_SIZE)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"UAV and Target Trajectories — slot {slot}")
    ax.grid(True, ls=":", alpha=0.4)

    fig.savefig(os.path.join(plots_dir, f"slot_{slot:04d}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_eval_rescue(backlog_log, rescued_cum_log, delay_log, unassigned_log,
                     plots_dir: str):
    """The mission panel: backlog, cumulative rescues, and running average delay.

    This replaces the PCRLB curve as the primary evaluation plot. PCRLB measured
    tracking accuracy on the premise that every target had a UAV on it; with
    |K| > |U| most targets are unsensed in any given slot by construction, so a
    coverage-era accuracy average no longer says whether the fleet is clearing
    its backlog. These three series do:

      backlog |K^t|  -- the objective itself. Eq. (19) is exactly Delta t / N
                        times the AREA UNDER THIS CURVE, so a lower curve is a
                        lower average rescue delay, and the shaded area is the
                        quantity being minimised.
      cumulative rescues -- throughput reads as the slope.
      running D-bar  -- the objective as it accumulates over the episode.

    Unassigned targets are shaded on the backlog axis: under overload leaving a
    target unheld is a legitimate allocation choice, but it is one whose cost
    should be visible rather than hidden inside the backlog total.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(plots_dir, exist_ok=True)
    slots = np.arange(1, len(backlog_log) + 1)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    ax = axes[0]
    ax.fill_between(slots, 0, backlog_log, color="steelblue", alpha=0.25)
    ax.plot(slots, backlog_log, color="steelblue", lw=1.5, label=r"backlog $|K^t|$")
    if unassigned_log is not None:
        ax.fill_between(slots, 0, unassigned_log, color="crimson", alpha=0.45,
                        label="held by nobody")
    from config.params import NUM_UAVS
    ax.axhline(NUM_UAVS, color="black", ls="--", lw=1.0,
               label=f"|U| = {NUM_UAVS} (above this line the fleet is oversubscribed)")
    ax.set_ylabel("targets awaiting rescue")
    ax.set_title(r"Backlog — shaded area $\times\,\Delta t/N$ is the average rescue delay")
    ax.legend(fontsize=8)
    ax.grid(True, ls=":", alpha=0.4)

    ax = axes[1]
    ax.plot(slots, rescued_cum_log, color="seagreen", lw=1.8)
    ax.set_ylabel("targets rescued (cumulative)")
    ax.set_title("Rescue throughput — the slope is the rescue rate")
    ax.grid(True, ls=":", alpha=0.4)

    ax = axes[2]
    ax.plot(slots, delay_log, color="darkorange", lw=1.8)
    ax.set_ylabel(r"$\bar{D}$ (s)")
    ax.set_xlabel("Slot")
    ax.set_title("Running average rescue delay (Eq. 19)")
    ax.grid(True, ls=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "eval_rescue.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_eval_pcrlb(pcrlb_log: list, pcrlb_per_target_log: dict, plots_dir: str):
    """PCRLB over the evaluation episode — kept as a secondary tracking
    diagnostic. It is no longer the objective (see plot_eval_rescue), but it
    still shows the tracking-accuracy mechanism that drives rescue: a target
    whose PCRLB climbs is one nobody is sensing, and its rescue probability is
    falling with it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    os.makedirs(plots_dir, exist_ok=True)
    slots = np.arange(1, len(pcrlb_log) + 1)

    fig, ax = plt.subplots(figsize=(10, 4))
    tgt_colors = cm.Set2(np.linspace(0, 0.9, len(pcrlb_per_target_log)))
    for (k, vals), color in zip(pcrlb_per_target_log.items(), tgt_colors):
        v = np.asarray(vals, dtype=float)
        if np.all(np.isnan(v)):
            continue
        ax.semilogy(slots, np.maximum(v, 1e-6), color=color, lw=1.0, alpha=0.85,
                    label=f"Target {k}")
    ax.semilogy(slots, np.maximum(pcrlb_log, 1e-6), color="black", lw=1.8, label="Average")

    ax.set_xlabel("Slot")
    ax.set_ylabel("PCRLB (m^2)")
    ax.set_title("Evaluation PCRLB (tracking diagnostic)")
    if len(pcrlb_per_target_log) <= 12:
        ax.legend(fontsize=8)
    ax.grid(True, ls=":", alpha=0.4, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "eval_pcrlb.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# def plot_eval_energy(energy_log: dict, plots_dir: str, failure_log: dict = None):
#     """Consumed energy (kJ) per UAV over the evaluation episode."""
#     import matplotlib
#     matplotlib.use("Agg")
#     import matplotlib.pyplot as plt
#     import matplotlib.cm as cm

#     os.makedirs(plots_dir, exist_ok=True)
#     failure_log = failure_log or {}

#     fig, ax = plt.subplots(figsize=(10, 4))
#     uav_ids   = sorted(energy_log.keys())
#     uav_colors = cm.tab10(np.linspace(0, 0.9, len(uav_ids)))
#     n_slots   = max(len(energy_log[i]) for i in uav_ids)
#     slots     = np.arange(1, n_slots + 1)

#     # Pad any shorter series (failed UAVs) to full length with their last value.
#     matrix = np.array([
#         np.pad(energy_log[i], (0, n_slots - len(energy_log[i])), mode="edge")
#         for i in uav_ids
#     ])

#     ax.stackplot(slots, matrix, labels=[f"UAV {i}" for i in uav_ids], colors=uav_colors, alpha=0.8)

#     # Mark failure slots at the top of the failed UAV's own band (cumulative up to that layer).
#     cumulative = np.cumsum(matrix, axis=0)
#     for idx, i in enumerate(uav_ids):
#         fi = failure_log.get(i)
#         if fi is not None and 0 < fi <= n_slots:
#             ax.scatter(fi, cumulative[idx, fi - 1], color="red", marker="X", s=60, zorder=6)

#     ax.set_xlabel("Slot")
#     ax.set_ylabel("Consumed energy (kJ)")
#     ax.set_title("Evaluation Consumed Energy")
#     ax.legend(fontsize=8, loc="upper left")
#     ax.grid(True, ls=":", alpha=0.4)
#     fig.tight_layout()
#     fig.savefig(os.path.join(plots_dir, "eval_energy.png"), dpi=150, bbox_inches="tight")
#     plt.close(fig)


def plot_mode_comparison(results: dict, plots_dir: str):
    """
    Compare evaluation modes on the mission metrics.

    results : {mode_name: the dict returned by eval_marl / eval_agentic}

    The modes are compared on backlog, cumulative rescues and running average
    delay rather than on PCRLB. Note what is and is not shared between the runs:
    births, target motion and measurement noise come from the same seed and are
    IDENTICAL, so both modes are handed the same arriving load. Rescues are not
    scheduled and will differ — that divergence is the result being measured, not
    a confound, which is why the rescue curves may separate from slot one.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from config.params import NUM_UAVS

    os.makedirs(plots_dir, exist_ok=True)
    colors = {"marl": "steelblue", "agentic": "darkorange", "greedy": "steelblue"}

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for mode, res in results.items():
        c = colors.get(mode)
        slots = np.arange(1, len(res["backlog"]) + 1)
        axes[0].plot(slots, res["backlog"], lw=1.4, color=c, label=mode)
        axes[1].plot(slots, res["rescued_cum"], lw=1.8, color=c, label=mode)
        axes[2].plot(slots, res["delay"], lw=1.8, color=c, label=mode)

    axes[0].axhline(NUM_UAVS, color="black", ls="--", lw=1.0,
                    label=f"|U| = {NUM_UAVS}")
    axes[0].set_ylabel(r"backlog $|K^t|$")
    axes[0].set_title("Backlog — lower is a shorter average rescue delay")
    axes[1].set_ylabel("rescued (cumulative)")
    axes[1].set_title("Rescue throughput")
    axes[2].set_ylabel(r"$\bar{D}$ (s)")
    axes[2].set_title("Running average rescue delay (Eq. 19)")
    axes[2].set_xlabel("Slot")
    for a in axes:
        a.legend(fontsize=9)
        a.grid(True, ls=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "rescue.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- completed-delay distribution -------------------------------------
    # The backlog curve can hide a mode that rescues a few targets very fast
    # while letting the rest sit. The per-target delay spread shows that.
    fig, ax = plt.subplots(figsize=(8, 4))
    data   = [np.asarray(results[m]["rescue_delays"], dtype=float) for m in results]
    labels = list(results)
    if any(len(d) for d in data):
        ax.boxplot([d if len(d) else np.array([np.nan]) for d in data], labels=labels)
    ax.set_ylabel("rescue delay (slots)")
    ax.set_title("Completed rescue delays per target")
    ax.grid(True, ls=":", alpha=0.4, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "delay_distribution.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def plot_multi_run(runs: list, plots_dir: str):
    """
    Overlay several eval runs' mission curves on shared axes.

    runs : [(label, npz_dict), ...] as loaded by evaluate.utils.load_raw_eval.

    Used for the lambda sweep, where the runs are NOT the same scenario: each
    lambda produces a different load regime, so the curves are compared for their
    SHAPE (does the backlog hold, drain or run away) rather than slot by slot.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    os.makedirs(plots_dir, exist_ok=True)
    colors = cm.viridis(np.linspace(0.05, 0.85, max(len(runs), 1)))

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for (label, d), c in zip(runs, colors):
        slots = np.arange(1, len(d["backlog"]) + 1)
        axes[0].plot(slots, d["backlog"], lw=1.3, color=c, label=label)
        if "rescued_cum" in d:
            axes[1].plot(slots, d["rescued_cum"], lw=1.6, color=c, label=label)
        if "delay" in d:
            axes[2].plot(slots, d["delay"], lw=1.6, color=c, label=label)

    from config.params import NUM_UAVS
    axes[0].axhline(NUM_UAVS, color="black", ls="--", lw=1.0,
                    label=f"|U| = {NUM_UAVS}")
    axes[0].set_ylabel(r"backlog $|K^t|$")
    axes[0].set_title("Backlog")
    axes[1].set_ylabel("rescued (cumulative)")
    axes[1].set_title("Rescue throughput")
    axes[2].set_ylabel(r"$\bar{D}$ (s)")
    axes[2].set_title("Running average rescue delay (Eq. 19)")
    axes[2].set_xlabel("Slot")
    for a in axes:
        a.legend(fontsize=8)
        a.grid(True, ls=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "multi_run.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
