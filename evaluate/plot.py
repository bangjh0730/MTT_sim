import os
import numpy as np

from config.params import MAP_SIZE, NUM_UAVS


def plot_trajectories(env, uav_traj: dict, tgt_traj: dict, plots_dir: str, slot: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.lines import Line2D

    os.makedirs(plots_dir, exist_ok=True)
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
        ax.plot(pts[:, 0], pts[:, 1], color=color, lw=1.2, ls="--")
        fpts = pts[finite]
        ax.scatter(*fpts[0],  color=color, marker="s", s=40, zorder=5)
        ax.scatter(*fpts[-1], color=color, marker="X", s=55, zorder=5)

    for i, color in zip(uav_traj, uav_colors):
        pts = np.array(uav_traj[i])
        ax.plot(pts[:, 0], pts[:, 1], color=color, lw=1.6, label=f"UAV {i}")
        ax.scatter(*pts[0],  color=color, marker="o", s=60, zorder=5)
        ax.scatter(*pts[-1], color=color, marker="^", s=80, zorder=5)

    ax.scatter(*env.bs.pos, color="black", marker="*", s=200, zorder=6, label="BS")

    legend_extra = [
        Line2D([0], [0], ls="--", color="grey", label="Target path"),
        Line2D([0], [0], marker="s", color="grey", ls="none", label="Target start"),
        Line2D([0], [0], marker="X", color="grey", ls="none", label="Target end"),
        Line2D([0], [0], marker="^", color="grey", ls="none", label="UAV end"),
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + legend_extra, labels + [e.get_label() for e in legend_extra],
              loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0,
              fontsize=8)

    ax.set_xlim(0, MAP_SIZE)
    ax.set_ylim(0, MAP_SIZE)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"UAV and Target Trajectories — slot {slot}")
    ax.grid(True, ls=":", alpha=0.4)

    fig.savefig(os.path.join(plots_dir, f"slot_{slot:04d}.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def plot_eval_rescue(backlog_log, rescued_cum_log, delay_log, unassigned_log,
                     plots_dir: str):
    """Mission panel: backlog, cumulative rescues, running average delay.

    Eq. (19) is dt/N times the area under the backlog curve, so the shaded
    region is the quantity being minimised. Unassigned targets are shaded
    separately - a legitimate choice under overload, but one with a visible cost.
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
    ax.axhline(NUM_UAVS, color="black", ls="--", lw=1.0,
               label=f"|U| = {NUM_UAVS} (above this the fleet is oversubscribed)")
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
    fig.savefig(os.path.join(plots_dir, "eval_rescue.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def plot_multi_run(runs: list, plots_dir: str):
    """Overlay several runs on shared axes.

    runs : [(label, npz_dict), ...] from evaluate.utils.load_raw_eval. For the
    lambda sweep the runs are different regimes, so compare curve SHAPE rather
    than slot by slot.
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

    axes[0].axhline(NUM_UAVS, color="black", ls="--", lw=1.0, label=f"|U| = {NUM_UAVS}")
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
