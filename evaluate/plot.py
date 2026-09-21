import os
import numpy as np

from config.params import MAP_SIZE


def plot_trajectories(
    env,
    uav_traj:    dict,
    tgt_traj:    dict,
    plots_dir:   str,
    slot:        int,
    failure_log: dict = None,   # {uav_id: traj_index} physical failure slot
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.lines import Line2D

    failure_log = failure_log or {}

    fig, ax = plt.subplots(figsize=(8, 8))

    uav_colors = cm.tab10(np.linspace(0, 0.9, len(uav_traj)))
    tgt_colors = cm.Set2(np.linspace(0, 0.9, len(tgt_traj)))

    for k, color in zip(tgt_traj, tgt_colors):
        pts = np.array(tgt_traj[k], dtype=float)
        finite = ~np.isnan(pts[:, 0])
        if not finite.any():
            continue   # target never alive in this run (birth/death model)
        # NaN gaps break the line, so a slot reused by a later target shows as a
        # separate segment; start/end markers use the first/last live position.
        ax.plot(pts[:, 0], pts[:, 1], color=color, lw=1.5, ls="--", label=f"Target {k}")
        fpts = pts[finite]
        ax.scatter(*fpts[0],  color=color, marker="s", s=60, zorder=5)
        ax.scatter(*fpts[-1], color=color, marker="X", s=80, zorder=5)

    for i, color in zip(uav_traj, uav_colors):
        pts      = np.array(uav_traj[i])
        fail_idx = failure_log.get(i)   # trajectory index of failure, or None

        if fail_idx is not None and fail_idx < len(pts):
            # Draw only the active portion of the trajectory.
            ax.plot(pts[:fail_idx + 1, 0], pts[:fail_idx + 1, 1],
                    color=color, lw=1.5, label=f"UAV {i}")
            ax.scatter(*pts[0], color=color, marker="o", s=60, zorder=5)

            # Red X at the failure position with slot annotation.
            fp = pts[fail_idx]
            ax.scatter(fp[0], fp[1], color="red", marker="X",
                       s=80, zorder=8, linewidths=1)
            ax.annotate(
                f"UAV {i} failed\nslot {fail_idx}",
                xy=(fp[0], fp[1]), fontsize=6, color="red",
                xytext=(7, 5), textcoords="offset points",
            )
        else:
            ax.plot(pts[:, 0], pts[:, 1], color=color, lw=1.5, label=f"UAV {i}")
            ax.scatter(*pts[0],  color=color, marker="o", s=60, zorder=5)
            ax.scatter(*pts[-1], color=color, marker="^", s=80, zorder=5)

    ax.scatter(*env.bs.pos, color="black", marker="*", s=200, zorder=6, label="BS")

    legend_extra = [
        Line2D([0], [0], marker="s", color="grey", ls="none", label="Start"),
        Line2D([0], [0], marker="X", color="grey", ls="none", label="End (target)"),
        Line2D([0], [0], marker="^", color="grey", ls="none", label="End (UAV)"),
    ]
    if failure_log:
        legend_extra.append(
            Line2D([0], [0], marker="X", color="red", ls="none",
                   markersize=9, label="UAV failure")
        )

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

    fig.savefig(os.path.join(plots_dir, f"slot_{slot:04d}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_eval_pcrlb(pcrlb_log: list, pcrlb_per_target_log: dict, plots_dir: str,
                    failure_slots: list = None):
    """PCRLB over the evaluation episode — average plus per-target, log-scaled."""
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
            continue   # target never alive in this run (birth/death model)
        ax.semilogy(slots, np.maximum(v, 1e-6), color=color, lw=1.0, alpha=0.85,
                    label=f"Target {k}")
    ax.semilogy(slots, np.maximum(pcrlb_log, 1e-6), color="black", lw=1.8, label="Average")

    # Mark UAV failure slots as red X on the x-axis itself (x in data coords, y
    # pinned to the axes bottom via a blended transform), not on the curve.
    failure_slots = failure_slots or []
    xaxis_tf = ax.get_xaxis_transform()
    for s in failure_slots:
        if 1 <= s <= len(pcrlb_log):
            ax.scatter(s, 0.0, transform=xaxis_tf, color="red", marker="X",
                       s=70, zorder=6, clip_on=False)
    if failure_slots:
        ax.scatter([], [], color="red", marker="X", s=70, label="UAV failure")

    ax.set_xlabel("Slot")
    ax.set_ylabel("PCRLB (m²)")
    ax.set_title("Evaluation PCRLB")
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
    Compare evaluation modes on one axis each.

    results : {mode_name: eval_marl/eval_agentic's full return tuple}
      - avg PCRLB curve: one line per mode (log-scaled) — the first element.
      - total consumed energy curve: one line per mode (summed over all UAVs, kJ).

    Only the leading PCRLB series is read here; the rest of the tuple is ignored
    rather than unpacked by position, so adding a per-run log to the eval return
    doesn't break this plot.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(plots_dir, exist_ok=True)
    colors = {"marl": "steelblue", "agentic": "darkorange"}

    # ── Average PCRLB ─────────────────────────────────────────────────────────
    # No per-UAV failure markers here: the group failure model (envs/failure.py)
    # samples failures conditioned on each mode's own assignment state, so the
    # same seed produces different failure slots in marl vs agentic — overlaying
    # them on a shared axis would misleadingly suggest a synchronized comparison.
    fig, ax = plt.subplots(figsize=(10, 4))
    for mode, (pcrlb, *_rest) in results.items():
        slots = np.arange(1, len(pcrlb) + 1)
        ax.semilogy(slots, np.maximum(pcrlb, 1e-6), lw=1.6,
                    color=colors.get(mode), label=mode)
    ax.set_xlabel("Slot")
    ax.set_ylabel("Average PCRLB (m²)")
    ax.set_title("Average PCRLB — MARL vs Agentic")
    ax.legend(fontsize=9)
    ax.grid(True, ls=":", alpha=0.4, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "pcrlb.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Total consumed energy ─────────────────────────────────────────────────
    # fig, ax = plt.subplots(figsize=(10, 4))
    # for mode, (_pcrlb, energy, _fslots, _rmse, _bslots) in results.items():
    #     slots = np.arange(1, len(energy) + 1)
    #     ax.plot(slots, energy, lw=1.6, color=colors.get(mode), label=mode)
    # ax.set_xlabel("Slot")
    # ax.set_ylabel("Total consumed energy (kJ)")
    # ax.set_title("Total Consumed Energy — MARL vs Agentic")
    # ax.legend(fontsize=9)
    # ax.grid(True, ls=":", alpha=0.4)
    # fig.tight_layout()
    # fig.savefig(os.path.join(plots_dir, "energy.png"), dpi=150, bbox_inches="tight")
    # plt.close(fig)
