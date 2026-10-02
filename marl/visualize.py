"""Roll out one episode with the trained policy and draw what happened.

    python -m marl.visualize --save ./results --seed 0 [--p-birth 0.1] [--gif]

No LLM: targets are assigned by the training partitioner, fixed to a clean
style (k-means, natural sizes, no noise) so the picture shows the policy, not
the partitioner's randomised difficulty draws. Writes to <save>/viz/:

  trajectories.png  the episode in four time windows - UAV paths and sensing
                    footprints, target paths coloured by the UAV holding them,
                    births, rescues and targets still waiting
  timeline.png      targets held per UAV over time, and one lifeline per target
                    from birth to rescue with its sensing events
  trajectories.gif  (with --gif) the same episode animated
"""

import argparse
import os

import numpy as np

from config.params import MAP_SIZE, DT, T_SLOTS_TRAIN, P_BIRTH

# ---- style: reference palette, light mode (categorical slots 1-3, all-pairs
# validated for 3 series; aqua is below 3:1 on the surface, so UAVs are also
# direct-labelled) ----
UAV_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
UNASSIGNED = "#898781"
SURFACE    = "#fcfcfb"
INK        = "#0b0b0b"
INK_2      = "#52514e"
MUTED      = "#898781"
GRID       = "#e1e0d9"
AXIS       = "#c3c2b7"
BAND       = "#f0efec"


# ---------------------------------------------------------------------------
def rollout(save_path, seed=0, T=T_SLOTS_TRAIN, deterministic=True):
    """One episode with the trained actor; returns a per-slot log."""
    import torch
    from envs.MTTEnv import MTTEnv
    from envs.births import apply_births
    from envs.schedule import DisturbanceSchedule
    from marl.MAPPO import MAPPO
    from marl.partitioner import TrainingPartitioner

    np.random.seed(seed)
    torch.manual_seed(seed)
    agent = MAPPO()
    agent.load(save_path)

    env = MTTEnv(t_slots=T)
    env.seed_motion(seed)
    part = TrainingPartitioner(np.random.default_rng(seed))
    env.reset()
    part.reset(env)
    part.mode, part.skew, part.noise_frac = "kmeans", "natural", 0.0
    part.resolve(env)
    schedule = DisturbanceSchedule(seed, T)

    U = env.num_uavs
    log = {
        "T": T, "U": U,
        "uav_pos":  np.zeros((T + 1, U, 2)),
        "tgt_pos":  {},     # uid -> {slot: (x, y)} true position
        "holder":   {},     # uid -> {slot: uav or -1}
        "sensed":   {},     # uid -> [slots with a delivered measurement]
        "born":     {},     # uid -> birth slot
        "rescued":  {},     # uid -> rescue slot
        "sensing":  np.full((T, U), -1),   # uid each UAV pointed at, per slot
        "held":     np.zeros((T, U), dtype=int),
    }
    uid_of = {}     # env id -> unique id (env ids are reused after a rescue)
    next_uid = [0]

    def uid(k):
        if k not in uid_of:
            uid_of[k] = next_uid[0]; next_uid[0] += 1
        return uid_of[k]

    for k in env.targets:
        log["born"][uid(k)] = 0
    for i in range(U):
        log["uav_pos"][0, i] = env.uavs[i].pos2d

    state, info = env._system_state(), None
    for t in range(T):
        born = apply_births(env, schedule)
        for k in born:
            uid_of.pop(k, None)
            log["born"][uid(k)] = t
        rescued = info["rescued"] if info is not None else []
        if part.update(env, born=born, rescued=rescued):
            state = env._system_state()

        holder = {k: i for i, s in env.assignment_table().items() for k in s}
        for k, tgt in env.targets.items():
            u = uid(k)
            log["tgt_pos"].setdefault(u, {})[t] = tgt.pos
            log["holder"].setdefault(u, {})[t] = holder.get(k, -1)
        for i in range(U):
            log["held"][t, i] = len(env.uavs[i].assignment_set)

        actions = agent.select_actions(state, info, deterministic=deterministic)
        agent.buffer = type(agent.buffer)()
        state, info = env.step(actions)

        for i in range(U):
            log["uav_pos"][t + 1, i] = env.uavs[i].pos2d
            k = info["sensing"].get(i)
            if k is not None:
                log["sensing"][t, i] = uid(k)
        for k in info["sensed_now"]:
            log["sensed"].setdefault(uid(k), []).append(t)
        for k in info["rescued"]:
            log["rescued"][uid(k)] = t
            uid_of.pop(k, None)

    log["D"] = env.avg_rescue_delay
    log["completed"] = env.mean_completed_delay
    log["n_born"] = env.n_born_total
    log["n_rescued"] = len(env.rescue_delays)
    return log


# ---------------------------------------------------------------------------
def _style_map(ax):
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, MAP_SIZE); ax.set_ylim(0, MAP_SIZE); ax.set_aspect("equal")
    ticks = np.arange(0, MAP_SIZE + 1, 500)
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels([f"{v/1000:g}" for v in ticks]); ax.set_yticklabels([f"{v/1000:g}" for v in ticks])
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(True, color=GRID, lw=0.6)
    for s in ax.spines.values():
        s.set_color(AXIS)


def _color(h):
    return UAV_COLORS[h] if h is not None and h >= 0 else UNASSIGNED


def plot_trajectories(log, out, n_windows=4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    from marl.preprocess import R_DETECT_2

    T, U = log["T"], log["U"]
    r_det = float(np.sqrt(R_DETECT_2))
    edges = np.linspace(0, T, n_windows + 1).astype(int)
    cols = 2 if n_windows > 1 else 1
    rows = int(np.ceil(n_windows / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 5.4 * rows), facecolor=SURFACE)
    axes = np.atleast_1d(axes).ravel()

    for w in range(n_windows):
        ax, a, b = axes[w], edges[w], edges[w + 1]
        _style_map(ax)

        # target paths, each segment in the colour of the UAV holding it then
        segs, cols_ = [], []
        for u, pos in log["tgt_pos"].items():
            slots = [s for s in range(a, b) if s in pos]
            for s0, s1 in zip(slots[:-1], slots[1:]):
                if s1 == s0 + 1:
                    segs.append([pos[s0], pos[s1]])
                    cols_.append(_color(log["holder"][u][s0]))
        ax.add_collection(LineCollection(segs, colors=cols_, linewidths=1.0, alpha=0.55, zorder=2))

        n_born = n_resc = n_wait = 0
        for u, pos in log["tgt_pos"].items():
            bs, rs = log["born"].get(u), log["rescued"].get(u)
            if bs is not None and a <= bs < b and bs in pos:
                h = log["holder"][u][bs]
                ax.plot(*pos[bs], "o", ms=6, mfc="none", mec=_color(h), mew=1.3, zorder=4)
                n_born += bs > 0 or a == 0
            if rs is not None and a <= rs < b and rs in pos:
                ax.plot(*pos[rs], "o", ms=7, mfc=_color(log["holder"][u][rs]), mec=SURFACE, mew=1.5, zorder=5)
                n_resc += 1
            last = b - 1
            if last in pos and (rs is None or rs >= b):
                ax.plot(*pos[last], "s", ms=5.5, mfc="none", mec=_color(log["holder"][u][last]), mew=1.3, zorder=4)
                n_wait += 1

        for i in range(U):
            p = log["uav_pos"][a:b + 1, i]
            c = UAV_COLORS[i]
            ax.plot(p[:, 0], p[:, 1], color=c, lw=2.0, zorder=6, solid_capstyle="round")
            ax.plot(*p[0], "o", ms=5, color=c, mec=SURFACE, mew=1.2, zorder=7)
            ax.plot(*p[-1], "^", ms=10, color=c, mec=SURFACE, mew=1.5, zorder=8)
            ax.add_patch(plt.Circle(p[-1], r_det, fill=False, ec=c, lw=1.0, ls=(0, (4, 3)), alpha=0.8, zorder=3))
            ax.annotate(f"UAV {i+1}", p[-1], xytext=(7, 7), textcoords="offset points",
                        fontsize=8, color=INK, fontweight="bold", zorder=9)

        ax.set_title(f"{a*DT:.0f}–{b*DT:.0f} s   born {n_born} · rescued {n_resc} · waiting {n_wait}",
                     fontsize=9.5, color=INK, loc="left")
    for ax in axes[n_windows:]:
        ax.axis("off")

    handles = [Line2D([], [], color=UAV_COLORS[i], lw=2, label=f"UAV {i+1} path (and targets it holds)") for i in range(U)]
    handles += [
        Line2D([], [], color=UNASSIGNED, lw=1, label="unassigned target"),
        Line2D([], [], ls="none", marker="o", ms=6, mfc="none", mec=INK_2, mew=1.3, label="target born"),
        Line2D([], [], ls="none", marker="o", ms=7, mfc=INK_2, mec=SURFACE, label="target rescued"),
        Line2D([], [], ls="none", marker="s", ms=5.5, mfc="none", mec=INK_2, mew=1.3, label="waiting at window end"),
        Line2D([], [], ls="none", marker="^", ms=9, color=INK_2, label="UAV at window end"),
        Line2D([], [], color=INK_2, lw=1, ls=(0, (4, 3)), label=f"sensing footprint ({r_det:.0f} m)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8.5,
               labelcolor=INK_2, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"Trained policy, one episode ({T*DT:.0f} s)   D = {log['D']:.1f} s · "
                 f"{log['n_rescued']} of {log['n_born']} targets rescued · map in km",
                 fontsize=11, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_timeline(log, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    T, U = log["T"], log["U"]
    t_s = np.arange(T) * DT
    order = sorted(log["tgt_pos"], key=lambda u: (log["born"][u], u))
    n = len(order)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 3 + min(0.022 * n, 9)), sharex=True,
                                   gridspec_kw={"height_ratios": [1, max(2.2, 0.009 * n)]},
                                   facecolor=SURFACE)

    # --- targets held per UAV, with the 2-4 per UAV band ---
    ax1.set_facecolor(SURFACE)
    from matplotlib.patches import Patch
    ax1.axhspan(2, 4, color=BAND, zorder=0)
    for i in range(U):
        ax1.plot(t_s, log["held"][:, i], color=UAV_COLORS[i], lw=1.6, label=f"UAV {i+1}")
    ax1.set_ylabel("targets held", color=INK_2, fontsize=9)
    ax1.set_ylim(0, max(6, log["held"].max() + 1))
    h1, _ = ax1.get_legend_handles_labels()
    ax1.legend(handles=h1 + [Patch(color=BAND, label="2–4 per UAV")], loc="upper left",
               bbox_to_anchor=(1.005, 1), frameon=False, fontsize=8.5, labelcolor=INK_2)

    # --- one lifeline per target, coloured by holder, ticks at sensing ---
    ax2.set_facecolor(SURFACE)
    segs, cols_ = [], []
    for y, u in enumerate(order):
        slots = sorted(log["tgt_pos"][u])
        start = slots[0]
        for a_, b_ in zip(slots, slots[1:] + [None]):
            # close a run where the holder changes or the lifeline ends
            if b_ is None or log["holder"][u][b_] != log["holder"][u][a_] or b_ != a_ + 1:
                segs.append([(start * DT, y), ((a_ + 1) * DT, y)])
                cols_.append(_color(log["holder"][u][a_]))
                start = b_
    ax2.add_collection(LineCollection(segs, colors=cols_, linewidths=1.6, capstyle="butt"))
    sx, sy = [], []
    for y, u in enumerate(order):
        for s in log["sensed"].get(u, []):
            sx.append((s + 0.5) * DT); sy.append(y)
    ax2.plot(sx, sy, "|", ms=3.5, mew=0.8, color=INK, alpha=0.55, zorder=3)
    rx, ry, rc = [], [], []
    for y, u in enumerate(order):
        rs = log["rescued"].get(u)
        if rs is not None:
            rx.append((rs + 1) * DT); ry.append(y); rc.append(_color(log["holder"][u].get(rs)))
    ax2.scatter(rx, ry, s=14, c=rc, edgecolors=SURFACE, linewidths=0.8, zorder=4)
    ax2.set_ylim(n, -1)
    ax2.set_ylabel("targets, in order of appearance", color=INK_2, fontsize=9)
    ax2.set_xlabel("time (s)", color=INK_2, fontsize=9)
    ax2.set_xlim(0, T * DT)

    for ax in (ax1, ax2):
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
        ax.grid(True, axis="x", color=GRID, lw=0.6)
        for s in ax.spines.values():
            s.set_color(AXIS)
    ax2.set_yticks([])

    handles = [Line2D([], [], color=UAV_COLORS[i], lw=2, label=f"held by UAV {i+1}") for i in range(U)]
    handles += [Line2D([], [], color=UNASSIGNED, lw=2, label="unassigned"),
                Line2D([], [], ls="none", marker="|", ms=7, color=INK, label="measured"),
                Line2D([], [], ls="none", marker="o", ms=6, color=INK_2, mec=SURFACE, label="rescued")]
    ax2.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.005, 1),
               frameon=False, fontsize=8.5, labelcolor=INK_2)
    ax1.set_title(f"D = {log['D']:.1f} s · mean delay of rescued targets "
                  f"{log['completed']:.1f} s · {log['n_rescued']} of {log['n_born']} rescued",
                  fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def animate(log, out, every=2, trail=60, fps=20):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from marl.preprocess import R_DETECT_2

    T, U = log["T"], log["U"]
    r_det = float(np.sqrt(R_DETECT_2))
    fig, ax = plt.subplots(figsize=(6, 6.3), facecolor=SURFACE)

    def frame(t):
        ax.clear(); _style_map(ax)
        for u, pos in log["tgt_pos"].items():
            if t not in pos:
                continue
            past = [pos[s] for s in range(max(0, t - trail), t + 1) if s in pos]
            h = log["holder"][u][t]
            if len(past) > 1:
                ax.plot(*np.array(past).T, color=_color(h), lw=0.8, alpha=0.45)
            ax.plot(*pos[t], "o", ms=5, color=_color(h), mec=SURFACE, mew=0.8, zorder=4)
        for u, rs in log["rescued"].items():
            if 0 <= t - rs < 8 and rs in log["tgt_pos"][u]:
                ax.plot(*log["tgt_pos"][u][rs], "o", ms=14, mfc="none",
                        mec=_color(log["holder"][u][rs]), mew=1.5, alpha=1 - (t - rs) / 8, zorder=5)
        for i in range(U):
            c, p = UAV_COLORS[i], log["uav_pos"][max(0, t - trail):t + 1, i]
            ax.plot(p[:, 0], p[:, 1], color=c, lw=2, alpha=0.9, zorder=6)
            ax.add_patch(plt.Circle(p[-1], r_det, fill=False, ec=c, lw=1, ls=(0, (4, 3)), alpha=0.7))
            k = log["sensing"][t, i] if t < T else -1
            if k >= 0 and t in log["tgt_pos"].get(k, {}):
                ax.plot(*np.array([p[-1], log["tgt_pos"][k][t]]).T, color=c, lw=1, alpha=0.9, zorder=5)
            ax.plot(*p[-1], "^", ms=10, color=c, mec=SURFACE, mew=1.5, zorder=8)
            ax.annotate(f"UAV {i+1}", p[-1], xytext=(7, 7), textcoords="offset points",
                        fontsize=8, color=INK, fontweight="bold", zorder=9)
        live = sum(1 for pos in log["tgt_pos"].values() if t in pos)
        done = sum(1 for rs in log["rescued"].values() if rs < t)
        ax.set_title(f"t = {t*DT:5.1f} s   live targets {live} · rescued {done}   (line: sensing)",
                     fontsize=9.5, color=INK, loc="left")

    anim = FuncAnimation(fig, frame, frames=range(0, T, every))
    anim.save(out, writer=PillowWriter(fps=fps), savefig_kwargs={"facecolor": SURFACE})
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Visualise one episode of the trained UAV policy")
    ap.add_argument("--save", default="./results", help="directory with marl_actor.pth")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--slots", type=int, default=T_SLOTS_TRAIN)
    ap.add_argument("--p-birth", type=float, default=None,
                    help="override P_BIRTH; defaults to the configured "
                         f"{P_BIRTH} from config.params")
    ap.add_argument("--stochastic", action="store_true", help="sample actions instead of the mean")
    ap.add_argument("--gif", action="store_true", help="also write an animation")
    ap.add_argument("--out", default=None, help="output directory (default <save>/viz)")
    args = ap.parse_args()

    if args.p_birth is not None:
        import envs.schedule as _sched
        _sched.P_BIRTH = args.p_birth
    out = args.out or os.path.join(args.save, "viz")
    os.makedirs(out, exist_ok=True)

    log = rollout(args.save, seed=args.seed, T=args.slots, deterministic=not args.stochastic)
    print(f"[viz] D={log['D']:.2f} s | rescued {log['n_rescued']}/{log['n_born']} | "
          f"mean completed delay {log['completed']:.2f} s")
    plot_trajectories(log, os.path.join(out, "trajectories.png"))
    plot_timeline(log, os.path.join(out, "timeline.png"))
    if args.gif:
        animate(log, os.path.join(out, "trajectories.gif"))
    print(f"[viz] saved to {out}/")


if __name__ == "__main__":
    main()
