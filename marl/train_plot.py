import os
import numpy as np


def _smooth(x: list, window: int) -> np.ndarray:
    """Centred moving average; edges handled by shrinking the window."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    half = window // 2
    for i in range(len(x)):
        lo = max(0, i - half)
        hi = min(len(x), i + half + 1)
        out[i] = x[lo:hi].mean()
    return out


def plot_training_curves(
    reward_hist: list,
    pcrlb_hist: list,
    actor_loss_hist: list,
    critic_loss_hist: list,
    plots_dir: str,
    num_episodes: int,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(plots_dir, exist_ok=True)
    episodes = np.arange(1, len(reward_hist) + 1)
    window   = max(1, num_episodes // 100)   # 1% of total episodes

    # ── 1. Reward curve ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    per_slot = np.asarray(reward_hist, dtype=np.float64)
    ax.plot(episodes, per_slot, color="steelblue", alpha=0.25, lw=0.6, label="Raw")
    ax.plot(episodes, _smooth(per_slot, window), color="steelblue", lw=1.8,
            label=f"Smoothed (w={window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean per-slot reward")
    ax.set_title("Training Reward")
    ax.legend(fontsize=9)
    ax.grid(True, ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "train_reward.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 2. PCRLB curve ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    pcrlb = np.asarray(pcrlb_hist, dtype=np.float64)
    ax.plot(episodes, pcrlb, color="darkorange", alpha=0.25, lw=0.6, label="Raw")
    ax.plot(episodes, _smooth(pcrlb, window), color="darkorange", lw=1.8,
            label=f"Smoothed (w={window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("PCRLB (m²)")
    ax.set_title("PCRLB — Tracking Phase (last 50 slots)")
    ax.legend(fontsize=9)
    ax.grid(True, ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "train_pcrlb.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 3. Actor & critic loss ────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    actor_l  = np.asarray(actor_loss_hist,  dtype=np.float64)
    critic_l = np.asarray(critic_loss_hist, dtype=np.float64)

    ax1.plot(episodes, actor_l, color="mediumvioletred", alpha=0.25, lw=0.6)
    ax1.plot(episodes, _smooth(actor_l, window), color="mediumvioletred", lw=1.8)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Loss")
    ax1.set_title("Actor Loss (PPO clip)")
    ax1.grid(True, ls=":", alpha=0.4)

    ax2.plot(episodes, critic_l, color="teal", alpha=0.25, lw=0.6)
    ax2.plot(episodes, _smooth(critic_l, window), color="teal", lw=1.8)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Loss")
    ax2.set_title("Critic Loss (Huber)")
    ax2.grid(True, ls=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "train_loss.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Training plots saved to {plots_dir}/")
