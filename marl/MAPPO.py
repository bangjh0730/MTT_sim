import os
import time
import numpy as np
import torch

from config.params import NUM_UAVS, NUM_TARGETS, V_MAX
from marl.actor_critic import MAPPOActor, CentralizedCritic
from marl.preprocess   import local_obs, global_state, OBS_DIM, GLOBAL_OBS_DIM
from marl.reward       import per_agent_rewards
from marl.train_plot   import plot_training_curves

DEFAULT_EPISODES = 20000   # canonical training episode count


# ---------------------------------------------------------------------------
class RolloutBuffer:
    """Stores one full episode of transitions for all agents."""

    def __init__(self):
        self.local_obs = []   # [T] of np [N, obs_dim]
        self.global_obs = []  # [T] of np [GLOBAL_OBS_DIM]  (centralized S^t)
        self.actions   = []   # [T] of np [N, 3]       (tanh-squashed)
        self.logprobs  = []   # [T] of np [N]
        self.values    = []   # [T] scalars            (centralized V)
        self.rewards   = []   # [T] of np [N]          (per-agent)
        self.dones     = []   # [T] bools

    def clear(self):
        self.__init__()


# ---------------------------------------------------------------------------
class MAPPO:
    """
    Multi-Agent PPO with Centralized Training Decentralized Execution (CTDE).

    Implements the Dec-POMDP:
      - Shared actor  πθ  acts on local obs  o^t_i
      - Centralized critic  Vφ  acts on global obs  S^t  (all agents concatenated)
      - Per-agent reward  r^t_i
      - Per-agent GAE; critic trained on mean team return
    """

    def __init__(
        self,
        num_uavs:   int   = NUM_UAVS,
        obs_dim:    int   = OBS_DIM,
        gamma:      float = 0.99,
        gae_lambda: float = 0.95,
        eps_clip:   float = 0.2,
        k_epochs:   int   = 15,   # Table II
        lr_actor:   float = 3e-4,
        lr_critic:  float = 3e-4,
        entropy_c:  float = 0.01,
        update_every_episodes: int = 10,
    ):
        self.num_uavs   = num_uavs
        self.gamma      = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip   = eps_clip
        self.k_epochs   = k_epochs
        self.entropy_c  = entropy_c

        self.update_every_episodes = update_every_episodes
        self.episode_count = 0

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")

        self.actor  = MAPPOActor(obs_dim).to(self.device)
        self.critic = CentralizedCritic(GLOBAL_OBS_DIM).to(self.device)

        self.actor_optim  = torch.optim.Adam(self.actor.parameters(),  lr=lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Compile to TorchScript after optimizer creation so optimizer param references
        # bind to the original tensors; scripted modules share those same tensor objects.
        self.actor  = torch.jit.script(self.actor)
        self.critic = torch.jit.script(self.critic)

        self.buffer = RolloutBuffer()

        self.accum_obs          = []
        self.accum_actions      = []
        self.accum_old_logprobs = []
        self.accum_adv          = []
        self.accum_returns      = []
        self.accum_g_obs        = []

        # Track history to return valid metrics when skipping training steps
        self.last_actor_loss  = 0.0
        self.last_critic_loss = 0.0

        self.ret_mean = 0.0
        self.ret_var  = 1.0
        self.count    = 1e-4

    # ------------------------------------------------------------------ rollout

    def select_actions(self, state: dict, info: dict = None,
                       deterministic: bool = False) -> dict:
        """
        Produce one action per UAV.
        `info` is the dict from the previous env.step().
        Pass info=None for the very first slot (after reset).
        Pass deterministic=True during eval to use mean actions (no sampling noise).

        Returns env-compatible actions: {uav_id: (dvx, dvy)}.
        Stores (obs, action, logprob, value) in the buffer.
        """
        all_l_obs_np = np.stack([local_obs(state, i, info) for i in range(self.num_uavs)])

        with torch.no_grad():
            all_l_obs_t = torch.tensor(all_l_obs_np, dtype=torch.float32).to(self.device)
            # Actor only — OBS_DIM=13 is fixed regardless of NUM_UAVS/NUM_TARGETS.
            actions_t, logprobs_t = self.actor.get_action(all_l_obs_t, deterministic=deterministic)

        action_arr  = actions_t.cpu().numpy()    # (N, ACT_DIM)
        logprob_arr = logprobs_t.cpu().numpy()   # (N,)

        if not deterministic:
            # Critic and rollout buffer are only needed during training.
            g_state = global_state(state, info)            # (GLOBAL_OBS_DIM,)
            with torch.no_grad():
                g_obs_t = torch.tensor(g_state[None, :], dtype=torch.float32).to(self.device)
                # Denormalize: critic predicts normalized returns ~N(0,1),
                # but GAE mixes it with raw rewards, so scale back to reward space.
                value   = self.critic(g_obs_t).item() * (np.sqrt(self.ret_var) + 1e-8) + self.ret_mean
            self.buffer.local_obs.append(all_l_obs_np)
            self.buffer.global_obs.append(g_state)
            self.buffer.actions.append(action_arr)
            self.buffer.logprobs.append(logprob_arr)
            self.buffer.values.append(value)

        # Scale tanh output [-1, 1] -> env units. The resulting speed is clamped
        # to V_MAX inside UAV.step, so constraint (20a) always holds.
        env_actions = {}
        for i in range(self.num_uavs):
            a = action_arr[i]
            env_actions[i] = (float(a[0]) * V_MAX,    # dvx in [-V_MAX, V_MAX]
                              float(a[1]) * V_MAX)    # dvy in [-V_MAX, V_MAX]
        return env_actions

    def store_outcome(self, rewards: np.ndarray, done: bool):
        """rewards: [N] per-agent rewards."""
        self.buffer.rewards.append(rewards.copy())
        self.buffer.dones.append(done)

    # ------------------------------------------------------------------ update

    def update(self):
        """
        PPO update over the stored episode.

        Per-agent GAE: each agent uses its own reward r^t_i with the shared
        centralized critic value V(S^t) as baseline.
        The critic is trained on the mean return across agents.

        Returns (actor_loss, critic_loss) as Python floats.
        """
        T = len(self.buffer.rewards)
        N = self.num_uavs

        # rewards_arr: [T, N]
        rewards_arr = np.stack(self.buffer.rewards)

        # ---- Per-agent GAE --------------------------------------------------
        advantages = np.zeros((T, N), dtype=np.float32)
        returns    = np.zeros((T, N), dtype=np.float32)

        for i in range(N):
            gae = 0.0
            for t in reversed(range(T)):
                next_val = self.buffer.values[t + 1] if t < T - 1 else 0.0
                not_done = 1.0 - float(self.buffer.dones[t])
                delta = (
                    rewards_arr[t, i]
                    + self.gamma * next_val * not_done
                    - self.buffer.values[t]
                )
                gae              = delta + self.gamma * self.gae_lambda * not_done * gae
                advantages[t, i] = gae
                returns[t, i]    = gae + self.buffer.values[t]

        # Critic trained on mean return across agents → [T]
        mean_returns = returns.mean(axis=1)
        mean_returns_repeated = np.repeat(mean_returns, N)

        self.accum_obs.append(np.stack(self.buffer.local_obs).reshape(T * N, -1))
        self.accum_actions.append(np.stack(self.buffer.actions).reshape(T * N, -1))
        self.accum_old_logprobs.append(np.stack(self.buffer.logprobs).reshape(T * N))
        self.accum_adv.append(advantages.reshape(-1))

        # Repeat the centralized state N times per step to align with the decentralized actors
        g_obs_stacked = np.stack(self.buffer.global_obs)   # (T, GLOBAL_OBS_DIM)
        self.accum_g_obs.append(np.repeat(g_obs_stacked, N, axis=0))

        # Appending the team return rather than conflicting individual returns
        self.accum_returns.append(mean_returns_repeated)

        # Clear the single-episode buffer for the next rollout
        self.buffer.clear()
        self.episode_count += 1

        # If macro-batch is incomplete, skip optimization and report previous step losses
        if self.episode_count < self.update_every_episodes:
            return self.last_actor_loss, self.last_critic_loss

        # ---- Tensors --------------------------------------------------------
        all_obs      = torch.tensor(np.concatenate(self.accum_obs), dtype=torch.float32).to(self.device)
        all_actions  = torch.tensor(np.concatenate(self.accum_actions), dtype=torch.float32).to(self.device)
        old_logprobs = torch.tensor(np.concatenate(self.accum_old_logprobs), dtype=torch.float32).to(self.device)
        g_obs_t      = torch.tensor(np.concatenate(self.accum_g_obs), dtype=torch.float32).to(self.device)

        flat_returns = np.concatenate(self.accum_returns)
        flat_adv     = np.concatenate(self.accum_adv)

        batch_mean  = np.mean(flat_returns)
        batch_var   = np.var(flat_returns)
        batch_count = len(flat_returns)

        # Welford chunked update: the all-time mean/var converge to fixed constants.
        # This stationarity is load-bearing — the critic predicts normalized returns
        # but its raw output is fed back into the GAE bootstrap, so the normalizer
        # must NOT drift (an EMA here destabilized and collapsed training ~ep 19k).
        delta    = batch_mean - self.ret_mean
        new_mean = self.ret_mean + delta * batch_count / (self.count + batch_count)
        m_a = self.ret_var * self.count
        m_b = batch_var * batch_count
        M2  = m_a + m_b + delta ** 2 * self.count * batch_count / (self.count + batch_count)

        self.ret_mean = new_mean
        self.ret_var  = M2 / (self.count + batch_count)
        self.count   += batch_count

        # Normalize target returns to a stable range
        norm_returns = (flat_returns - self.ret_mean) / (np.sqrt(self.ret_var) + 1e-8)
        returns_t    = torch.tensor(norm_returns, dtype=torch.float32).to(self.device)

        # Normalize advantages
        adv_flat = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)
        adv_t    = torch.tensor(adv_flat, dtype=torch.float32).to(self.device)

        # Clear macro buffers for the next cycle
        self.accum_obs.clear()
        self.accum_actions.clear()
        self.accum_old_logprobs.clear()
        self.accum_adv.clear()
        self.accum_returns.clear()
        self.accum_g_obs.clear()
        self.episode_count = 0

        # ---- PPO epochs -----------------------------------------------------
        for _ in range(self.k_epochs):
            logprobs, entropy = self.actor.evaluate(all_obs, all_actions)
            logprobs          = logprobs.view(-1)
            old_logprobs_flat = old_logprobs.view(-1)

            ratios = torch.exp((logprobs - old_logprobs_flat).clamp(-10.0, 10.0))

            surr1 = ratios * adv_t
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * adv_t
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_c * entropy.mean()

            # The critic network now attempts to predict normalized target scales
            values      = self.critic(g_obs_t)
            critic_loss = torch.nn.functional.huber_loss(values, returns_t, delta=1.0)

            self.actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optim.step()

            self.critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optim.step()

        self.last_actor_loss  = float(actor_loss.item())
        self.last_critic_loss = float(critic_loss.item())
        return self.last_actor_loss, self.last_critic_loss

    # ------------------------------------------------------------------ I/O

    def save(self, path: str = "./results"):
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(),  os.path.join(path, "marl_actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(path, "marl_critic.pth"))

    def load(self, path: str = "./results"):
        self.actor.load_state_dict(
            torch.load(os.path.join(path, "marl_actor.pth"), map_location=self.device)
        )
        try:
            self.critic.load_state_dict(
                torch.load(os.path.join(path, "marl_critic.pth"), map_location=self.device)
            )
        except RuntimeError:
            print("[MAPPO] Critic size mismatch (NUM_UAVS changed) — eval-only mode, actor loaded.")


# ---------------------------------------------------------------------------
# Live targets per training episode, drawn per episode. With 3 UAVs this spans
# ~1.3 to ~3.7 per UAV. The ceiling sits just above the observation padding
# (M = NUM_TARGETS) so sets rarely overflow it; overflow still reaches the actor
# through the set summary, but loses individual member positions.
TRAIN_POPULATION = (4, 11)


def _set_population(env, n_pop: int) -> None:
    """Bring the live-target count to n_pop at episode start."""
    for k in sorted(env.targets)[n_pop:]:
        env.remove_target(k)
        env.n_born_total -= 1
    while len(env.targets) < n_pop:
        if env.spawn_target() is None:
            break


def _replenish(env, n_pop: int) -> list:
    """Refill to the episode's population; the ids returned are ordinary births."""
    born = []
    while len(env.targets) < n_pop:
        k = env.spawn_target()
        if k is None:
            break
        born.append(k)
    return born


def MAPPO_run(env, num_episodes: int = DEFAULT_EPISODES, save_path: str = "./results"):
    """Dec-POMDP training loop. No LLM.

    Assignment sets come from TrainingPartitioner, a geometric surrogate for the
    allocator, randomised over style, reachability, size skew and untidiness so
    the actor stays neutral about who forms the sets.

    Arrivals are NOT the evaluation birth process: training holds a fixed
    live-target population per episode, replenishing on rescue. The evaluation
    queue's length depends on how well the fleet flies, so no single birth rate
    works across training - a rate that keeps a random policy out of saturation
    leaves a competent one with 84% empty sets, and vice versa. Holding the
    population fixed keeps the distribution of set sizes the same in episode 1
    and episode 20,000. Rescue still runs, so the loop is genuinely closed.
    """
    from marl.partitioner import TrainingPartitioner

    agent = MAPPO()
    rng   = np.random.default_rng()
    part  = TrainingPartitioner(rng)

    reward_hist        = []
    backlog_hist       = []     # mean |K^t| over the episode
    delay_hist         = []     # D-bar (Eq. 19), seconds
    rescued_hist       = []     # targets rescued per episode
    actor_loss_hist    = []
    critic_loss_hist   = []

    t_start = time.time()
    for ep in range(num_episodes):
        state     = env.reset()
        info      = None
        ep_reward = 0.0
        ep_backlog = []

        # This episode's live-target population, ~1.3 to ~3.7 per UAV.
        n_pop = int(rng.integers(TRAIN_POPULATION[0], TRAIN_POPULATION[1] + 1))
        _set_population(env, n_pop)

        # Draws this episode's partition style and lays down the opening sets.
        part.reset(env)
        state = env._system_state()

        for t in range(env.T):
            born     = _replenish(env, n_pop)
            rescued  = info["rescued"] if info is not None else []
            # Birth / rescue / hold-timer triggers. Between them the partition is
            # held while targets and UAVs drift, and serving that staleness is
            # most of what the actor is learning.
            if part.update(env, born=born, rescued=rescued):
                state = env._system_state()

            actions     = agent.select_actions(state, info)
            state, info = env.step(actions)

            rewards = per_agent_rewards(info, env.uavs)
            done    = (t == env.T - 1)
            agent.store_outcome(rewards, done)

            ep_reward += float(rewards.mean())
            ep_backlog.append(info["backlog"])

        al, cl = agent.update()

        reward_hist.append(ep_reward)
        backlog_hist.append(float(np.mean(ep_backlog)))
        delay_hist.append(float(env.avg_rescue_delay))
        rescued_hist.append(len(env.rescue_delays))
        actor_loss_hist.append(al)
        critic_loss_hist.append(cl)

        if (ep + 1) % 100 == 0:
            elapsed   = time.time() - t_start
            remaining = elapsed / (ep + 1) * (num_episodes - ep - 1)
            hh, mm    = divmod(int(remaining), 3600)
            mm, ss    = divmod(mm, 60)
            print(
                f"Episode {ep+1:4d} | "
                f"D={delay_hist[-1]:7.1f} s | backlog={backlog_hist[-1]:5.2f} | "
                f"rescued={rescued_hist[-1]:3d} | "
                f"Reward={reward_hist[-1]/env.T:6.3f} | "
                f"Actor={al:.4f} | Critic={cl:.4f} | "
                f"ETA {hh:02d}:{mm:02d}:{ss:02d}"
            )

    agent.save(save_path)
    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, "marl_reward.npy"),       reward_hist)
    np.save(os.path.join(save_path, "marl_delay.npy"),        delay_hist)
    np.save(os.path.join(save_path, "marl_backlog.npy"),      backlog_hist)
    np.save(os.path.join(save_path, "marl_rescued.npy"),      rescued_hist)
    np.save(os.path.join(save_path, "marl_actor_loss.npy"),   actor_loss_hist)
    np.save(os.path.join(save_path, "marl_critic_loss.npy"),  critic_loss_hist)

    plots_dir = os.path.join(save_path, "plots")
    plot_training_curves(
        reward_hist, delay_hist,
        actor_loss_hist, critic_loss_hist,
        plots_dir, num_episodes,
    )

    print(f"Training complete. Saved to {save_path}/")
    return agent
