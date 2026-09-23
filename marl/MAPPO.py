import os
import time
import numpy as np
import torch

from config.params import NUM_UAVS, NUM_TARGETS, V_MAX
from marl.actor_critic import MAPPOActor, CentralizedCritic
from marl.preprocess   import (local_obs, critic_obs, pad_sets,
                               EGO_DIM, MEM_DIM, U_DIM, T_DIM)
from marl.reward       import per_agent_rewards
from marl.train_plot   import plot_training_curves

DEFAULT_EPISODES = 20000   # canonical training episode count


# ---------------------------------------------------------------------------
class RolloutBuffer:
    """One episode of transitions, flattened in (slot, agent) order.

    Member and target sets vary in size, so they are kept as lists of arrays
    and padded only when a batch is formed.
    """

    def __init__(self):
        self.ego      = []   # [T*N] of np [EGO_DIM]
        self.mem      = []   # [T*N] of np [|A_i|, MEM_DIM]
        self.raw      = []   # [T*N] of np [2]      pre-tanh velocity
        self.idx      = []   # [T*N] ints            sensing index, -1 if none
        self.logprobs = []   # [T*N] floats
        self.c_ego    = []   # [T*N] of np [EGO_DIM]
        self.c_uav    = []   # [T*N] of np [|U|, U_DIM]
        self.c_tgt    = []   # [T*N] of np [|K|, T_DIM]
        self.values   = []   # [T] of np [N]        per-agent V_i(S^t)
        self.rewards  = []   # [T] of np [N]        per-agent
        self.dones    = []   # [T] bools

    def clear(self):
        self.__init__()


# ---------------------------------------------------------------------------
class MAPPO:
    """
    Multi-Agent PPO with Centralized Training Decentralized Execution (CTDE).

    Implements the Dec-POMDP:
      - Shared actor  πθ  acts on local obs  o^t_i and outputs
        (dvx, dvy, which member of A_i to sense)
      - Centralized critic  V_i(S^t)  per agent, over the full state
      - Per-agent reward  r^t_i
      - Per-agent GAE against the per-agent value
    Both networks read targets as sets, so there is no ceiling on |A_i| or |K|.
    """

    def __init__(
        self,
        num_uavs:   int   = NUM_UAVS,
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

        # The PPO update (full batch over ~15k samples with padded sets) runs on
        # the GPU when there is one; rollout inference is a batch of |U| and is
        # faster on the CPU, so it uses CPU copies synced after every update.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cpu    = torch.device("cpu")

        self.actor  = MAPPOActor(EGO_DIM, MEM_DIM).to(self.device)
        self.critic = CentralizedCritic(EGO_DIM, U_DIM, T_DIM).to(self.device)
        self._actor_cpu  = MAPPOActor(EGO_DIM, MEM_DIM)
        self._critic_cpu = CentralizedCritic(EGO_DIM, U_DIM, T_DIM)
        self._sync_cpu()

        self.actor_optim  = torch.optim.Adam(self.actor.parameters(),  lr=lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.buffer = RolloutBuffer()
        self._accum: list = []   # finished episodes awaiting the macro-batch update

        # Track history to return valid metrics when skipping training steps
        self.last_actor_loss  = 0.0
        self.last_critic_loss = 0.0

        self.ret_mean = 0.0
        self.ret_var  = 1.0
        self.count    = 1e-4

    # ------------------------------------------------------------------ tensors

    def _sync_cpu(self):
        self._actor_cpu.load_state_dict({k: v.cpu() for k, v in self.actor.state_dict().items()})
        self._critic_cpu.load_state_dict({k: v.cpu() for k, v in self.critic.state_dict().items()})

    def _t(self, x, dtype=torch.float32, device=None):
        return torch.as_tensor(x, dtype=dtype, device=device or self.device)

    def _actor_batch(self, egos, mems, device=None):
        mem, mask = pad_sets(mems, MEM_DIM)
        return (self._t(np.stack(egos), device=device), self._t(mem, device=device),
                self._t(mask, torch.bool, device=device))

    def _critic_batch(self, egos, uavs, tgts, device=None):
        u, um = pad_sets(uavs, U_DIM)
        t, tm = pad_sets(tgts, T_DIM)
        return (self._t(np.stack(egos), device=device), self._t(u, device=device),
                self._t(um, torch.bool, device=device), self._t(t, device=device),
                self._t(tm, torch.bool, device=device))

    def _critic_obs_all(self, state):
        return list(zip(*[critic_obs(state, i) for i in range(self.num_uavs)]))

    def _values(self, c_ego, c_uav, c_tgt) -> np.ndarray:
        """Per-agent V_i(S^t) in reward units, (N,)."""
        with torch.no_grad():
            v = self._critic_cpu(*self._critic_batch(c_ego, c_uav, c_tgt, self.cpu)).numpy()
        # Denormalize: critic predicts normalized returns ~N(0,1),
        # but GAE mixes it with raw rewards, so scale back to reward space.
        return v * (np.sqrt(self.ret_var) + 1e-8) + self.ret_mean

    # ------------------------------------------------------------------ rollout

    def select_actions(self, state: dict, info: dict = None,
                       deterministic: bool = False) -> dict:
        """
        Produce one action per UAV.
        `info` is the dict from the previous env.step().
        Pass info=None for the very first slot (after reset).
        Pass deterministic=True during eval (mean velocity, most likely member).

        Returns env-compatible actions: {uav_id: (dvx, dvy, target id or None)}.
        Stores (obs, action, logprob, value) in the buffer.
        """
        egos, mems, ids = zip(*[local_obs(state, i, info) for i in range(self.num_uavs)])

        with torch.no_grad():
            act_t, raw_t, idx_t, lp_t = self._actor_cpu.get_action(
                *self._actor_batch(egos, mems, self.cpu), deterministic=deterministic)
        act = act_t.cpu().numpy()
        idx = idx_t.cpu().numpy()

        if not deterministic:
            # Critic and rollout buffer are only needed during training.
            c_ego, c_uav, c_tgt = self._critic_obs_all(state)
            b = self.buffer
            b.ego.extend(egos);   b.mem.extend(mems)
            b.raw.extend(raw_t.cpu().numpy()); b.idx.extend(idx.tolist())
            b.logprobs.extend(lp_t.cpu().numpy().tolist())
            b.c_ego.extend(c_ego); b.c_uav.extend(c_uav); b.c_tgt.extend(c_tgt)
            b.values.append(self._values(c_ego, c_uav, c_tgt))

        # Scale tanh output [-1, 1] -> env units. The resulting speed is clamped
        # to V_MAX inside UAV.step, so constraint (20a) always holds.
        env_actions = {}
        for i in range(self.num_uavs):
            k = ids[i][idx[i]] if idx[i] >= 0 else None
            env_actions[i] = (float(act[i, 0]) * V_MAX,    # dvx in [-V_MAX, V_MAX]
                              float(act[i, 1]) * V_MAX,    # dvy in [-V_MAX, V_MAX]
                              k)
        return env_actions

    def store_outcome(self, rewards: np.ndarray, done: bool):
        """rewards: [N] per-agent rewards."""
        self.buffer.rewards.append(rewards.copy())
        self.buffer.dones.append(done)

    # ------------------------------------------------------------------ update

    def update(self, last_state: dict = None, last_info: dict = None):
        """
        PPO update over the stored episode.

        Per-agent GAE: each agent uses its own reward r^t_i against its own
        value V_i(S^t), and the critic is trained on each agent's own return.

        The episode ends on a time limit, not a terminal state: the mission
        goes on and the critic is not shown t. So the last slot bootstraps
        from V_i(S^T) of `last_state` instead of 0; zeroing it would teach the
        critic a value collapse it cannot anticipate.

        Returns (actor_loss, critic_loss) as Python floats.
        """
        b = self.buffer
        T = len(b.rewards)
        N = self.num_uavs

        rewards_arr = np.stack(b.rewards)   # [T, N]
        values_arr  = np.stack(b.values)    # [T, N]
        if last_state is not None:
            last_values = self._values(*self._critic_obs_all(last_state))
        else:
            last_values = np.zeros(N, dtype=np.float32)
        next_values = np.concatenate([values_arr[1:], last_values[None, :]], axis=0)

        # ---- Per-agent GAE --------------------------------------------------
        advantages = np.zeros((T, N), dtype=np.float32)
        gae = np.zeros(N, dtype=np.float32)
        for t in reversed(range(T)):
            delta = rewards_arr[t] + self.gamma * next_values[t] - values_arr[t]
            gae   = delta + self.gamma * self.gae_lambda * gae
            advantages[t] = gae
        returns = advantages + values_arr

        # Row order (slot, agent) matches the buffer's flattened samples.
        self._accum.append(dict(
            ego=b.ego, mem=b.mem, raw=b.raw, idx=b.idx, logprobs=b.logprobs,
            c_ego=b.c_ego, c_uav=b.c_uav, c_tgt=b.c_tgt,
            adv=advantages.reshape(-1), ret=returns.reshape(-1)))

        # Clear the single-episode buffer for the next rollout
        self.buffer = RolloutBuffer()
        self.episode_count += 1

        # If macro-batch is incomplete, skip optimization and report previous step losses
        if self.episode_count < self.update_every_episodes:
            return self.last_actor_loss, self.last_critic_loss

        cat = lambda key: [x for ep in self._accum for x in ep[key]]
        actor_in  = self._actor_batch(cat("ego"), cat("mem"))
        critic_in = self._critic_batch(cat("c_ego"), cat("c_uav"), cat("c_tgt"))
        raw_t     = self._t(np.stack(cat("raw")))
        idx_t     = self._t(np.array(cat("idx")), torch.long)
        old_lp    = self._t(np.array(cat("logprobs")))
        flat_adv     = np.concatenate([ep["adv"] for ep in self._accum])
        flat_returns = np.concatenate([ep["ret"] for ep in self._accum])
        self._accum = []
        self.episode_count = 0

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
        returns_t = self._t((flat_returns - self.ret_mean) / (np.sqrt(self.ret_var) + 1e-8))

        # Normalize advantages
        adv_t = self._t((flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8))

        # ---- PPO epochs -----------------------------------------------------
        for _ in range(self.k_epochs):
            logprobs, entropy = self.actor.evaluate(*actor_in, raw_t, idx_t)

            ratios = torch.exp((logprobs - old_lp).clamp(-10.0, 10.0))

            surr1 = ratios * adv_t
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * adv_t
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_c * entropy.mean()

            # The critic network now attempts to predict normalized target scales
            values      = self.critic(*critic_in)
            critic_loss = torch.nn.functional.huber_loss(values, returns_t, delta=1.0)

            self.actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optim.step()

            self.critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optim.step()

        self._sync_cpu()
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
            print("[MAPPO] Critic mismatch — eval-only mode, actor loaded.")
        self._sync_cpu()


# ---------------------------------------------------------------------------
def MAPPO_run(env, num_episodes: int = DEFAULT_EPISODES, save_path: str = "./results"):
    """Dec-POMDP training loop. No LLM.

    Assignment sets come from TrainingPartitioner, a geometric surrogate for the
    allocator, randomised over style, reachability, size skew and untidiness so
    the actor stays neutral about who forms the sets.

    Targets follow the same process as evaluation: NUM_TARGETS at t = 0, then
    births with probability P_BIRTH per slot (Eq. 5) and rescues with p_r
    (Eq. 6). The live count is whatever those produce - no population is held
    and nothing is capped - so the policy trains on the load it will meet.
    A fresh birth schedule is drawn per episode.
    """
    from marl.partitioner import TrainingPartitioner
    from envs.births      import apply_births
    from envs.schedule    import DisturbanceSchedule

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

        schedule = DisturbanceSchedule(int(rng.integers(2**31)), env.T)

        # Draws this episode's partition style and lays down the opening sets.
        part.reset(env)
        state = env._system_state()

        for t in range(env.T):
            born     = apply_births(env, schedule)
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

        al, cl = agent.update(last_state=state, last_info=info)

        reward_hist.append(ep_reward / env.T)   # mean per-slot reward
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
                f"Reward={reward_hist[-1]:6.3f} | "
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
