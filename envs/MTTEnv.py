import numpy as np

from config.params import (
    NUM_UAVS, NUM_TARGETS, MAP_SIZE, T_SLOTS,
    SNR_MIN, V_MAX_TARGET, R_MIN, N_FAIL_DETECT, MAX_TARGETS,
)
from components import BS, UAV, Target

# Broad initial EKF prior handed to the BS for EVERY target — the initial ones at
# reset and any target born mid-mission alike (position std 200 m, velocity std 5 m/s).
SIGMA0_INIT = np.diag([200.0**2, 200.0**2, 5.0**2, 5.0**2])
from envs.ekf import predict, update
from envs.ISAC import measure, uplink_snr_batch, uplink_rate, feasible_dwell_time
from envs.PCRLB import predict_bfim_batch, observation_info, pcrlb_batch
from envs.energy import slot_energy


class MTTEnv:
    """
    Multi-UAV Multi-Target Tracking simulation environment.

    Per-slot loop:
      1. Resolve tau from actions; UAV kinematic step
      2. ISAC sensing  -> range-and-bearing measurements
      3. BS EKF predict + sequential update per target
      4. BFIM update, PCRLB computation
      5. Target motion for next slot

    AAI (monitor / planner) and MAPPO are NOT implemented here.
    Pass actions as {uav_id: (delta_psi, dvx, dvy, tau)}.
    Set tau=None to auto-select the midpoint of the feasible dwell-time interval.
    """

    def __init__(self, num_uavs=NUM_UAVS, num_targets=NUM_TARGETS, map_size=MAP_SIZE, t_slots=T_SLOTS):
        self.num_uavs = num_uavs
        self.num_targets = num_targets
        self.map_size = map_size
        self.T = t_slots
        self.t = 0

        self.bs: BS = None
        self.uavs: dict = {}
        self.targets: dict = {}
        self.ekf_state: dict = {}   # {k: (mu [4], Sigma [4x4])}  -- updated by BS EKF each slot
        self.bfim: dict = {}        # {k: J [4x4]}

        # BS-side failure detection: counts consecutive slots with no uplink from each UAV.
        # The BS declares a UAV lost only after N_FAIL_DETECT silent slots.
        self._no_signal_count: dict = {}
        self._bs_known_active: dict = {}

        # Dedicated, per-target-id RNG streams for target process noise (see
        # Target.step). Defaults to fresh entropy (training); evaluation calls
        # seed_motion(seed) so target trajectories are identical across modes
        # regardless of how much of the global stream each policy's measurements/
        # actions consume per slot. A separate stream PER id (rather than one shared
        # stream drawn in live-id order) means target k's own motion depends only on
        # how many times k itself has stepped — never on whether some other target
        # id happens to be alive that slot, which would otherwise shift the shared
        # stream's draw order and contaminate every other target's motion.
        self._motion_rngs = [np.random.default_rng() for _ in range(MAX_TARGETS)]

    def seed_motion(self, seed) -> None:
        """Reseed the per-target-id motion RNG streams. Call once before an eval run."""
        seqs = np.random.SeedSequence(seed).spawn(MAX_TARGETS)
        self._motion_rngs = [np.random.default_rng(s) for s in seqs]

    # ------------------------------------------------------------------
    def reset(self) -> dict:
        self.t = 0

        # BS at map center
        self.bs = BS(pos2d=[self.map_size / 2, self.map_size / 2])

        # Targets: uniformly random angles, minimum angular separation enforced so
        # targets don't cluster.  Evenly-spaced base angles caused a systematic
        # directional bias (k=2 always southwest) that the shared policy exploited.
        bs_center = np.array([self.map_size / 2, self.map_size / 2])
        self.targets = {}
        min_sep = 2.0 * np.pi / self.num_targets / 2   # half the even-spacing gap
        angles  = []
        for _ in range(self.num_targets):
            for attempt in range(100):
                candidate = np.random.uniform(0.0, 2.0 * np.pi)
                if all(abs((candidate - a + np.pi) % (2 * np.pi) - np.pi) >= min_sep
                       for a in angles):
                    angles.append(candidate)
                    break
            else:
                angles.append(np.random.uniform(0.0, 2.0 * np.pi))  # fallback
        for k in range(self.num_targets):
            angle = angles[k]
            r     = np.random.uniform(self.map_size * 0.3, self.map_size * 0.6)
            pos   = np.clip(
                bs_center + r * np.array([np.cos(angle), np.sin(angle)]),
                0.05 * self.map_size, 0.95 * self.map_size,
            )
            spd = np.random.uniform(0.0, V_MAX_TARGET)
            ang = np.random.uniform(0.0, 2.0 * np.pi)
            vel = spd * np.array([np.cos(ang), np.sin(ang)])
            self.targets[k] = Target(target_id=k, init_pos=pos, init_vel=vel)

        # UAV spawn: evenly-spaced angles around BS, shuffled so UAV id != spawn sector.
        spawn_angles = [
            2.0 * np.pi * i / self.num_uavs + np.random.uniform(-np.pi / self.num_uavs / 2,
                                                                   np.pi / self.num_uavs / 2)
            for i in range(self.num_uavs)
        ]
        np.random.shuffle(spawn_angles)

        spawn_positions = []
        for angle in spawn_angles:
            r_init = np.random.uniform(200.0, 250.0)
            pos = np.clip(
                bs_center + r_init * np.array([np.cos(angle), np.sin(angle)]),
                0.05 * self.map_size, 0.95 * self.map_size,
            )
            spawn_positions.append(pos)

        # Angular assignment: match UAVs to targets by bearing from the BS, not by
        # distance. A target that happens to spawn close to the BS would otherwise
        # grab every nearby UAV; angular matching spreads UAVs by sector instead, so
        # each target is covered by the UAVs heading roughly in its direction.
        def _ang(p):
            return np.arctan2(p[1] - bs_center[1], p[0] - bs_center[0])

        uav_ang = [_ang(spawn_positions[i]) for i in range(self.num_uavs)]
        tgt_ang = [_ang(self.targets[k].pos) for k in range(self.num_targets)]

        def _ang_dist(a, b):
            d = abs(a - b) % (2.0 * np.pi)
            return min(d, 2.0 * np.pi - d)

        dist = np.array([
            [_ang_dist(uav_ang[i], tgt_ang[k]) for k in range(self.num_targets)]
            for i in range(self.num_uavs)
        ])
        assignment_list = [None] * self.num_uavs
        assigned_uavs   = set()
        for k in np.random.permutation(self.num_targets):
            available = [i for i in range(self.num_uavs) if i not in assigned_uavs]
            nearest_i = min(available, key=lambda i: dist[i][k])
            assignment_list[nearest_i] = k
            assigned_uavs.add(nearest_i)
        for i in range(self.num_uavs):
            if assignment_list[i] is None:
                assignment_list[i] = int(np.argmin(dist[i]))

        self.uavs = {}
        for i in range(self.num_uavs):
            k   = assignment_list[i]
            pos = spawn_positions[i]

            self.uavs[i] = UAV(
                uav_id=i,
                init_pos=pos.tolist(),
            )
            self.uavs[i].assignment = k

        # EKF and BFIM initialisation with broad uncertainty.
        # The BS broadcasts this initial estimate to all UAVs as their mission briefing.
        # From slot 1 onward the BS EKF is propagated every slot (predict step always runs;
        # update step runs only when a UAV delivers a valid measurement).
        self.ekf_state = {}
        self.bfim = {}
        for k, tgt in self.targets.items():
            self._register_estimate(k, tgt)

        # Reset UAV located flags (diagnostic only — no longer gates observation or reward)
        for uav in self.uavs.values():
            uav.located = False

        self._no_signal_count = {i: 0 for i in range(self.num_uavs)}
        self._bs_known_active = {i: True for i in range(self.num_uavs)}

        return self._system_state()

    # ------------------------------------------------------------------
    def _register_estimate(self, k: int, tgt: Target) -> None:
        """Initialise the BS-side estimate for target k with the shared broad prior.
        Used identically for the initial targets (reset) and any target born
        mid-mission (spawn_target)."""
        mu0 = tgt.state.copy()
        self.ekf_state[k] = (mu0, SIGMA0_INIT.copy())
        self.bs.add_target(k, mu0, SIGMA0_INIT.copy())
        self.bfim[k] = np.linalg.inv(SIGMA0_INIT)

    def spawn_target(self, init_state=None):
        """Birth a new target in a free id slot [0, MAX_TARGETS). Returns the new id,
        or None if all slots are occupied. Evaluation-only (see envs/birth_death).

        init_state = (x, y, vx, vy) supplies the spawn state from the pre-computed
        disturbance schedule so births are identical across modes; if None, a random
        state is drawn (from the global stream)."""
        free = [k for k in range(MAX_TARGETS) if k not in self.targets]
        if not free:
            return None
        k = free[0]

        if init_state is None:
            bs_center = np.array([self.map_size / 2, self.map_size / 2])
            angle = np.random.uniform(0.0, 2.0 * np.pi)
            r     = np.random.uniform(self.map_size * 0.3, self.map_size * 0.6)
            pos   = np.clip(
                bs_center + r * np.array([np.cos(angle), np.sin(angle)]),
                0.05 * self.map_size, 0.95 * self.map_size,
            )
            spd = np.random.uniform(0.0, V_MAX_TARGET)
            ang = np.random.uniform(0.0, 2.0 * np.pi)
            vel = spd * np.array([np.cos(ang), np.sin(ang)])
        else:
            pos = np.array(init_state[:2], dtype=float)
            vel = np.array(init_state[2:], dtype=float)

        self.targets[k] = Target(target_id=k, init_pos=pos, init_vel=vel)
        self._register_estimate(k, self.targets[k])
        return k

    def remove_target(self, k: int) -> None:
        """Remove target k from the mission: drop its true state, estimate, BFIM, and
        clear it from any UAV's assignment. Evaluation-only."""
        self.targets.pop(k, None)
        self.ekf_state.pop(k, None)
        self.bfim.pop(k, None)
        self.bs.remove_target(k)
        for uav in self.uavs.values():
            if uav.assignment == k:
                uav.assignment = None

    # ------------------------------------------------------------------
    def step(self, actions: dict) -> tuple:
        """
        Execute one simulation slot.

        Parameters
        ----------
        actions : {uav_id: (dvx, dvy, tau)}

        Returns
        -------
        next_state : dict   -- system state S^{t+1}
        info       : dict   -- diagnostics (PCRLB, SNR, rate, energy, ...)
        """
        self.t += 1

        active_ids = [i for i, uav in self.uavs.items() if uav.active]

        # ---- 1. UAV kinematic step ----
        # Tau doesn't affect kinematics (position/velocity); energy is based on pre-move speed.
        # Tau resolution is deferred until post-move so tau_max uses the same gamma as the
        # rate gate — eliminating the pre/post-move uplink SNR mismatch that caused spurious
        # EKF measurement exclusions whenever a UAV moved away from the BS.
        slot_energy_map: dict = {}
        for i, uav in self.uavs.items():
            if not uav.active:
                continue
            dvx, dvy, _ = actions.get(i, (0.0, 0.0, None))
            e_cost = slot_energy(uav.speed)  # energy at pre-move speed
            slot_energy_map[i] = e_cost
            uav.step(dvx, dvy, uav.tau, e_cost)  # tau placeholder; overwritten below

        # ---- 2. Post-move: resolve tau then compute sensing ----
        # Both tau_max and the rate gate now use the same post-move uplink SNR,
        # so tau <= tau_max guarantees rate >= R_MIN whenever the channel can support it.
        measurements: dict = {k: [] for k in self.targets}
        snr_map: dict = {}
        rate_map: dict = {}

        if active_ids:
            _pos_post = np.stack([self.uavs[i].pos2d for i in active_ids])
            gamma_map = dict(zip(active_ids, uplink_snr_batch(_pos_post, self.bs.pos)))
        else:
            gamma_map = {}

        for i, uav in self.uavs.items():
            if not uav.active:
                continue
            _, _, tau = actions.get(i, (0.0, 0.0, None))
            k = uav.assignment
            tgt_est = self.ekf_state[k][0][:2] if k is not None else uav.pos2d
            tau_min_i, tau_max_i = feasible_dwell_time(uav, tgt_est, self.bs, gamma=gamma_map[i])
            if tau is None:
                tau = (tau_min_i + tau_max_i) / 2.0
            uav.tau = float(np.clip(tau, tau_min_i, tau_max_i))

            if k is None:
                continue
            tgt_pos = self.targets[k].pos
            z, R, snr = measure(uav, tgt_pos)
            snr_map[i] = snr
            rate_map[i] = uplink_rate(uav, self.bs, gamma=gamma_map[i])
            if snr >= SNR_MIN and rate_map[i] >= R_MIN:
                measurements[k].append((z, R, uav.pos2d.copy()))
                uav.located = True

        self._gamma_cache = gamma_map

        # ---- 3. EKF predict then sequential update ----
        mu_preds: dict = {}
        for k in self.targets:
            mu, Sigma = self.ekf_state[k]
            mu_pred, Sigma_pred = predict(mu, Sigma)
            mu_preds[k] = mu_pred

            mu_cur, Sigma_cur = mu_pred, Sigma_pred
            for z, R, uav_pos2d in measurements[k]:
                mu_cur, Sigma_cur = update(mu_cur, Sigma_cur, uav_pos2d, z, R)

            # Constrained EKF: the BS knows a target cannot move faster than
            # V_MAX_TARGET, so clamp the estimated speed (preserving heading). The
            # near-singular range-bearing geometry when a UAV sits directly over its
            # target can otherwise kick the velocity estimate to physically
            # impossible values, which propagate through the predict step and blow up
            # the estimate (and the actor's normalised observation).
            spd = float(np.hypot(mu_cur[2], mu_cur[3]))
            if spd > V_MAX_TARGET:
                mu_cur[2:] *= V_MAX_TARGET / spd

            self.ekf_state[k] = (mu_cur, Sigma_cur)
            self.bs.estimates[k] = (mu_cur, Sigma_cur)

        # ---- 4. BFIM update and PCRLB ----
        # Batch BFIM prediction across all targets in one numpy call.
        target_keys  = list(self.targets.keys())
        J_pred_batch = predict_bfim_batch(np.stack([self.bfim[k] for k in target_keys]))

        for idx, k in enumerate(target_keys):
            I_total = np.zeros((4, 4))
            for i, uav in self.uavs.items():
                if uav.assignment == k and uav.active:
                    I_total += observation_info(uav.pos2d, mu_preds[k], snr_map.get(i, 0.0))
            self.bfim[k] = J_pred_batch[idx] + I_total

        # Batch PCRLB across all targets — one inv() call for all K matrices.
        pcrlb_arr        = pcrlb_batch(np.stack([self.bfim[k] for k in target_keys]))
        pcrlb_per_target = {k: float(pcrlb_arr[j]) for j, k in enumerate(target_keys)}
        avg_pcrlb        = float(np.mean(pcrlb_arr)) if target_keys else 0.0

        # ---- 5. Target motion ----
        # Each target id draws from its OWN dedicated stream (see seed_motion), so
        # target k's motion depends only on k's own step count — never on whether
        # some other target id is alive that slot — and matches across modes.
        for k, tgt in self.targets.items():
            tgt.step(map_size=self.map_size, rng=self._motion_rngs[k])

        # ---- 6. BS failure detection: count consecutive silent slots per UAV ----
        # When a UAV goes silent for N_FAIL_DETECT slots the BS declares it lost
        # and clears its assignment — it cannot be counted as a tracker anymore.
        active_set = set(active_ids)
        for i in self.uavs:
            if i in active_set:
                self._no_signal_count[i] = 0
            else:
                self._no_signal_count[i] = self._no_signal_count.get(i, 0) + 1
            was_known_active          = self._bs_known_active.get(i, True)
            self._bs_known_active[i]  = self._no_signal_count[i] < N_FAIL_DETECT
            if was_known_active and not self._bs_known_active[i]:
                self.uavs[i].assignment = None   # BS removes the lost UAV from its assignment table

        info = {
            "t": self.t,
            "pcrlb": avg_pcrlb,
            "pcrlb_per_target": pcrlb_per_target,
            "snr_db": {
                i: 10 * np.log10(max(snr_map.get(i, 1e-10), 1e-10))
                for i in self.uavs
            },
            "snr_linear":    {i: snr_map.get(i, 0.0)             for i in self.uavs},
            "rate_Mbps":     {i: rate_map.get(i, 0.0) / 1e6      for i in self.uavs},
            "slot_energy_J": {i: slot_energy_map.get(i, 0.0)     for i in self.uavs},
            "energy_J":      {i: self.uavs[i].energy              for i in self.uavs},
            "rho":           {i: self.uavs[i].residual_energy           for i in self.uavs},
            "assignments":     {i: self.uavs[i].assignment              for i in self.uavs},
            "ekf_means":       {k: self.ekf_state[k][0][:2].copy()      for k in self.targets},
            # EKF position uncertainty (std, m) — the filter's own belief, matching obs[12].
            "sigma_pos_per_target": {
                k: float(np.sqrt(np.trace(self.ekf_state[k][1][:2, :2])))
                for k in self.targets
            },
            "bs_known_active": dict(self._bs_known_active),
            "no_signal_count": dict(self._no_signal_count),
        }
        return self._system_state(), info

    # ------------------------------------------------------------------
    def _system_state(self) -> dict:
        """Full system state S^t."""
        return {
            "uavs": {i: uav.state.copy() for i, uav in self.uavs.items()},
            "assignments": {i: uav.assignment for i, uav in self.uavs.items()},
            "gamma": dict(getattr(self, "_gamma_cache", {})),
            "rho":          {i: uav.residual_energy for i, uav in self.uavs.items()},
            "tau":          {i: uav.tau             for i, uav in self.uavs.items()},
            "active":       dict(self._bs_known_active),
            "located":      {i: uav.located         for i, uav in self.uavs.items()},
            "targets": {
                k: (self.ekf_state[k][0].copy(), self.ekf_state[k][1].copy())
                for k in self.targets
            },
        }
