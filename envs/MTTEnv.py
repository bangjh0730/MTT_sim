import numpy as np

from config.params import (
    NUM_UAVS, NUM_TARGETS, MAP_SIZE, T_SLOTS, DT,
    SNR_MIN, V_MAX_TARGET, R_MIN, MAX_TARGETS,
)
from components import BS, UAV, Target

# Broad initial EKF prior handed to the BS for EVERY target — the initial ones at
# reset and any target born mid-mission alike (position std 200 m, velocity std 5 m/s).
SIGMA0_INIT = np.diag([200.0**2, 200.0**2, 5.0**2, 5.0**2])
from envs.ekf import predict, update
from envs.ISAC import (measure, uplink_snr_batch, uplink_rate,
                       feasible_dwell_time, radar_snr, TAU_REF)
from envs.PCRLB import predict_bfim_batch, observation_info, pcrlb_batch
from envs.energy import slot_energy
from envs.rescue import apply_rescues, rescue_prob


class MTTEnv:
    """
    Multi-UAV Multi-Target Tracking simulation environment — OVERLOADED regime.

    There are fewer UAVs than targets (|K| > |U| is normal, not a failure), so the
    BS assigns each UAV a SET of targets and the UAV cycles its sensing among
    them. Targets leave only by being RESCUED, at a rate driven by how well they
    are currently tracked, and the mission objective is the average rescue delay.

    Per-slot loop:
      1. Resolve tau from actions; UAV kinematic step
      2. Fast-timescale sensing-target selection (highest tr(Sigma) in the set)
         -> ISAC sensing -> range-and-bearing measurement for that one target
      3. BS EKF predict + update per target (predict-only for unsensed targets,
         so their covariance grows — the backlog dynamic of this regime)
      4. RESCUE stage: each live target is rescued w.p. lambda/(lambda+tr Sigma)
      5. BFIM update, PCRLB computation (diagnostic)
      6. Target motion for next slot

    The agentic AI (detector / judge / planner) and MAPPO are NOT implemented
    here. Pass actions as {uav_id: (dvx, dvy, tau)}. Set tau=None to auto-select
    the midpoint of the feasible dwell-time interval.
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

        # ---- rescue-delay bookkeeping (the mission metric) ------------------
        self.rescue_delays:   list = []   # delay in slots, one per rescued target
        self.rescued_targets: list = []   # (id, birth_slot, rescue_slot, delay)
        self.backlog_sum:     int  = 0    # sum_t |K^t| — the numerator of Eq. (19)
        self.n_born_total:    int  = 0    # N — every target that ever appeared

        # Dedicated, per-target-id RNG streams for target process noise (see
        # Target.step) and for the rescue draw. A separate stream PER id (rather
        # than one shared stream drawn in live-id order) means target k's motion
        # and rescue draws depend only on how many times k itself has stepped —
        # never on whether some other target id happens to be alive that slot.
        # That matters more than ever now: rescue removes targets at
        # policy-dependent times, so a shared stream's draw order would differ
        # between modes and contaminate every other target.
        self._motion_rngs  = [np.random.default_rng() for _ in range(MAX_TARGETS)]
        self._rescue_rngs  = [np.random.default_rng() for _ in range(MAX_TARGETS)]

    def seed_motion(self, seed) -> None:
        """Reseed the per-target-id motion and rescue RNG streams.
        Call once before an eval run."""
        seqs = np.random.SeedSequence(seed).spawn(2 * MAX_TARGETS)
        self._motion_rngs = [np.random.default_rng(s) for s in seqs[:MAX_TARGETS]]
        self._rescue_rngs = [np.random.default_rng(s) for s in seqs[MAX_TARGETS:]]

    # ------------------------------------------------------------------
    def reset(self) -> dict:
        self.t = 0
        self.rescue_delays   = []
        self.rescued_targets = []
        self.backlog_sum     = 0
        self.n_born_total    = 0

        # BS at map center
        self.bs = BS(pos2d=[self.map_size / 2, self.map_size / 2])

        # Targets: uniformly random angles, minimum angular separation enforced so
        # targets don't cluster.  Evenly-spaced base angles caused a systematic
        # directional bias that the shared policy exploited.
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
            r     = np.random.uniform(self.map_size * 0.15, self.map_size * 0.45)
            pos   = np.clip(
                bs_center + r * np.array([np.cos(angle), np.sin(angle)]),
                0.05 * self.map_size, 0.95 * self.map_size,
            )
            spd = np.random.uniform(0.0, V_MAX_TARGET)
            ang = np.random.uniform(0.0, 2.0 * np.pi)
            vel = spd * np.array([np.cos(ang), np.sin(ang)])
            self.targets[k] = Target(target_id=k, init_pos=pos, init_vel=vel, birth_slot=0)
        self.n_born_total = len(self.targets)

        # UAV spawn: evenly-spaced angles around BS, shuffled so UAV id != spawn sector.
        spawn_angles = [
            2.0 * np.pi * i / self.num_uavs + np.random.uniform(-np.pi / self.num_uavs / 2,
                                                                   np.pi / self.num_uavs / 2)
            for i in range(self.num_uavs)
        ]
        np.random.shuffle(spawn_angles)

        spawn_positions = []
        for angle in spawn_angles:
            r_init = np.random.uniform(100.0, 150.0)
            pos = np.clip(
                bs_center + r_init * np.array([np.cos(angle), np.sin(angle)]),
                0.05 * self.map_size, 0.95 * self.map_size,
            )
            spawn_positions.append(pos)

        self.uavs = {}
        for i in range(self.num_uavs):
            self.uavs[i] = UAV(uav_id=i, init_pos=spawn_positions[i].tolist())

        # EKF and BFIM initialisation with broad uncertainty.
        self.ekf_state = {}
        self.bfim = {}
        for k, tgt in self.targets.items():
            self._register_estimate(k, tgt)

        # Initial partition of the targets across the UAVs. The old code matched
        # UAVs to targets one-to-one by bearing, which only makes sense when there
        # are spares; here every UAV must leave with several targets. Partition by
        # bearing sector, then balance the set sizes — see _initial_partition.
        self._initial_partition()
        self._compute_set_snr()   # prime, so slot 1's observation is not blank

        # Reset UAV located flags (diagnostic only — no longer gates observation or reward)
        for uav in self.uavs.values():
            uav.located = False

        return self._system_state()

    # ------------------------------------------------------------------
    def _initial_partition(self) -> None:
        """Balanced, geometry-aware initial partition of all live targets.

        Every target goes to exactly one UAV (no target starts unassigned — the
        agent may later choose to strand one, but the episode should not begin
        that way), and set sizes differ by at most one, so no UAV starts with a
        backlog the others do not share. Ties are broken by bearing from the BS,
        the same sector logic the one-to-one matcher used, so a UAV's initial set
        is at least spatially coherent before the targets wander apart.
        """
        bs_center = self.bs.pos

        def _ang(p):
            return np.arctan2(p[1] - bs_center[1], p[0] - bs_center[0])

        def _ang_dist(a, b):
            d = abs(a - b) % (2.0 * np.pi)
            return min(d, 2.0 * np.pi - d)

        uav_ids  = sorted(self.uavs)
        uav_ang  = {i: _ang(self.uavs[i].pos2d) for i in uav_ids}
        sets     = {i: set() for i in uav_ids}

        cap_hi   = int(np.ceil(len(self.targets) / max(len(uav_ids), 1)))
        # Farthest-first: the targets with the least ambiguous sector claim get
        # placed while every UAV still has room.
        order = sorted(self.targets, key=lambda k: -np.linalg.norm(self.targets[k].pos - bs_center))
        for k in order:
            a = _ang(self.targets[k].pos)
            room = [i for i in uav_ids if len(sets[i]) < cap_hi] or uav_ids
            i = min(room, key=lambda i: (_ang_dist(uav_ang[i], a), len(sets[i])))
            sets[i].add(k)

        self.apply_assignments(sets)

    # ------------------------------------------------------------------
    def apply_assignments(self, table: dict) -> None:
        """Install an assignment table {uav_id: iterable of target ids}.

        Filters out dead target ids and enforces that a target is held by at most
        one UAV (the table is a PARTITION of a subset of the live targets —
        duplicate holders would double-count the sensing capacity the allocator
        thinks it has).

        A live target absent from every set is left UNASSIGNED on purpose: under
        overload that is a real allocation choice, and silently patching it here
        would mean the repair rule, not the agent, was making the decision.
        """
        live  = set(self.targets)
        taken = set()
        clean = {}
        for i in sorted(self.uavs):
            ks = table.get(i, set()) or set()
            keep = set()
            for k in sorted({int(x) for x in ks if x is not None}):
                if k in live and k not in taken:
                    keep.add(k)
                    taken.add(k)
            self.uavs[i].set_assignment(keep)
            # set_assignment may truncate under the one-to-many ablation; release
            # whatever it dropped so another UAV can still take it.
            taken -= (keep - self.uavs[i].assignment_set)
            clean[i] = set(self.uavs[i].assignment_set)
        self.bs.set_assignments(clean)

    def assignment_table(self) -> dict:
        """Current assignment sets as {uav_id: set(target ids)}."""
        return {i: set(uav.assignment_set) for i, uav in self.uavs.items()}

    # ------------------------------------------------------------------
    def _pick_sensing_target(self, uav) -> int:
        """Fast-timescale selection: which ONE member of the set to sense now.

        A UAV takes a single radar measurement per slot, so holding a set of m
        targets means m-1 of them go unsensed every slot and run predict-only EKF
        steps, growing tr(Sigma) and losing rescue probability. That starvation is
        not a modelling wart — it is the entire cost of breadth, and it is what
        the assignment agent is actually trading against when it sizes a set.

        The rule is highest tr(Sigma_pos): the member whose estimate has decayed
        most since it was last measured. That is exactly the member whose rescue
        probability lambda/(lambda + tr Sigma) is lowest, so refreshing it buys
        the largest marginal gain in rescue rate; and because sensing collapses
        tr(Sigma) back toward the floor, the rule self-schedules into a round
        robin over the set without any explicit pointer or dwell counter.

        Note this is a FIXED scheduler, not a learned one: the MARL policy's job
        is where to FLY, not which member to look at. Those are separable because
        the scheduler's choice is fully determined by the covariances, which the
        policy influences only through the geometry it achieves.
        """
        members = [k for k in uav.assignment_set if k in self.targets]
        if not members:
            uav.sensing_target = None
            return None
        k = max(members, key=lambda k: float(np.trace(self.ekf_state[k][1][:2, :2])))
        uav.sensing_target = k
        return k

    # ------------------------------------------------------------------
    def _compute_set_snr(self) -> dict:
        """{uav: {member: SNR it would get pointing there from where it is}}.

        Evaluated at the NOMINAL dwell TAU_REF, not at the dwell resolved this
        slot, so the number answers "does my position serve this member" rather
        than "did I happen to spend enough dwell on it this slot". Only one member
        is actually measured per slot; the rest of this map is what tells the
        actor whether the members it is NOT serving right now are still within
        reach from here, or whether it has chased one target out to the edge of
        the map and stranded the others.
        """
        out = {}
        for i, uav in self.uavs.items():
            out[i] = {
                k: float(radar_snr(uav, self.ekf_state[k][0][:2], tau=TAU_REF))
                for k in uav.assignment_set if k in self.targets
            }
        self._set_snr_cache = out
        return out

    # ------------------------------------------------------------------
    def _register_estimate(self, k: int, tgt: Target) -> None:
        """Initialise the BS-side estimate for target k with the shared broad prior."""
        mu0 = tgt.state.copy()
        self.ekf_state[k] = (mu0, SIGMA0_INIT.copy())
        self.bs.add_target(k, mu0, SIGMA0_INIT.copy())
        self.bfim[k] = np.linalg.inv(SIGMA0_INIT)

    def spawn_target(self, init_state=None):
        """Birth a new target in a free id slot [0, MAX_TARGETS). Returns the new
        id, or None if all slots are occupied.

        The new target is NOT assigned to anyone: a birth is precisely the event
        the assignment agent exists to respond to, and auto-placing it here would
        make that decision mechanically instead.
        """
        free = [k for k in range(MAX_TARGETS) if k not in self.targets]
        if not free:
            return None
        k = free[0]

        if init_state is None:
            bs_center = np.array([self.map_size / 2, self.map_size / 2])
            angle = np.random.uniform(0.0, 2.0 * np.pi)
            r     = np.random.uniform(self.map_size * 0.15, self.map_size * 0.45)
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

        self.targets[k] = Target(target_id=k, init_pos=pos, init_vel=vel,
                                 birth_slot=self.t)
        self._register_estimate(k, self.targets[k])
        self.n_born_total += 1
        return k

    def remove_target(self, k: int) -> None:
        """Remove target k from the mission: drop its true state, estimate, BFIM,
        and clear it from every UAV's assignment set."""
        self.targets.pop(k, None)
        self.ekf_state.pop(k, None)
        self.bfim.pop(k, None)
        self.bs.remove_target(k)
        for uav in self.uavs.values():
            uav.drop_target(k)

    # ------------------------------------------------------------------
    @property
    def avg_rescue_delay(self) -> float:
        """D-bar = (Delta t / N) * sum_t |K^t|  — Eq. (19).

        Counted over TARGET-SLOTS, so a target still awaiting rescue at the end of
        the episode contributes the slots it has already waited rather than being
        dropped from the average. N is every target that ever appeared.
        """
        return DT * self.backlog_sum / max(self.n_born_total, 1)

    @property
    def mean_completed_delay(self) -> float:
        """Mean delay in SECONDS over the targets actually rescued. Complements
        avg_rescue_delay: this one ignores censored lives, so the two together
        separate 'rescues were fast' from 'few rescues happened'."""
        return DT * float(np.mean(self.rescue_delays)) if self.rescue_delays else 0.0

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
        info       : dict   -- diagnostics (rescue, backlog, PCRLB, SNR, ...)
        """
        self.t += 1

        uav_ids = sorted(self.uavs)

        # ---- 1. UAV kinematic step ----
        slot_energy_map: dict = {}
        for i, uav in self.uavs.items():
            dvx, dvy, _ = actions.get(i, (0.0, 0.0, None))
            e_cost = slot_energy(uav.speed)  # energy at pre-move speed
            slot_energy_map[i] = e_cost
            uav.step(dvx, dvy, uav.tau, e_cost)  # tau placeholder; overwritten below

        # ---- 2. Post-move: pick the sensing target, resolve tau, sense ----
        # Both tau_max and the rate gate use the same post-move uplink SNR, so
        # tau <= tau_max guarantees rate >= R_MIN whenever the channel supports it.
        measurements: dict = {k: [] for k in self.targets}
        snr_map: dict = {}
        rate_map: dict = {}
        sensed_now: set = set()

        _pos_post = np.stack([self.uavs[i].pos2d for i in uav_ids])
        gamma_map = dict(zip(uav_ids, uplink_snr_batch(_pos_post, self.bs.pos)))

        for i, uav in self.uavs.items():
            _, _, tau = actions.get(i, (0.0, 0.0, None))
            # One member of the set is measured this slot — the neediest one.
            k = self._pick_sensing_target(uav)
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
                sensed_now.add(k)
                self.targets[k].last_sensed_slot = self.t

        self._gamma_cache = gamma_map

        # ---- 3. EKF predict then sequential update ----
        # Targets nobody sensed this slot run a predict-only step, so their
        # covariance GROWS. With |K| > |U| that is the common case, and it is the
        # mechanism that turns assignment quality into rescue delay.
        mu_preds: dict = {}
        for k in self.targets:
            mu, Sigma = self.ekf_state[k]
            mu_pred, Sigma_pred = predict(mu, Sigma)
            mu_preds[k] = mu_pred

            mu_cur, Sigma_cur = mu_pred, Sigma_pred
            for z, R, uav_pos2d in measurements[k]:
                mu_cur, Sigma_cur = update(mu_cur, Sigma_cur, uav_pos2d, z, R)

            # Constrained EKF: the BS knows a target cannot move faster than
            # V_MAX_TARGET, so clamp the estimated speed (preserving heading).
            spd = float(np.hypot(mu_cur[2], mu_cur[3]))
            if spd > V_MAX_TARGET:
                mu_cur[2:] *= V_MAX_TARGET / spd

            self.ekf_state[k] = (mu_cur, Sigma_cur)
            self.bs.estimates[k] = (mu_cur, Sigma_cur)

        # ---- 4. Rescue stage ----
        # Runs on the post-update covariance, so a target measured this slot is
        # judged on its refreshed (small) tr(Sigma). Backlog is counted BEFORE the
        # removals: a target rescued at slot t did spend slot t waiting.
        backlog_before = len(self.targets)
        self.backlog_sum += backlog_before
        trace_pre = {k: float(np.trace(self.ekf_state[k][1][:2, :2])) for k in self.targets}
        rescued = apply_rescues(self)

        # ---- 5. BFIM update and PCRLB (diagnostic) ----
        target_keys  = list(self.targets.keys())
        if target_keys:
            J_pred_batch = predict_bfim_batch(np.stack([self.bfim[k] for k in target_keys]))
            for idx, k in enumerate(target_keys):
                I_total = np.zeros((4, 4))
                for i, uav in self.uavs.items():
                    if uav.sensing_target == k:
                        I_total += observation_info(uav.pos2d, mu_preds[k], snr_map.get(i, 0.0))
                self.bfim[k] = J_pred_batch[idx] + I_total

            pcrlb_arr        = pcrlb_batch(np.stack([self.bfim[k] for k in target_keys]))
            pcrlb_per_target = {k: float(pcrlb_arr[j]) for j, k in enumerate(target_keys)}
            avg_pcrlb        = float(np.mean(pcrlb_arr))
        else:
            pcrlb_per_target = {}
            avg_pcrlb        = 0.0

        # ---- Set reachability (after the EKF update and the rescue stage) ----
        # For every member of every UAV's set, the SNR that UAV WOULD get if it
        # pointed its radar there from where it is now. Only one member is
        # actually measured per slot; this is what says whether the UAV is parked
        # somewhere that can still serve the REST of its set, or has chased one
        # target out to the edge of the map and stranded the others. It is the
        # positioning signal the actor is trained on (marl/reward.py).
        #
        # Computed here, not right after sensing, so it reads the same estimates
        # and the same set membership that _system_state() is about to report:
        # rescued members are already gone and the EKF update has landed, so the
        # actor's observation and its reward describe one consistent world.
        set_snr = self._compute_set_snr()

        # ---- 6. Target motion ----
        for k, tgt in self.targets.items():
            tgt.step(map_size=self.map_size, rng=self._motion_rngs[k])

        assigned = {k for uav in self.uavs.values() for k in uav.assignment_set}
        info = {
            "t": self.t,
            # ---- mission metrics (overloaded regime) ----
            "backlog":            backlog_before,          # |K^t| before removals
            "rescued":            [k for k, _ in rescued], # ids rescued this slot
            "rescue_delays":      [d for _, d in rescued], # their delays, in slots
            "n_rescued_total":    len(self.rescue_delays),
            "avg_rescue_delay_s": self.avg_rescue_delay,   # Eq. (19), seconds
            "n_unassigned":       len(set(self.targets) - assigned),
            "load":               {i: self.uavs[i].load for i in self.uavs},
            "trace_pos_per_target":   trace_pre,
            "rescue_prob_per_target": {k: rescue_prob(v) for k, v in trace_pre.items()},
            "time_since_sensed": {
                k: self.t - self.targets[k].last_sensed_slot for k in self.targets
            },
            "sensed_now":        sorted(sensed_now),
            # {uav: {target: SNR it would get pointing there from here}} — the
            # positioning signal the actor is trained on (see marl/reward.py).
            "set_snr":           set_snr,
            # ---- tracking diagnostics ----
            "pcrlb": avg_pcrlb,
            "pcrlb_per_target": pcrlb_per_target,
            "snr_db": {
                i: 10 * np.log10(max(snr_map.get(i, 1e-10), 1e-10))
                for i in self.uavs
            },
            "snr_linear":    {i: snr_map.get(i, 0.0)             for i in self.uavs},
            "rate_Mbps":     {i: rate_map.get(i, 0.0) / 1e6      for i in self.uavs},
            "slot_energy_J": {i: slot_energy_map.get(i, 0.0)     for i in self.uavs},
            "energy_J":      {i: self.uavs[i].energy             for i in self.uavs},
            "rho":           {i: self.uavs[i].residual_energy    for i in self.uavs},
            "assignments":   {i: sorted(self.uavs[i].assignment_set) for i in self.uavs},
            "sensing":       {i: self.uavs[i].sensing_target     for i in self.uavs},
            "ekf_means":     {k: self.ekf_state[k][0][:2].copy() for k in self.targets},
            "sigma_pos_per_target": {
                k: float(np.sqrt(np.trace(self.ekf_state[k][1][:2, :2])))
                for k in self.targets
            },
        }
        return self._system_state(), info

    # ------------------------------------------------------------------
    def _system_state(self) -> dict:
        """Full system state S^t."""
        return {
            "t": self.t,
            "uavs": {i: uav.state.copy() for i, uav in self.uavs.items()},
            # One-to-many: a SET per UAV, plus the single member being sensed now.
            "assignments": {i: set(uav.assignment_set) for i, uav in self.uavs.items()},
            "sensing":     {i: uav.sensing_target      for i, uav in self.uavs.items()},
            "load":        {i: uav.load                for i, uav in self.uavs.items()},
            "gamma": dict(getattr(self, "_gamma_cache", {})),
            "rho":          {i: uav.residual_energy for i, uav in self.uavs.items()},
            "tau":          {i: uav.tau             for i, uav in self.uavs.items()},
            "located":      {i: uav.located         for i, uav in self.uavs.items()},
            "targets": {
                k: (self.ekf_state[k][0].copy(), self.ekf_state[k][1].copy())
                for k in self.targets
            },
            "time_since_sensed": {
                k: self.t - self.targets[k].last_sensed_slot for k in self.targets
            },
            "set_snr": {i: dict(v) for i, v in getattr(self, "_set_snr_cache", {}).items()},
            "birth_slot": {k: self.targets[k].birth_slot for k in self.targets},
            "backlog": len(self.targets),
        }
