import numpy as np

from config.params import (
    NUM_UAVS, NUM_TARGETS, MAP_SIZE, T_SLOTS, DT,
    SNR_MIN, V_MAX_TARGET, R_MIN, MAX_TARGETS,
)
from components import BS, UAV, Target
from envs.ekf import predict, update
from envs.ISAC import measure, uplink_snr_batch, uplink_rate, radar_snr
from envs.rescue import apply_rescues, rescue_prob

# Initial EKF prior for every target: position std 200 m, velocity std 5 m/s.
SIGMA0_INIT = np.diag([200.0**2, 200.0**2, 5.0**2, 5.0**2])


class MTTEnv:
    """Multi-UAV multi-target search and rescue, overloaded regime.

    Fewer UAVs than targets, so the BS assigns each UAV a SET and the UAV senses
    ONE member per slot. Unsensed members run predict-only EKF steps, their
    tr(Sigma) grows and their rescue probability falls - that starvation is the
    cost of breadth and what the allocator trades against. Targets leave only by
    being rescued; the objective is the average rescue delay, Eq. (19).

    Per slot: kinematics (1) -> sensing-target pick + measurement (7)-(10) ->
    EKF (14)-(18) -> rescue (6) -> target motion (2).

    Actions are {uav_id: (dvx, dvy)}. The dwell split is fixed, not a control.
    """

    def __init__(self, num_uavs=NUM_UAVS, num_targets=NUM_TARGETS,
                 map_size=MAP_SIZE, t_slots=T_SLOTS):
        self.num_uavs    = num_uavs
        self.num_targets = num_targets
        self.map_size    = map_size
        self.T = t_slots
        self.t = 0

        self.bs: BS = None
        self.uavs: dict = {}
        self.targets: dict = {}
        self.ekf_state: dict = {}   # {k: (mu[4], Sigma[4x4])}

        self.rescue_delays:   list = []   # slots, one per rescued target
        self.rescued_targets: list = []   # (id, birth, rescue, delay)
        self.backlog_sum:     int  = 0    # sum_t |K^t|, numerator of Eq. (19)
        self.n_born_total:    int  = 0    # N

        # p_r of each live target at the end of the previous slot; the reward is
        # the per-slot improvement, so it has to survive the overwrite.
        self._prev_pr: dict = {}

        # Per-id RNG streams so a target's motion and rescue draws depend only on
        # its own lifetime, not on which other ids are alive. Matters because
        # rescue removes targets at policy-dependent times.
        self._motion_rngs = [np.random.default_rng() for _ in range(MAX_TARGETS)]
        self._rescue_rngs = [np.random.default_rng() for _ in range(MAX_TARGETS)]

    def seed_motion(self, seed) -> None:
        seqs = np.random.SeedSequence(seed).spawn(2 * MAX_TARGETS)
        self._motion_rngs = [np.random.default_rng(s) for s in seqs[:MAX_TARGETS]]
        self._rescue_rngs = [np.random.default_rng(s) for s in seqs[MAX_TARGETS:]]

    # ------------------------------------------------------------------
    def _random_target_state(self):
        """Spawn state: position uniform over the area, heading and speed uniform."""
        pos = np.random.uniform(0.05 * self.map_size, 0.95 * self.map_size, size=2)
        spd = np.random.uniform(0.0, V_MAX_TARGET)
        ang = np.random.uniform(0.0, 2.0 * np.pi)
        return pos, spd * np.array([np.cos(ang), np.sin(ang)])

    def reset(self) -> dict:
        self.t = 0
        self.rescue_delays   = []
        self.rescued_targets = []
        self.backlog_sum     = 0
        self.n_born_total    = 0
        self._prev_pr        = {}

        self.bs = BS(pos2d=[self.map_size / 2, self.map_size / 2])

        self.targets = {}
        for k in range(self.num_targets):
            pos, vel = self._random_target_state()
            self.targets[k] = Target(k, pos, vel, birth_slot=0)
        self.n_born_total = len(self.targets)

        bs_center = self.bs.pos
        self.uavs = {}
        for i in range(self.num_uavs):
            ang = np.random.uniform(0.0, 2.0 * np.pi)
            r   = np.random.uniform(100.0, 150.0)
            pos = np.clip(bs_center + r * np.array([np.cos(ang), np.sin(ang)]),
                          0.05 * self.map_size, 0.95 * self.map_size)
            self.uavs[i] = UAV(i, pos.tolist())

        self.ekf_state = {}
        for k, tgt in self.targets.items():
            self._register_estimate(k, tgt)

        self._initial_partition()
        self._compute_set_snr()
        return self._system_state()

    def _initial_partition(self) -> None:
        """Every target to its nearest UAV under an equal-share cap, so no
        target starts unassigned and no UAV opens with an unshared backlog."""
        uav_ids = sorted(self.uavs)
        sets    = {i: set() for i in uav_ids}
        cap_hi  = int(np.ceil(len(self.targets) / max(len(uav_ids), 1)))
        centre  = self.bs.pos

        for k in sorted(self.targets,
                        key=lambda k: -float(np.linalg.norm(self.targets[k].pos - centre))):
            p = self.targets[k].pos
            room = [i for i in uav_ids if len(sets[i]) < cap_hi] or uav_ids
            i = min(room, key=lambda i: (float(np.linalg.norm(self.uavs[i].pos2d - p)),
                                         len(sets[i])))
            sets[i].add(k)
        self.apply_assignments(sets)

    # ------------------------------------------------------------------
    def apply_assignments(self, table: dict) -> None:
        """Install {uav_id: iterable of target ids}.

        Drops dead ids and enforces that a target is held by at most one UAV.
        A live target in nobody's set is left unassigned on purpose: under
        overload that is a real allocation choice, and patching it here would
        mean the repair rule, not the agent, was deciding.
        """
        live  = set(self.targets)
        taken = set()
        clean = {}
        for i in sorted(self.uavs):
            keep = set()
            for k in sorted({int(x) for x in (table.get(i) or set()) if x is not None}):
                if k in live and k not in taken:
                    keep.add(k); taken.add(k)
            self.uavs[i].set_assignment(keep)
            # set_assignment may truncate under the one-to-many ablation; release
            # whatever it dropped so another UAV can take it.
            taken -= (keep - self.uavs[i].assignment_set)
            clean[i] = set(self.uavs[i].assignment_set)
        self.bs.set_assignments(clean)

    def assignment_table(self) -> dict:
        return {i: set(uav.assignment_set) for i, uav in self.uavs.items()}

    # ------------------------------------------------------------------
    def _pick_sensing_target(self, uav) -> int:
        """Which ONE member to sense this slot: highest tr(Sigma_pos).

        That is the member with the lowest p_r, so refreshing it buys the most;
        and since sensing collapses tr(Sigma), the rule self-schedules into a
        round robin. A fixed scheduler, not a learned one - the policy's job is
        where to fly.
        """
        members = [k for k in uav.assignment_set if k in self.targets]
        if not members:
            uav.sensing_target = None
            return None
        k = max(members, key=lambda k: float(np.trace(self.ekf_state[k][1][:2, :2])))
        uav.sensing_target = k
        return k

    def _compute_set_snr(self) -> dict:
        """{uav: {member: SNR it would get pointing there from here}}. Diagnostic."""
        out = {i: {k: float(radar_snr(uav, self.ekf_state[k][0][:2]))
                   for k in uav.assignment_set if k in self.targets}
               for i, uav in self.uavs.items()}
        self._set_snr_cache = out
        return out

    # ------------------------------------------------------------------
    def _register_estimate(self, k: int, tgt) -> None:
        mu0 = tgt.state.copy()
        self.ekf_state[k] = (mu0, SIGMA0_INIT.copy())
        self.bs.add_target(k, mu0, SIGMA0_INIT.copy())
        # Seed the reward baseline so a new target contributes a zero delta on
        # its first slot rather than a spurious jump.
        self._prev_pr[k] = rescue_prob(float(np.trace(SIGMA0_INIT[:2, :2])))

    def spawn_target(self, init_state=None):
        """Birth a target in a free id slot. NOT assigned to anyone - a birth is
        the event the allocator exists to respond to."""
        free = [k for k in range(MAX_TARGETS) if k not in self.targets]
        if not free:
            return None
        k = free[0]

        if init_state is None:
            pos, vel = self._random_target_state()
        else:
            pos = np.array(init_state[:2], dtype=float)
            vel = np.array(init_state[2:], dtype=float)

        self.targets[k] = Target(k, pos, vel, birth_slot=self.t)
        self._register_estimate(k, self.targets[k])
        self.n_born_total += 1
        return k

    def remove_target(self, k: int) -> None:
        self.targets.pop(k, None)
        self.ekf_state.pop(k, None)
        self._prev_pr.pop(k, None)
        self.bs.remove_target(k)
        for uav in self.uavs.values():
            uav.drop_target(k)

    # ------------------------------------------------------------------
    @property
    def avg_rescue_delay(self) -> float:
        """D-bar = (dt / N) * sum_t |K^t| - Eq. (19).

        Over target-slots, so a target still waiting at the end contributes the
        slots it has already waited instead of being dropped.
        """
        return DT * self.backlog_sum / max(self.n_born_total, 1)

    @property
    def mean_completed_delay(self) -> float:
        """Mean delay in seconds over targets actually rescued. With
        avg_rescue_delay this separates 'rescues were fast' from 'few happened'."""
        return DT * float(np.mean(self.rescue_delays)) if self.rescue_delays else 0.0

    # ------------------------------------------------------------------
    def step(self, actions: dict) -> tuple:
        """One slot. actions: {uav_id: (dvx, dvy)}. Returns (state, info)."""
        self.t += 1
        uav_ids = sorted(self.uavs)

        for i, uav in self.uavs.items():
            dvx, dvy = actions.get(i, (0.0, 0.0))[:2]
            uav.step(dvx, dvy)

        # ---- sensing: one member per UAV ----
        measurements: dict = {k: [] for k in self.targets}
        snr_map, rate_map, sensed_now = {}, {}, set()

        _pos = np.stack([self.uavs[i].pos2d for i in uav_ids])
        gamma_map = dict(zip(uav_ids, uplink_snr_batch(_pos, self.bs.pos)))

        for i, uav in self.uavs.items():
            rate_map[i] = uplink_rate(gamma_map[i])
            k = self._pick_sensing_target(uav)
            if k is None:
                continue
            z, R, snr = measure(uav, self.targets[k].pos)
            snr_map[i] = snr
            # Delivered only if the echo is detectable AND the uplink carries it.
            if snr >= SNR_MIN and rate_map[i] >= R_MIN:
                measurements[k].append((z, R, uav.pos2d.copy()))
                sensed_now.add(k)
                self.targets[k].last_sensed_slot = self.t

        self._gamma_cache = gamma_map

        # ---- EKF: predict always, update where a measurement arrived ----
        for k in self.targets:
            mu, Sigma = self.ekf_state[k]
            mu_cur, Sigma_cur = predict(mu, Sigma)
            for z, R, uav_pos2d in measurements[k]:
                mu_cur, Sigma_cur = update(mu_cur, Sigma_cur, uav_pos2d, z, R)

            # Constrained EKF: near-singular range-bearing geometry directly
            # overhead can kick the velocity estimate to impossible values.
            spd = float(np.hypot(mu_cur[2], mu_cur[3]))
            if spd > V_MAX_TARGET:
                mu_cur[2:] *= V_MAX_TARGET / spd

            self.ekf_state[k] = (mu_cur, Sigma_cur)
            self.bs.estimates[k] = (mu_cur, Sigma_cur)

        # ---- rescue ----
        # Backlog counted before removals: a target rescued at t did wait slot t.
        backlog_before = len(self.targets)
        self.backlog_sum += backlog_before

        trace_pos = {k: float(np.trace(self.ekf_state[k][1][:2, :2])) for k in self.targets}
        pr_now    = {k: rescue_prob(v) for k, v in trace_pos.items()}
        pr_prev   = {k: self._prev_pr.get(k, pr_now[k]) for k in self.targets}
        sets_pre  = {i: sorted(self.uavs[i].assignment_set) for i in self.uavs}

        rescued = apply_rescues(self)
        self._prev_pr = {k: pr_now[k] for k in self.targets}

        for k, tgt in self.targets.items():
            tgt.step(map_size=self.map_size, rng=self._motion_rngs[k])

        set_snr = self._compute_set_snr()
        assigned = {k for uav in self.uavs.values() for k in uav.assignment_set}

        info = {
            "t": self.t,
            "backlog":            backlog_before,
            "rescued":            [k for k, _ in rescued],
            "rescue_delays":      [d for _, d in rescued],
            "n_rescued_total":    len(self.rescue_delays),
            "avg_rescue_delay_s": self.avg_rescue_delay,
            "n_unassigned":       len(set(self.targets) - assigned),
            "load":               {i: self.uavs[i].load for i in self.uavs},
            # reward inputs, Eq. (25); sets are pre-removal so a UAV keeps credit
            # for a target it just got rescued
            "rescue_prob":        pr_now,
            "rescue_prob_prev":   pr_prev,
            "assignments_pre":    sets_pre,
            "trace_pos_per_target": trace_pos,
            "time_since_sensed": {k: self.t - self.targets[k].last_sensed_slot
                                  for k in self.targets},
            "sensed_now":  sorted(sensed_now),
            "snr_db":      {i: 10 * np.log10(max(snr_map.get(i, 1e-10), 1e-10))
                            for i in self.uavs},
            "snr_linear":  {i: snr_map.get(i, 0.0)        for i in self.uavs},
            "rate_Mbps":   {i: rate_map.get(i, 0.0) / 1e6 for i in self.uavs},
            "assignments": {i: sorted(self.uavs[i].assignment_set) for i in self.uavs},
            "sensing":     {i: self.uavs[i].sensing_target for i in self.uavs},
            "set_snr":     set_snr,
            "ekf_means":   {k: self.ekf_state[k][0][:2].copy() for k in self.targets},
        }
        return self._system_state(), info

    # ------------------------------------------------------------------
    def _system_state(self) -> dict:
        """Full system state S^t - Eq. (21)."""
        return {
            "t": self.t,
            "uavs":        {i: uav.state.copy()        for i, uav in self.uavs.items()},
            "assignments": {i: set(uav.assignment_set) for i, uav in self.uavs.items()},
            "sensing":     {i: uav.sensing_target      for i, uav in self.uavs.items()},
            "load":        {i: uav.load                for i, uav in self.uavs.items()},
            "gamma":       dict(getattr(self, "_gamma_cache", {})),
            "set_snr":     {i: dict(v) for i, v in getattr(self, "_set_snr_cache", {}).items()},
            "targets": {k: (self.ekf_state[k][0].copy(), self.ekf_state[k][1].copy())
                        for k in self.targets},
            "rescue_prob": {k: rescue_prob(float(np.trace(self.ekf_state[k][1][:2, :2])))
                            for k in self.targets},
            "time_since_sensed": {k: self.t - self.targets[k].last_sensed_slot
                                  for k in self.targets},
            "birth_slot": {k: self.targets[k].birth_slot for k in self.targets},
            "backlog": len(self.targets),
        }
