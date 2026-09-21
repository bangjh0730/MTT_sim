import numpy as np
from config.params import F_MAT, Q_MAT, V_MAX_TARGET

# Q_MAT is rank-2 (singular), so Cholesky fails. Pre-compute the matrix square root
# via eigendecomposition once at import time; sampling is then a cheap mat-vec multiply.
_vals, _vecs = np.linalg.eigh(Q_MAT)
_Q_SQRT = _vecs @ np.diag(np.sqrt(np.maximum(_vals, 0.0)))


class Target:
    """
    Ground target following constant-velocity motion with stochastic perturbation.
    State: s_k^t = [x_k, y_k, v_x,k, v_y,k]

    A target persists from its birth until it is RESCUED; there is no random
    death. Rescue is policy-dependent (it is driven by the BS's own tracking
    uncertainty), so it is drawn online, never pre-scheduled — see envs/rescue.py.
    """

    def __init__(self, target_id: int, init_pos: np.ndarray, init_vel: np.ndarray,
                 birth_slot: int = 0):
        self.id = target_id
        self.state = np.array(
            [init_pos[0], init_pos[1], init_vel[0], init_vel[1]], dtype=float
        )
        # ---- rescue bookkeeping ------------------------------------------
        # A target now leaves the mission only by being RESCUED, so its whole
        # life is the quantity the experiment measures. birth_slot is stamped at
        # spawn (0 for the initial targets); rescue_slot is stamped when it is
        # removed, and the difference is that target's rescue delay in slots.
        self.birth_slot       = int(birth_slot)
        self.rescue_slot      = None
        self.rescued          = False
        # Last slot a UAV actually delivered a measurement for this target. The
        # gap (t - last_sensed_slot) is how long it has been going stale while
        # its holder cycled through the rest of its set.
        self.last_sensed_slot = int(birth_slot)

    def rescue_delay(self, t_now: int) -> int:
        """Slots spent awaiting rescue — up to the rescue, or up to t_now if the
        target is still waiting (a censored life at the end of the episode)."""
        end = self.rescue_slot if self.rescue_slot is not None else t_now
        return int(end - self.birth_slot)

    @property
    def pos(self) -> np.ndarray:
        """2D horizontal position [x, y]."""
        return self.state[:2].copy()

    @property
    def vel(self) -> np.ndarray:
        """2D velocity [vx, vy]."""
        return self.state[2:].copy()

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.state[2:]))

    def step(self, map_size: float = None, rng=None) -> None:
        """
        Advance one slot: s_k^{t+1} = F s_k^t + w_k,  w_k ~ N(0, Q).
        Optionally reflects at map boundaries to keep the target in area.

        rng : draw source for the process noise. Defaults to the global np.random
        stream (training). Evaluation passes a dedicated, seeded Generator so target
        motion is independent of how much of the global stream each policy consumes
        elsewhere (measurement noise, action sampling) — otherwise the same seed
        gives different target trajectories in marl vs agentic mode.
        """
        rng = rng if rng is not None else np.random
        w = _Q_SQRT @ rng.standard_normal(4)
        self.state = F_MAT @ self.state + w

        # Clamp speed to v_max_k
        spd = np.linalg.norm(self.state[2:])
        if spd > V_MAX_TARGET:
            self.state[2:] *= V_MAX_TARGET / spd

        # Boundary reflection within [5%, 95%] of map
        if map_size is not None:
            lo = 0.05 * map_size
            hi = 0.95 * map_size
            for d in range(2):
                if self.state[d] < lo:
                    self.state[d] = 2 * lo - self.state[d]
                    self.state[d + 2] *= -1
                elif self.state[d] > hi:
                    self.state[d] = 2 * hi - self.state[d]
                    self.state[d + 2] *= -1
                self.state[d] = np.clip(self.state[d], lo, hi)
