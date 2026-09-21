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
    """

    def __init__(self, target_id: int, init_pos: np.ndarray, init_vel: np.ndarray):
        self.id = target_id
        self.state = np.array(
            [init_pos[0], init_pos[1], init_vel[0], init_vel[1]], dtype=float
        )
        self.active = True

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
