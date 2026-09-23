import numpy as np
from config.params import F_MAT, Q_MAT, V_MAX_TARGET

# Q is rank-2 so Cholesky fails; take the matrix square root once at import.
_vals, _vecs = np.linalg.eigh(Q_MAT)
_Q_SQRT = _vecs @ np.diag(np.sqrt(np.maximum(_vals, 0.0)))


class Target:
    """Ground target, constant velocity with perturbation - Eq. (2).

    Persists from birth until rescued; there is no random death.
    """

    def __init__(self, target_id, init_pos, init_vel, birth_slot: int = 0):
        self.id = target_id
        self.state = np.array([init_pos[0], init_pos[1],
                               init_vel[0], init_vel[1]], dtype=float)

        self.birth_slot       = int(birth_slot)
        self.rescue_slot      = None
        self.rescued          = False
        self.last_sensed_slot = int(birth_slot)

    def rescue_delay(self, t_now: int) -> int:
        """Slots spent awaiting rescue; censored at t_now if still waiting."""
        end = self.rescue_slot if self.rescue_slot is not None else t_now
        return int(end - self.birth_slot)

    @property
    def pos(self) -> np.ndarray:
        return self.state[:2].copy()

    @property
    def vel(self) -> np.ndarray:
        return self.state[2:].copy()

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.state[2:]))

    def step(self, map_size: float = None, rng=None) -> None:
        """Advance one slot. `rng` is the target's own stream (see MTTEnv)."""
        rng = rng if rng is not None else np.random
        self.state = F_MAT @ self.state + _Q_SQRT @ rng.standard_normal(4)

        spd = np.linalg.norm(self.state[2:])
        if spd > V_MAX_TARGET:
            self.state[2:] *= V_MAX_TARGET / spd

        if map_size is not None:
            lo, hi = 0.05 * map_size, 0.95 * map_size
            for d in range(2):
                if self.state[d] < lo:
                    self.state[d] = 2 * lo - self.state[d]
                    self.state[d + 2] *= -1
                elif self.state[d] > hi:
                    self.state[d] = 2 * hi - self.state[d]
                    self.state[d + 2] *= -1
                self.state[d] = np.clip(self.state[d], lo, hi)
