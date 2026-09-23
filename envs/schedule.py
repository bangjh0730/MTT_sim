import numpy as np

from config.params import P_BIRTH, MAP_SIZE, V_MAX_TARGET


class DisturbanceSchedule:
    """Pre-drawn target arrivals for one episode (training and evaluation).

    Births are drawn up front from the seed so the arriving load is identical
    across runs regardless of policy. Rescues are NOT scheduled: they depend on
    tracking quality, which is the thing being measured.
    """

    def __init__(self, seed, T: int):
        rng = np.random.default_rng(0 if seed is None else seed)
        self.births: dict = {}   # slot -> (x, y, vx, vy)

        for t in range(1, T + 1):
            if rng.random() < P_BIRTH:
                x, y = rng.uniform(0.05 * MAP_SIZE, 0.95 * MAP_SIZE, size=2)
                spd = rng.uniform(0.0, V_MAX_TARGET)
                ang = rng.uniform(0.0, 2.0 * np.pi)
                self.births[t] = (float(x), float(y),
                                  spd * np.cos(ang), spd * np.sin(ang))
