import numpy as np

from config.params import P_BIRTH


class DisturbanceSchedule:
    """Pre-drawn birth TIMES for one episode (training and evaluation).

    Drawn up front from the seed so the arriving load is identical across runs
    regardless of policy. Where each survivor appears is the env's site field,
    which places them in birth order from its own stream - so positions are
    policy-independent too. Rescues are NOT scheduled: they depend on tracking
    quality, which is the thing being measured.
    """

    def __init__(self, seed, T: int):
        rng = np.random.default_rng(0 if seed is None else seed)
        self.births = {t for t in range(1, T + 1) if rng.random() < P_BIRTH}
