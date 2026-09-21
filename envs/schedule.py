import numpy as np

from config.params import P_BIRTH, MAP_SIZE, V_MAX_TARGET


class DisturbanceSchedule:
    """
    Pre-computed, state-independent target arrivals for one evaluation episode.

    Drawing every birth (slot + spawn state) up front from the seed makes the
    arriving load IDENTICAL across modes for the same seed, regardless of policy:
    the number of RNG draws per slot no longer depends on how many targets happen
    to be alive.

    WHAT IS NOT HERE, and why. The old schedule also pre-drew random DEATHS, so
    that the whole target set was policy-independent and the same-seed comparison
    was exact. Targets no longer die randomly — they are RESCUED, with
    p_r = lambda / (lambda + tr(Sigma)), which is policy-dependent by design:
    better tracking shrinks tr(Sigma) and rescues faster. Rescue therefore cannot
    be pre-scheduled without destroying the very effect the experiment measures.

    The comparison survives because the split is clean: BIRTHS (the load the
    system is handed) stay identical across modes, while RESCUES (how fast the
    system clears that load) are allowed — required — to differ. Rescue draws come
    from per-target-id streams in MTTEnv so each target's draws depend only on its
    own lifetime, not on which other ids are alive.
    """

    def __init__(self, seed, T: int):
        rng = np.random.default_rng(0 if seed is None else seed)
        cen = MAP_SIZE / 2.0

        self.births: dict = {}   # slot -> (x, y, vx, vy) spawn state

        # Fixed draw order per slot keeps the schedule fully deterministic.
        for t in range(1, T + 1):
            if rng.random() < P_BIRTH:
                angle = rng.uniform(0.0, 2.0 * np.pi)
                r     = rng.uniform(MAP_SIZE * 0.15, MAP_SIZE * 0.45)
                x = float(np.clip(cen + r * np.cos(angle), 0.05 * MAP_SIZE, 0.95 * MAP_SIZE))
                y = float(np.clip(cen + r * np.sin(angle), 0.05 * MAP_SIZE, 0.95 * MAP_SIZE))
                spd = rng.uniform(0.0, V_MAX_TARGET)
                ang = rng.uniform(0.0, 2.0 * np.pi)
                self.births[t] = (x, y, spd * np.cos(ang), spd * np.sin(ang))
