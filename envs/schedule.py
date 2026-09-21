import numpy as np

from config.params import (
    P_BIRTH, P_SURVIVE, BETA_FAIL, NUM_TARGETS, NUM_UAVS,
    MAP_SIZE, V_MAX_TARGET,
)


class DisturbanceSchedule:
    """
    Pre-computed, state-independent disturbance events for one evaluation episode.

    The stochastic disturbances used to be drawn ONLINE, where the number of RNG
    draws in a slot depends on the live-target and active-UAV counts. Because marl
    and agentic drive the swarm into different states, the shared seed's stream
    desynchronised and the two modes saw different births/deaths/failures. Drawing
    the whole schedule up front — every birth (slot + spawn state), random death
    (slot + victim selector) and shock failure (slot + UAV ids) — from the seed makes
    these IDENTICAL across modes for the same seed, regardless of policy.

    Nothing outside this schedule perturbs the target set any more. Fleet-shed
    removal — dropping the worst-tracked target to restore |K| <= |U| — used to be
    the one unscheduled removal, and was left unscheduled precisely because which
    target is worst-tracked reflects the policy under test. It has since been
    deleted outright: envs/failure.py now refuses any failure that would drop the
    fleet below the live target count, so the invariant holds without removing
    anything (see envs/birth_death.py). The disturbance sequence is therefore fully
    determined by this schedule, and identical across modes for the same seed.
    """

    def __init__(self, seed, T: int):
        rng = np.random.default_rng(0 if seed is None else seed)
        cen = MAP_SIZE / 2.0

        self.births:   dict = {}   # slot -> (x, y, vx, vy) spawn state
        self.deaths:   dict = {}   # slot -> fraction in [0,1) selecting the victim
        self.failures: dict = {}   # slot -> [uav_ids] that fail this slot

        # Per-slot aggregate rate for a random survival death (the per-target model
        # summed over the nominal target count), so it is independent of live count.
        death_rate = NUM_TARGETS * (1.0 - P_SURVIVE)

        # Fixed draw order per slot keeps the schedule fully deterministic.
        for t in range(1, T + 1):
            if rng.random() < P_BIRTH:
                angle = rng.uniform(0.0, 2.0 * np.pi)
                r     = rng.uniform(MAP_SIZE * 0.3, MAP_SIZE * 0.6)
                x = float(np.clip(cen + r * np.cos(angle), 0.05 * MAP_SIZE, 0.95 * MAP_SIZE))
                y = float(np.clip(cen + r * np.sin(angle), 0.05 * MAP_SIZE, 0.95 * MAP_SIZE))
                spd = rng.uniform(0.0, V_MAX_TARGET)
                ang = rng.uniform(0.0, 2.0 * np.pi)
                self.births[t] = (x, y, spd * np.cos(ang), spd * np.sin(ang))

            if rng.random() < death_rate:
                self.deaths[t] = float(rng.random())

            fails = [i for i in range(NUM_UAVS) if rng.random() < BETA_FAIL]
            if fails:
                self.failures[t] = fails
