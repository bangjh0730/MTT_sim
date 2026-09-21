from config.params import NUM_TARGETS


def apply_uav_failures(env, schedule) -> list:
    """
    Shock-failure model for one slot — EVALUATION ONLY, driven by a pre-computed
    DisturbanceSchedule so failures are identical across modes for the same seed.

    schedule.failures[slot] lists the UAV ids drawn to fail this slot (each UAV
    independently at rate BETA_FAIL). Because UAV ids are stable across modes, the
    same UAVs are scheduled to fail; a failure is APPLIED only subject to the guards:

      1. No failures in the first 50 or last 100 slots (clean lock-on / run-out).
      2. The active fleet stays at or above the LIVE target count (and never below
         the nominal NUM_TARGETS), so |K^t| <= |U^t| always holds and every target
         can still be assigned a UAV.

    Guard 2 used to floor at NUM_TARGETS alone, on the reasoning that the live
    target count was mode-dependent and gating on it would make the admitted
    failure sequence mode-dependent too, defeating the shared schedule. That
    reasoning was circular. Births are admitted up to the live active count, so
    |K| could reach 7 with |U| = 7; a failure was then still admitted (7 > the
    nominal 5), |K| exceeded |U|, and envs/birth_death.py resolved it by DELETING
    the worst-tracked target. That deletion was the only tracking-quality-dependent
    input to |K| — i.e. the very thing that made |K| mode-dependent and was cited
    as the reason not to gate on it.

    Flooring at max(NUM_TARGETS, len(env.targets)) breaks the loop: a failure that
    would push the fleet below the live target count is simply never admitted, no
    target is ever deleted for being poorly tracked, and |K| goes back to being
    driven purely by the shared schedule — so the admitted failure sequence stays
    identical across modes, which is what the original guard wanted in the first
    place.

    A failed UAV stops all sensing/communication; the BS detects the loss only after
    N_FAIL_DETECT silent slots (tracked by MTTEnv._no_signal_count).

    Returns the list of UAV ids deactivated this slot.
    """
    failed = []

    if env.t <= 50 or env.t > env.T - 100:
        return failed

    scheduled = schedule.failures.get(env.t, [])
    if not scheduled:
        return failed

    active = sum(1 for uav in env.uavs.values() if uav.active)
    # Floor at the LIVE target count, never below the nominal one. Decremented per
    # admitted failure, so a slot that schedules two failures at the floor admits
    # only the first — the second would break |K| <= |U|.
    budget = active - max(NUM_TARGETS, len(env.targets))
    for i in scheduled:
        if budget <= 0:
            break
        if env.uavs[i].active:
            env.uavs[i].active = False   # physical truth only; BS learns via silence
            failed.append(int(i))
            budget -= 1

    return failed
