def apply_births(env, schedule):
    """Spawn this slot's scheduled target, if any - Eq. (5).

    No admission guard and no ceiling: |K| > |U| is the normal state, and
    refusing births would suppress the contention being studied.
    """
    born = []
    if env.t in schedule.births:
        born.append(env.spawn_target(init_state=schedule.births[env.t]))
    return born
