from config.params import MAX_TARGETS


def apply_births(env, schedule):
    """Spawn this slot's scheduled target, if any - Eq. (5). Evaluation only.

    No admission guard: |K| > |U| is the normal state, so refusing births to
    keep the fleet ahead would suppress the contention being studied. The only
    ceiling is MAX_TARGETS, the logging id space.
    """
    born = []
    if env.t in schedule.births and len(env.targets) < MAX_TARGETS:
        k = env.spawn_target(init_state=schedule.births[env.t])
        if k is not None:
            born.append(k)
    return born
