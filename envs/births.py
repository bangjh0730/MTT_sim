from config.params import MAX_TARGETS


def apply_births(env, schedule):
    """
    Target births for one slot — EVALUATION ONLY, driven by a pre-computed
    DisturbanceSchedule so the arriving load is identical across modes.

    There is NO admission guard any more. The old code refused a birth unless
    |K| < |U_active|, because the whole simulation rested on "every target has its
    own UAV". That invariant is exactly what this regime discards: with 3 UAVs and
    8+ targets, |K| > |U| is the normal state, and refusing births to protect the
    invariant would suppress the contention the experiment exists to study. The
    only remaining ceiling is MAX_TARGETS, the size of the logging id space, set
    far above any backlog a run can reach.

    There is also no death path here. A target leaves only by being rescued
    (envs/rescue.py), which runs after the EKF update because it reads tr(Sigma).
    The old guard also consulted the number of ACTIVE UAVs; there is no such
    thing now, since the fleet never loses a member.

    Returns `born`: a list of newly created target ids.
    """
    born = []

    if env.t in schedule.births and len(env.targets) < MAX_TARGETS:
        k = env.spawn_target(init_state=schedule.births[env.t])
        if k is not None:
            born.append(k)

    return born
