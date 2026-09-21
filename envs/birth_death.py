from config.params import MAX_TARGETS


def apply_target_dynamics(env, schedule):
    """
    Target birth/death for one slot — EVALUATION ONLY, driven by a pre-computed
    DisturbanceSchedule so the stochastic events are identical across modes.

    Each slot:
      * scheduled random death: remove one target if this slot is in schedule.deaths;
      * scheduled birth: spawn a target at the schedule's spawn state if this slot is
        in schedule.births and the invariant leaves room (|K| stays <= |U_active|).

    NO target is ever removed for being poorly tracked. Two such triggers used to
    live here and both are gone, for the same reason: each was driven by a mode's
    own tracking quality, so each let the disturbance sequence itself depend on the
    policy under test, destroying same-seed comparability.

      * A BS uncertainty-threshold death (tr(Sigma) > SIGMA_TH) removed the target
        outright once its uncertainty grew too large.
      * A "fleet-shed" dropped the WORST-TRACKED target whenever the active fleet
        had shrunk below the live target count, to restore |K| <= |U|.

    The fleet-shed is now unnecessary rather than merely undesirable: the invariant
    it enforced after the fact is enforced up front instead. Births are admitted
    only while |K| < |U_active| (below), and envs/failure.py now refuses a failure
    that would drop the fleet below the live target count, so |K| > |U| cannot arise
    from either. The assumption is what it always claimed to be — there is always a
    way to assign every target a UAV — rather than something bought by deleting a
    target the moment it became inconvenient.

    The one residual path to |K| > |U| is battery depletion, which no guard can
    pre-empt. The honest consequence is then simply that a target goes uncovered and
    its PCRLB climbs, which is exactly what the metric exists to measure; deleting
    the target instead would have hidden that failure.

    Returns (born, removed): born is a list of target ids; removed is a list of
    (target_id, cause) tuples, cause always "died" (the tuple keeps its shape for
    callers and for the saved logs).
    """
    born, removed = [], []

    # No scheduled births/deaths in the first 50 or last 100 slots (clean lock-on and
    # run-out for the metrics) — mirrors the failure guard.
    if env.t <= 50 or env.t > env.T - 100:
        return born, removed

    # ---- scheduled random survival death ----
    if env.t in schedule.deaths and len(env.targets) > 1:
        live = sorted(env.targets)
        victim = live[int(schedule.deaths[env.t] * len(live))]
        env.remove_target(victim)
        removed.append((victim, "died"))

    # ---- scheduled birth ----
    # A birth is applied only if the invariant leaves room: len(targets)+1 <= |U_active|.
    # Read the active count here rather than at the top of the slot: a failure this
    # same slot (envs/failure.py runs first) must be reflected, or a birth could be
    # admitted against a fleet that no longer exists.
    n_active = sum(1 for uav in env.uavs.values() if uav.active)
    if env.t in schedule.births and len(env.targets) < min(MAX_TARGETS, n_active):
        k = env.spawn_target(init_state=schedule.births[env.t])
        if k is not None:
            born.append(k)

    return born, removed
