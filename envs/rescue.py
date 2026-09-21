import numpy as np

from config.params import LAMBDA_RESCUE


def rescue_prob(trace_pos: float) -> float:
    """p_r = lambda / (lambda + tr(Sigma_pos)).

    Continuous in the tracking uncertainty, so no "is this target tracked" gate is
    needed: a target a UAV is actively sensing sits at small tr(Sigma) and draws a
    high rescue probability, while one nobody has sensed for a while has a
    tr(Sigma) that grows without bound under predict-only EKF steps, sending p_r
    smoothly to 0. That is the whole coupling between tracking quality and mission
    progress — and it is why rescue cannot be pre-scheduled: it depends on the
    policy under test.
    """
    return float(LAMBDA_RESCUE / (LAMBDA_RESCUE + max(trace_pos, 0.0)))


def apply_rescues(env):
    """
    Rescue stage for one slot. Runs AFTER the EKF update, so tr(Sigma) already
    reflects any measurement delivered this slot.

    Each live target draws independently from its OWN id-indexed RNG stream
    (env._rescue_rngs), so the draws a target sees depend only on how many slots
    it has itself been alive — never on which other ids happen to be alive, which
    would otherwise couple every target's fate to the policy's effect on the
    others through a shared draw order.

    Returns a list of (target_id, delay_in_slots) for the targets rescued.
    """
    rescued = []
    for k in sorted(env.targets):
        tr_pos = float(np.trace(env.ekf_state[k][1][:2, :2]))
        if env._rescue_rngs[k].random() < rescue_prob(tr_pos):
            tgt = env.targets[k]
            tgt.rescued     = True
            tgt.rescue_slot = env.t
            delay = tgt.rescue_delay(env.t)
            env.rescue_delays.append(delay)
            env.rescued_targets.append((k, tgt.birth_slot, env.t, delay))
            env.remove_target(k)
            rescued.append((k, delay))
    return rescued
