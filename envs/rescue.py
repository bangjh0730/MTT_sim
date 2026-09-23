import numpy as np

from config.params import LAMBDA_RESCUE


def rescue_prob(trace_pos: float) -> float:
    """p_r = lambda / (lambda + tr(Sigma_pos)) - Eq. (6).

    Continuous in the tracking uncertainty, so no "is this target tracked" gate
    is needed: an unsensed target's tr(Sigma) grows without bound and drives p_r
    to zero on its own.
    """
    return float(LAMBDA_RESCUE / (LAMBDA_RESCUE + max(trace_pos, 0.0)))


def apply_rescues(env):
    """Draw a rescue for every live target. Runs after the EKF update, so
    tr(Sigma) reflects any measurement delivered this slot.

    Each target draws from its own id-indexed stream, so its draws depend only
    on its own lifetime, not on which other ids are alive.

    Returns [(target_id, delay_in_slots), ...].
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
