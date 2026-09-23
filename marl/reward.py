import numpy as np


def per_agent_rewards(info: dict, uavs: dict) -> np.ndarray:
    """Eq. (25): r_i = sum_{k in A_i} (p_r,k(t) - p_r,k(t-1)).

    Sets are taken BEFORE this slot's rescue removals, so a UAV still gets
    credit for the final jump on a target it just got rescued.
    """
    N = len(uavs)
    rewards = np.zeros(N, dtype=np.float32)

    pr_now  = info.get("rescue_prob", {})
    pr_prev = info.get("rescue_prob_prev", {})
    sets    = info.get("assignments_pre", info.get("assignments", {}))

    for i in range(N):
        rewards[i] = float(sum(
            pr_now.get(k, 0.0) - pr_prev.get(k, pr_now.get(k, 0.0))
            for k in sets.get(i, ())
        ))
    return rewards
