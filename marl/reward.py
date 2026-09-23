import numpy as np


def per_agent_rewards(info: dict, uavs: dict) -> np.ndarray:
    """r_i = sum_{k in A_i} p_r,k(t): the expected number of rescues in UAV i's
    set this slot.

    Replaces the difference form sum_k (p_r,k(t) - p_r,k(t-1)) of Eq. (25).
    That form telescopes: summed over an episode it collapses to p_r at the end
    minus p_r at the start, so the dense per-slot signal cancels (every
    sensing spike is repaid by the decay that follows) and the return only
    credits the p_r a target happened to hold when its rescue was drawn. The
    level form is what the objective counts: with the number of live targets
    held, rescues per slot is sum_k p_r,k in expectation, and by Little's law
    more rescues per slot is lower delay.

    Sets are taken BEFORE this slot's rescue removals, so a UAV still gets
    credit for a target it just got rescued.
    """
    N = len(uavs)
    rewards = np.zeros(N, dtype=np.float32)

    pr_now = info.get("rescue_prob", {})
    sets   = info.get("assignments_pre", info.get("assignments", {}))

    for i in range(N):
        rewards[i] = float(sum(pr_now.get(k, 0.0) for k in sets.get(i, ())))
    return rewards
