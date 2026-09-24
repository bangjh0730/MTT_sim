import numpy as np


def per_agent_rewards(info: dict, uavs: dict) -> np.ndarray:
    """r_i = -sum_{k in A_i} (1 - p_r,k(t)): minus the expected number of UAV
    i's targets still waiting after this slot.

    Each member that waits costs 1 - that is the delay: summed over time the
    reward is -sum_t |A_i^t| in expectation, UAV i's share of the target-slots
    counted by Eq. (19). Each member's cost is cut by its rescue probability,
    so raising p_r is rewarded in the same slot.

    Replaces two earlier forms:
      - sum_k (p_r,k(t) - p_r,k(t-1)), Eq. (25): telescopes, so over an
        episode it collapses to p_r at the end minus p_r at the start.
      - sum_k p_r,k(t), expected rescues: with a birth process every target is
        eventually rescued, so rescues per slot average the birth rate whatever
        the policy does (measured 0.151 = P_BIRTH / |U| throughout training).
        It does not see how LONG targets wait, and a rescued target stops
        paying, so clearing a set earned nothing.

    Sets are taken BEFORE this slot's rescue removals, so a target rescued this
    slot is charged 1 - p_r for the slot it did wait.
    """
    N = len(uavs)
    rewards = np.zeros(N, dtype=np.float32)

    pr_now = info.get("rescue_prob", {})
    sets   = info.get("assignments_pre", info.get("assignments", {}))

    for i in range(N):
        rewards[i] = -float(sum(1.0 - pr_now.get(k, 0.0) for k in sets.get(i, ())))
    return rewards
