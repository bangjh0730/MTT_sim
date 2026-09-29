import numpy as np

from config.params import W_SHAPE


def per_agent_rewards(info: dict, uavs: dict) -> np.ndarray:
    """r_u = -|A_u| + W_SHAPE * sum_{k in A_u} [p~_r,k(t) - p_r,k(t-1)].

    First term: the objective itself. D of Eq. (19) is (dt/N) sum_t |K^t|, and
    the assignment is a partition, so |K^t| = sum_u |A_u^t| and charging each UAV
    its own live set makes the per-agent rewards sum to -|K^t|. Minimising it IS
    minimising mean rescue delay - a rescue pays off by removing the target from
    the set for every later slot, so "rescue fast" needs no separate bonus.

    Second term: Eq. (25), which is potential-based shaping on
    Phi_u = sum_{k in A_u} p_r,k. Phi(s') - Phi(s) leaves the optimal policy
    unchanged (Ng et al. 1999), so it adds no bias, and it pays for closing
    distance in the slot it happens instead of only when the rescue lands. That
    matters because the delay term alone credits a rescue as a small reduction
    smeared over all later slots.

    A target rescued this slot has no p_r,k(t) - it is gone from the target set.
    It is scored as p~ = 1, the absorbing success, so its contribution is
    +(1 - p_r,k(t-1)): a positive spike on the rescue slot. Defaulting the
    missing value to 0 instead would make the term negative and penalise the UAV
    for succeeding.

    Sets and p_r are read BEFORE this slot's rescue removals, so a target
    rescued this slot is charged for the slot it did wait.
    """
    N        = len(uavs)
    pr_now   = info.get("rescue_prob", {})
    pr_prev  = info.get("rescue_prob_prev", {})
    sets     = info.get("assignments_pre", info.get("assignments", {}))
    rescued  = set(info.get("rescued", ()))

    rewards = np.zeros(N, dtype=np.float32)
    for i in range(N):
        members = sets.get(i, ())
        shaping = 0.0
        for k in members:
            now  = 1.0 if k in rescued else pr_now.get(k, pr_prev.get(k, 0.0))
            shaping += now - pr_prev.get(k, pr_now.get(k, 0.0))
        rewards[i] = W_SHAPE * shaping - float(len(members))
    return rewards
