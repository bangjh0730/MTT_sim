import numpy as np

from config.params import W_SHAPE


def per_agent_rewards(info: dict, uavs: dict) -> np.ndarray:
    """r_u = -|A_u| + W_SHAPE * sum_{k in A_u} [p~_r,k(t) - p_r,k(t-1)].

    First term: the objective itself. Mean rescue delay is (dt/N) sum_t |K^t|,
    and the assignment is a partition, so |K^t| = sum_u |A_u^t| and charging each
    UAV its own live set makes the rewards sum to -|K^t|. A rescue pays off by
    removing the target from the set for every later slot, so "rescue fast"
    needs no separate bonus.

    Second term: potential-based shaping on Phi_u = sum_{k in A_u} p_r,k. Since
    it is Phi(s') - Phi(s) it cannot move the optimal policy (Ng et al. 1999),
    and it pays for closing distance in the slot it happens rather than only
    when the rescue lands.

    A target rescued this slot is gone from the target set and has no p_r,k(t).
    It scores p~ = 1, the absorbing success, contributing +(1 - p_r,k(t-1)).
    Defaulting the missing value to 0 would instead penalise the UAV for
    succeeding.

    Sets and p_r are read BEFORE this slot's rescue removals, so a target
    rescued this slot is charged for the slot it did wait.

    A distance-shaping term was tried and removed as unnecessary: the delay term
    alone already separates a policy that flies at its targets from one that
    flies off the map by ~17.6 return per slot. Summed over members it was also
    harmful, being minimised at the set's geometric median and so paying a UAV
    to hover equidistant instead of going to serve one target.
    """
    N       = len(uavs)
    pr_now  = info.get("rescue_prob", {})
    pr_prev = info.get("rescue_prob_prev", {})
    sets    = info.get("assignments_pre", info.get("assignments", {}))
    rescued = set(info.get("rescued", ()))

    rewards = np.zeros(N, dtype=np.float32)
    for i in range(N):
        members = sets.get(i, ())
        shaping = 0.0
        for k in members:
            now = 1.0 if k in rescued else pr_now.get(k, pr_prev.get(k, 0.0))
            shaping += now - pr_prev.get(k, pr_now.get(k, 0.0))
        rewards[i] = W_SHAPE * shaping - float(len(members))
    return rewards
