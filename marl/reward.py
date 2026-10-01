import numpy as np

# Shaping weight on the dense progress term. Sum_t Δp_r over a target's life is
# ≈1 (p_r climbs ~0→secured), same order as the +1 completion bonus, so β≈1
# balances "make progress" against "finish". Tune in ~[0.5, 2]; move to
# config.params if you want it swept.
BETA_SHAPING = 1.0


def per_agent_rewards(info: dict, uavs: dict) -> np.ndarray:
    """r_u = (members of A_u rescued this slot) + BETA * sum_{k in A_u} (p_r,k^t - p_r,k^{t-1})

    A UAV is rewarded for (a) completing a rescue and (b) the per-slot increase
    in rescue probability of the members it still holds - i.e. for driving down
    their tracking uncertainty. Delay is optimised implicitly through the
    discount: an earlier rescue and an earlier p_r climb are worth more under
    gamma < 1, so the policy is pushed to secure fast.

    Every term is controllable by the actor (move + which member to sense) and
    dense, unlike the flat -|A_u| charge, whose per-slot value was set by the
    assigner and so injected reward the policy could not act on.

    Credit rules per member k of A_u (read pre-removal, via assignments_pre):
      * rescued this slot  -> +1 completion bonus, its Δp_r is NOT also counted
        (no need to reason about the p_r at which the stochastic rescue fired);
      * present last slot and this slot, not rescued -> += p_r[k] - p_r_prev[k];
      * newly assigned this slot (no p_r_prev) -> contributes nothing yet, it
        starts accruing Δp_r next slot.

    Required info keys:
      assignments_pre : {uav_id: iterable of target ids}  (before rescue removals)
      rescued         : iterable of target ids rescued this slot
      p_r             : {target_id: p_r at slot t}   (after the EKF update)
      p_r_prev        : {target_id: p_r at slot t-1}

    Returns array shape [N], ordered by UAV id 0..N-1.
    """
    N        = len(uavs)
    sets     = info.get("assignments_pre", info.get("assignments", {}))
    rescued  = set(info.get("rescued", ()))
    p_r      = info.get("p_r", {})
    p_r_prev = info.get("p_r_prev", {})

    rewards = np.zeros(N, dtype=np.float32)
    for i in range(N):
        saved   = 0
        shaping = 0.0
        for k in sets.get(i, ()):
            if k in rescued:
                saved += 1                              # completion banked
            elif k in p_r and k in p_r_prev:
                shaping += p_r[k] - p_r_prev[k]         # dense progress
            # newly-assigned member (no prev): starts next slot
        rewards[i] = float(saved) + BETA_SHAPING * float(shaping)
    return rewards
