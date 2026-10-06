import numpy as np


def per_agent_rewards(info: dict, uavs: dict) -> np.ndarray:
    """r_u = number of members of A_u rescued this slot.

    Speed is carried by the discount, not by the reward: a rescue at slot d is
    worth gamma^d, so sooner is worth more. Charging -1 per slot a target is
    still held is the same objective up to a constant 1/(1-gamma) per target,
    but that constant scales with |A_u|, which the assigner sets rather than the
    policy, so paying on rescue keeps only what the UAV did.

    Sets are read BEFORE this slot's rescue removals, so a UAV keeps credit for
    a target it just got rescued.

    A track-quality term was tried and removed. It was meant to stop the policy
    writing off distant members, and did not: never-sensed stayed at 20% against
    18% without it, and rose to 27% when weighted more heavily. At its best
    weight it was also slightly worse on the objective (D 65.7 s against 62.9 s
    at matched settings).
    """
    sets    = info.get("assignments_pre", info.get("assignments", {}))
    rescued = set(info.get("rescued", ()))

    rewards = np.zeros(len(uavs), dtype=np.float32)
    for i in range(len(uavs)):
        rewards[i] = float(sum(1 for k in sets.get(i, ()) if k in rescued))
    return rewards
