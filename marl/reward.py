import numpy as np


def per_agent_rewards(info: dict, uavs: dict) -> np.ndarray:
    """r_u = (members of A_u rescued this slot) - |A_u|

    Reward a rescue, charge a target for every slot it is still waiting.

    The second term is the delay objective itself. Mean rescue delay is
    (dt / N) sum_t |K^t|, and the assignment is a partition, so
    |K^t| = sum_u |A_u^t| and charging each UAV its own live set makes the
    per-agent rewards sum to -|K^t| plus the rescues.

    Sets are read BEFORE this slot's rescue removals, so a target rescued this
    slot is both credited and charged for the slot it did wait.
    """
    N       = len(uavs)
    sets    = info.get("assignments_pre", info.get("assignments", {}))
    rescued = set(info.get("rescued", ()))

    rewards = np.zeros(N, dtype=np.float32)
    for i in range(N):
        members = sets.get(i, ())
        saved   = sum(1 for k in members if k in rescued)
        rewards[i] = float(saved) - float(len(members))
    return rewards
