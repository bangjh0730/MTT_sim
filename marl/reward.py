import numpy as np


def per_agent_rewards(info: dict, uavs: dict) -> np.ndarray:
    """r_i = sum_{k in A_i} p_r,k(t) - |K_a^t| / |U|.

    Two parts:
      - own expected rescues, sum of p_r over UAV i's set: credit for how well
        UAV i positions and senses (maximise rescue probability);
      - an equal share of the fleet backlog, |K_a^t| / |U|, where K_a^t is the
        live targets that have been assigned at least once: a target's waiting
        is charged from its first assignment until its rescue (minimise rescue
        delay). The metric D of Eq. (19) still counts from birth; the two agree
        whenever targets are assigned the slot they appear.
    Summed over UAVs the reward is -sum_{k in K} (1 - p_r,k) when every target
    is assigned: minus the expected number of targets still waiting after the
    slot.

    The delay is charged to the FLEET, not to each UAV's own set. Charging
    -sum_{k in A_i} (1 - p_r,k) per UAV collapsed training (backlog ~100,
    ~17 rescues per episode): the allocator re-partitions on every birth and
    rescue by where the UAVs are, reassigning about as many targets per slot as
    are rescued, so a UAV was charged at once for flying toward targets it
    would be handed, and paid off only later when it rescued them. A shared
    backlog cannot be moved between UAVs, and holding a target only adds to
    the own-rescue term.

    Earlier forms, for the record:
      - sum_k (p_r,k(t) - p_r,k(t-1)), Eq. (25): telescopes to p_r at the end
        minus p_r at the start.
      - sum_k p_r,k(t) alone: with a birth process every target is eventually
        rescued, so it averages P_BIRTH / |U| whatever the policy does and does
        not see how LONG targets wait.

    Sets and p_r are taken BEFORE this slot's rescue removals, so a target
    rescued this slot is counted for the slot it did wait.
    """
    N = len(uavs)
    pr_now  = info.get("rescue_prob", {})
    sets    = info.get("assignments_pre", info.get("assignments", {}))
    backlog = float(info.get("n_waiting", info.get("backlog", len(pr_now))))

    rewards = np.full(N, -backlog / N, dtype=np.float32)
    for i in range(N):
        rewards[i] += float(sum(pr_now.get(k, 0.0) for k in sets.get(i, ())))
    return rewards
