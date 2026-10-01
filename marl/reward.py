import math

import numpy as np

# Weight on the track term, relative to +1 for a completed rescue. At 5 the
# track term is ~4x the completion bonus over a target's life and re-acquiring a
# dead track outpays finishing one - chosen to beat the travel cost of fetching a
# distant member. Compare against 0, which gave 29 rescues and D 59 s but left
# 18% of assigned targets never sensed.
W_TRACK = 5.0


def _track_quality(tr: float) -> float:
    """Track quality in [0, 1] from tr(Sigma_pos): 1 is pinned, 0 is lost.

    log10 rather than p_r, which is already 3e-7 at sigma = 71 m and so charges
    nothing for a target left to rot. 0.110 at birth, 0.875 sensed from 200 m,
    0 past sigma = 1000 m.
    """
    return min(max(1.0 - (math.log10(max(tr, 1e-4)) + 4.0) / 10.0, 0.0), 1.0)


def per_agent_rewards(info: dict, uavs: dict) -> np.ndarray:
    """r_u = (members of A_u rescued) + W_TRACK * sum_{k in A_u} [u_k(t) - u_k(t-1)]

    Reward a rescue, reward holding the tracks of members still waiting. A
    rescued member scores +1 and its u change is not also counted; a member on
    its first slot has tr_prev seeded to its birth covariance, so it contributes
    nothing rather than a spurious jump.

    Not -|A_u|: that level is set by the assigner, and a birth raises it through
    no act of the policy. A change in u comes only from sensing or failing to.

    Not policy-invariant, deliberately. Shaping is invariant only as
    gamma*u(t) - u(t-1); written undiscounted it adds a standing reward of about
    (1-gamma)*u per slot per member, so the objective is delay plus a preference
    for keeping members tracked. That bias is what stops the policy writing off
    distant members, and it cannot be farmed by hoarding a pinned track because
    rescue fires involuntarily once p_r is high.

    Needs info keys: assignments_pre, rescued, trace_pos_per_target,
    trace_pos_per_target_prev.
    """
    N       = len(uavs)
    sets    = info.get("assignments_pre", info.get("assignments", {}))
    rescued = set(info.get("rescued", ()))
    tr_now  = info.get("trace_pos_per_target", {})
    tr_prev = info.get("trace_pos_per_target_prev", {})

    rewards = np.zeros(N, dtype=np.float32)
    for i in range(N):
        saved, track = 0, 0.0
        for k in sets.get(i, ()):
            if k in rescued:
                saved += 1
            elif k in tr_now and k in tr_prev:
                track += _track_quality(tr_now[k]) - _track_quality(tr_prev[k])
        rewards[i] = float(saved) + W_TRACK * float(track)
    return rewards
