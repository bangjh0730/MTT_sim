import numpy as np

from config.params import D_MIN, R_MIN, SNR_MIN

# ---------------------------------------------------------------------------
# Per-agent reward — SET-CENTRIC.
#
#   r^t_i = a1 * log2(SNR_sensed / SNR_min)            measurement quality
#         + a_reach * mean_{k in A_i} reach(k)          SET COVERAGE (positioning)
#         + a5 * mean_{k in A_i} p_r(k)                 mission objective
#         + a6 * (targets rescued out of A_i)           terminal event
#         + a2 * log2(R / R_min)                        uplink
#         - a3 * E - a4 * collisions
#
# Why the SNR term is no longer dominant. A UAV takes ONE measurement per slot,
# on whichever member the scheduler picked (the neediest). If the reward is
# mostly "maximise the SNR of that measurement", the optimal behaviour is to fly
# straight at that member — and since sensing it collapses its tr(Sigma), a
# DIFFERENT member is neediest next slot, so the actor is yanked toward a new
# target every slot and commutes between them forever, serving none well. The
# single-measurement SNR is a real quantity and still earns a term, but it is the
# wrong thing to maximise on its own.
#
# What the UAV is actually for is holding a position from which the WHOLE set
# stays serviceable, so the two set-wide terms carry the weight:
#
#   a_reach (dense, positional): the mean reachability of the set's members from
#     where the UAV is standing — the would-be radar SNR at each member, in
#     octaves above the detection floor. This is the only term that can express
#     "this spot serves four of them, that spot serves one", so it is what makes
#     the vantage-point behaviour learnable. It responds to position immediately,
#     every slot, for every member, whether or not that member was the one
#     measured — which is exactly the credit the SNR term cannot give.
#
#   a5 (dense, outcome): the mean rescue probability across the set. This is the
#     mission objective itself, read per-UAV. Because p_r = lambda/(lambda+tr Sigma)
#     saturates, its gradient is steep for a member whose estimate has decayed and
#     flat for one already well-tracked, so it pushes toward keeping EVERY member
#     serviceable rather than perfecting one and starving the rest.
#
# Assignment remains the agentic AI's decision, not the actor's. These terms only
# shape how well a UAV serves the set it was handed.
# ---------------------------------------------------------------------------

ALPHA1_POS  = 0.5   # sensed-target SNR reward  (log2(SNR/SNR_min) when >= min)
ALPHA1_NEG  = 1.0   # sensed-target SNR penalty (when below min)
ALPHA2_POS  = 1.0   # uplink rate reward
ALPHA2_NEG  = 2.0   # uplink rate penalty
ALPHA3      = 0.01  # energy penalty
ALPHA4      = 50.0  # collision penalty (per other UAV within D_MIN)
ALPHA_REACH = 8.0   # set coverage — the positioning signal
ALPHA5      = 10.0  # mean rescue probability over the set
ALPHA6      = 5.0   # per target rescued out of this UAV's set

_REACH_OCTAVES = 4.0   # must match marl/preprocess.py::_REACH_OCTAVES

# Both log-ratio terms are CLIPPED to this many octaves. Unclipped they are
# unbounded below -- a UAV pointing at a target well out of range sees
# log2(SNR/SNR_min) near -40, and a UAV with a dead uplink sees the same on the
# rate term. Either one would be five times the size of every set-wide term
# combined, which would quietly undo the whole point of this reward: the single
# largest lever available to the actor would once again be "raise the SNR of the
# one target I am pointing at", i.e. fly at it and abandon the set. Clipping
# keeps both terms informative near their thresholds, where the actor can
# actually act on them, and stops them dominating far from it.
_LOG_CLIP = 4.0


def _reach(snr: float) -> float:
    """Would-be radar SNR at a set member -> normalised reachability in [-1, 1].

    +1: comfortably detectable from where the UAV is now.
    -1: unreachable without moving.
    Shared definition with the observation, so the actor is rewarded on exactly
    the quantity it is shown.
    """
    if snr <= 0.0:
        return -1.0
    return float(np.clip(np.log2(snr / SNR_MIN) / _REACH_OCTAVES, -1.0, 1.0))


def per_agent_rewards(info: dict, uavs: dict) -> np.ndarray:
    """
    Compute per-agent reward.
    Returns array of shape [N] ordered by UAV id 0..N-1.
    """
    N = len(uavs)
    rewards = np.zeros(N, dtype=np.float32)

    # Vectorised collision count — squared distance avoids N*(N-1) sqrt calls.
    positions = np.array([uavs[i].pos2d for i in range(N)])    # (N, 2)
    diff      = positions[:, None, :] - positions[None, :, :]  # (N, N, 2)
    sq_dists  = (diff * diff).sum(axis=-1)                     # (N, N)
    np.fill_diagonal(sq_dists, np.inf)
    n_colls   = (sq_dists < D_MIN * D_MIN).sum(axis=1)         # (N,)

    pr_map      = info.get("rescue_prob_per_target", {})
    assignments = info.get("assignments", {})
    sensing     = info.get("sensing", {})
    set_snr     = info.get("set_snr", {})

    # A rescued target has already been dropped from its holder's set by the env,
    # so credit goes to the UAV that was sensing it in the slot it was rescued —
    # the same UAV, and the one whose flying earned it.
    rescued_by = {}
    for k in info.get("rescued", []):
        for i, ks in sensing.items():
            if ks == k:
                rescued_by[i] = rescued_by.get(i, 0) + 1

    for i in range(N):
        snr    = info["snr_linear"][i]
        rate   = info["rate_Mbps"][i] * 1e6       # bps
        energy = info["slot_energy_J"][i]         # J, consumed this slot

        members = [k for k in assignments.get(i, ()) if k in pr_map]

        # The one measurement actually taken this slot.
        log_snr = float(np.clip(np.log2(max(snr, 1e-10) / SNR_MIN),
                                -_LOG_CLIP, _LOG_CLIP))
        alpha1  = ALPHA1_POS if log_snr >= 0.0 else ALPHA1_NEG
        # A UAV with no set takes no measurement; do not charge it the
        # below-floor penalty for a measurement it was never asked to make.
        sensed_term = alpha1 * log_snr if members else 0.0

        log_ratio = float(np.clip(np.log2(max(rate, 1.0) / R_MIN),
                                  -_LOG_CLIP, _LOG_CLIP))
        alpha2    = ALPHA2_POS if log_ratio >= 0.0 else ALPHA2_NEG

        snr_i = set_snr.get(i, {}) or {}
        mean_reach = (float(np.mean([_reach(snr_i.get(k, 0.0)) for k in members]))
                      if members else 0.0)
        mean_pr    = (float(np.mean([pr_map[k] for k in members]))
                      if members else 0.0)

        rewards[i] = (
            sensed_term
            + ALPHA_REACH * mean_reach
            + ALPHA5 * mean_pr
            + ALPHA6 * rescued_by.get(i, 0)
            + alpha2 * log_ratio
            - ALPHA3 * energy
            - ALPHA4 * n_colls[i]
        )

    return rewards
