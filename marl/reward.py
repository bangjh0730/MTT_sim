import numpy as np

from config.params import D_MIN, R_MIN, SNR_MIN

# ---- Reward weights ----------------------------------------
# r^t_i = α1±·log2(SNR/SNR_min) + α2±·log2(R/R_min) - α3·E - α4·Σ1[dist<d_min]
# hierarchy: |SNR| > |Rate| > Energy; Collision is a hard-constraint-level penalty
ALPHA1_POS  = 2.0   # radar SNR reward      (log2(SNR/SNR_min) when SNR >= SNR_min)
ALPHA1_NEG  = 4.0   # radar SNR penalty     (log2(SNR/SNR_min) when SNR <  SNR_min, negative → penalty)
ALPHA2_POS  = 1.0   # uplink rate reward    (log2(R/R_min) when R >= R_min)
ALPHA2_NEG  = 2.0   # uplink rate penalty   (log2(R/R_min) when R <  R_min, larger → heavier)
ALPHA3      = 0.01  # energy penalty        (slot_energy_J ≈ 65–354 J → penalty ≈ 0.65–3.54)
ALPHA4      = 50.0  # collision penalty     (number of other UAVs within D_MIN of UAV i → −50 per neighbor)


def per_agent_rewards(info: dict, uavs: dict) -> np.ndarray:
    """
    Compute per-agent reward.
    Returns array of shape [N] ordered by UAV id 0..N-1.
    """
    N = len(uavs)
    rewards = np.zeros(N, dtype=np.float32)

    # Vectorized collision count — squared distance avoids N*(N-1) sqrt calls.
    positions = np.array([uavs[i].pos2d for i in range(N)])    # (N, 2)
    diff      = positions[:, None, :] - positions[None, :, :]  # (N, N, 2)
    sq_dists  = (diff * diff).sum(axis=-1)                     # (N, N)
    np.fill_diagonal(sq_dists, np.inf)
    n_colls   = (sq_dists < D_MIN * D_MIN).sum(axis=1)         # (N,)

    for i in range(N):
        snr    = info["snr_linear"][i]
        rate   = info["rate_Mbps"][i] * 1e6       # bps
        energy = info["slot_energy_J"][i]         # J, consumed this slot

        log_snr     = np.log2(max(snr, 1e-10) / SNR_MIN)
        alpha1      = ALPHA1_POS if log_snr >= 0.0 else ALPHA1_NEG
        log_ratio   = np.log2(max(rate, 1.0) / R_MIN)
        alpha2      = ALPHA2_POS if log_ratio >= 0.0 else ALPHA2_NEG

        rewards[i] = (
            alpha1 * log_snr
            + alpha2 * log_ratio
            - ALPHA3 * energy
            - ALPHA4 * n_colls[i]
        )

    return rewards
