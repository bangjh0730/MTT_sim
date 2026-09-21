import numpy as np
from config.params import (
    F_MAT, Q_MAT, SNR_MIN,
    SIGMA_R2_0, SIGMA_THETA2_0,
)
from envs.ekf import jacobian


_REG = 1e-4 * np.eye(4)   # regularisation for rank-2 Q_MAT


def predict_bfim(J_prev: np.ndarray) -> np.ndarray:
    """BFIM prediction step.

    Q_MAT is rank-2 (built from 2-D acceleration noise on a 4-D state), so
    inv(Q_MAT) is undefined.  When J_prev is large its inverse is near-zero,
    making the argument of the outer inv() near-singular.  Adding a small
    regulariser keeps eigenvalues finite and caps J at ~1/1e-4 = 1e4.
    """
    inner = Q_MAT + F_MAT @ np.linalg.solve(J_prev, F_MAT.T)
    return np.linalg.inv(inner + _REG)


_F_T = None   # lazily cached transpose to avoid repeated attribute lookup


def predict_bfim_batch(J_batch: np.ndarray) -> np.ndarray:
    """Batched BFIM prediction for K targets. J_batch: (K,4,4) → (K,4,4)."""
    global _F_T
    if _F_T is None:
        _F_T = np.ascontiguousarray(F_MAT.T)
    K     = J_batch.shape[0]
    X     = np.linalg.solve(J_batch, np.broadcast_to(_F_T, (K, 4, 4)))  # (K,4,4)
    inner = Q_MAT + F_MAT @ X                        # (K,4,4) via broadcast
    return np.linalg.inv(inner + _REG)


def observation_info(uav_pos2d: np.ndarray, mu_pred: np.ndarray, snr: float) -> np.ndarray:
    """Fisher information from one UAV measurement."""
    if snr >= SNR_MIN:
        H     = jacobian(mu_pred, uav_pos2d)
        # R is diagonal (σ0²/SNR) → inv(R) = diag of reciprocals. SNR itself is
        # bounded above by the 3D slant-range path loss (radar_snr uses altitude
        # H, not horizontal distance), so R never collapses to 0 and J stays
        # finite without an artificial measurement-noise floor.
        R_inv = np.diag([
            1.0 / (SIGMA_R2_0     / snr),
            1.0 / (SIGMA_THETA2_0 / snr),
        ])
        return H.T @ R_inv @ H
    return np.zeros((4, 4))


def pcrlb(J):
    """Position PCRLB for one target: tr of the 2x2 position block of J^{-1}."""
    return float(np.trace(np.linalg.inv(J)[:2, :2]))


def pcrlb_batch(J_batch: np.ndarray) -> np.ndarray:
    """Batched PCRLB for K targets. J_batch: (K,4,4) → (K,) position PCRLB values."""
    J_inv = np.linalg.inv(J_batch)          # (K, 4, 4)
    return J_inv[:, 0, 0] + J_inv[:, 1, 1] # tr of 2×2 position block per target


