import numpy as np
from config.params import F_MAT, Q_MAT


def _inv2x2(M):
    """Explicit inverse for a 2×2 matrix — avoids LAPACK overhead."""
    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    return np.array([[M[1, 1], -M[0, 1]], [-M[1, 0], M[0, 0]]]) / det


def jacobian(mu, uav_pos2d):
    dx = mu[0] - uav_pos2d[0]
    dy = mu[1] - uav_pos2d[1]
    r2 = max(dx**2 + dy**2, 1e-6)
    r  = np.sqrt(r2)
    return np.array([
        [ dx / r,   dy / r,  0.0, 0.0],
        [-dy / r2,  dx / r2, 0.0, 0.0],
    ])


def predict(mu, Sigma):
    mu_pred    = F_MAT @ mu
    Sigma_pred = F_MAT @ Sigma @ F_MAT.T + Q_MAT
    return mu_pred, Sigma_pred


def update(mu_pred, Sigma_pred, uav_pos2d, z, R):
    H = jacobian(mu_pred, uav_pos2d)

    dx     = mu_pred[0] - uav_pos2d[0]
    dy     = mu_pred[1] - uav_pos2d[1]
    z_pred = np.array([max(np.sqrt(dx**2 + dy**2), 1e-3), np.arctan2(dy, dx)])

    K   = Sigma_pred @ H.T @ _inv2x2(H @ Sigma_pred @ H.T + R)
    inn = z - z_pred
    inn[1] = (inn[1] + np.pi) % (2.0 * np.pi) - np.pi   # wrap bearing

    mu_new    = mu_pred + K @ inn
    I_KH      = np.eye(4) - K @ H
    Sigma_new = I_KH @ Sigma_pred @ I_KH.T + K @ R @ K.T  # Joseph form
    return mu_new, Sigma_new
