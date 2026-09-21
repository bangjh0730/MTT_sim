import numpy as np
from config.params import (
    SNR0, TAU0, R0, SNR_MIN,
    SIGMA_R2_0, SIGMA_THETA2_0,
    PTX, N0, FC, H,
    ETA_LOS, ETA_NLOS, C1, C2,
    B, R_MIN, DT,
)


def _slant_range(uav, target_pos2d):
    """3D range from UAV to a ground target: horizontal distance plus altitude H.
    The radar echo travels this slant path, not the horizontal projection — using
    the horizontal-only distance lets SNR diverge as a UAV passes overhead, which
    is unphysical. Slant range bounds SNR (and hence the Fisher information) above
    naturally, without an artificial measurement-noise floor."""
    dx = target_pos2d[0] - uav.x
    dy = target_pos2d[1] - uav.y
    return max(float(np.sqrt(dx**2 + dy**2 + H**2)), 1e-3)


def radar_snr(uav, target_pos2d):
    r = _slant_range(uav, target_pos2d)
    return SNR0 * (uav.tau / TAU0) * (r / R0) ** (-4)


def measure(uav, target_pos2d):
    """Returns (z, R, snr): noisy range-and-bearing measurement and its noise covariance."""
    dx = target_pos2d[0] - uav.x
    dy = target_pos2d[1] - uav.y
    r  = max(np.sqrt(dx**2 + dy**2), 1e-3)   # horizontal range/bearing measured by the EKF (2D state)

    snr = radar_snr(uav, target_pos2d)       # noise magnitude driven by the 3D slant-range SNR
    if snr <= 0.0:
        return np.array([r, np.arctan2(dy, dx)]), np.eye(2) * 1e10, 0.0

    var_r     = SIGMA_R2_0     / snr
    var_theta = SIGMA_THETA2_0 / snr
    R = np.diag([var_r, var_theta])
    # R is diagonal — sample each component directly, avoiding SVD inside multivariate_normal.
    z = np.array([
        r                    + np.sqrt(var_r)     * np.random.randn(),
        np.arctan2(dy, dx)   + np.sqrt(var_theta) * np.random.randn(),
    ])
    return z, R, snr


def uplink_snr(uav, bs):
    d = max(float(np.linalg.norm(bs.pos - uav.pos2d)), 1.0)
    fspl    = 20.0 * np.log10(4.0 * np.pi * FC * np.sqrt(d**2 + H**2) / 3e8)
    theta   = np.degrees(np.arctan2(H, d))
    p_los   = 1.0 / (1.0 + C1 * np.exp(-C2 * (theta - C1)))
    L_dB    = p_los * (fspl + ETA_LOS) + (1.0 - p_los) * (fspl + ETA_NLOS)
    return PTX * 10.0 ** (-L_dB / 10.0) / N0


def uplink_snr_batch(positions: np.ndarray, bs_pos: np.ndarray) -> np.ndarray:
    """Vectorised uplink SNR for N UAVs at once. positions: (N,2), bs_pos: (2,) → (N,)."""
    diffs = positions - bs_pos                                  # (N, 2)
    d     = np.maximum(np.sqrt((diffs * diffs).sum(axis=1)), 1.0)  # (N,)
    d3d   = np.sqrt(d * d + H * H)
    fspl  = 20.0 * np.log10(4.0 * np.pi * FC * d3d / 3e8)
    theta = np.degrees(np.arctan2(H, d))
    p_los = 1.0 / (1.0 + C1 * np.exp(-C2 * (theta - C1)))
    L_dB  = p_los * (fspl + ETA_LOS) + (1.0 - p_los) * (fspl + ETA_NLOS)
    return PTX * 10.0 ** (-L_dB / 10.0) / N0


def uplink_rate(uav, bs, gamma=None):
    if gamma is None:
        gamma = uplink_snr(uav, bs)
    comm_frac = max((DT - uav.tau) / DT, 0.0)
    return comm_frac * B * np.log2(1.0 + gamma)


def feasible_dwell_time(uav, target_pos2d, bs, gamma=None):
    if gamma is None:
        gamma = uplink_snr(uav, bs)
    r       = _slant_range(uav, target_pos2d)
    tau_min = (SNR_MIN * TAU0 / SNR0) * (r / R0) ** 4

    comm_cap = B * np.log2(1.0 + gamma)
    tau_max  = DT * (1.0 - R_MIN / comm_cap) if comm_cap > R_MIN else 0.0

    if tau_min > DT:
        # Target beyond max sensing range; no dwell needed — sensing would fail regardless.
        # Lower bound becomes 0 so the full slot is available for uplink.
        return 0.0, max(tau_max, 0.0)

    return tau_min, max(tau_max, tau_min)
