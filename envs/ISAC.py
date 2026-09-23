import numpy as np
from config.params import (
    SNR0, TAU0, R0, SNR_MIN, TAU_SENSE, TAU_COMM,
    SIGMA_R2_0, SIGMA_THETA2_0,
    PTX, N0, H, LAMBDA, GT, GC,
    B, R_MIN, DT,
)


def _slant_range(uav, target_pos2d):
    """3D range from UAV to a ground target: horizontal distance plus altitude H.

    The radar echo travels this slant path, not the horizontal projection — using
    the horizontal-only distance lets SNR diverge as a UAV passes overhead, which
    is unphysical. Slant range bounds SNR (and the resulting measurement
    precision) naturally, without an artificial noise floor.
    """
    dx = target_pos2d[0] - uav.x
    dy = target_pos2d[1] - uav.y
    return max(float(np.sqrt(dx**2 + dy**2 + H**2)), 1e-3)


# ---------------------------------------------------------------- radar sensing
def radar_snr(uav, target_pos2d, tau=TAU_SENSE):
    """Radar SNR of the echo from a target — Eq. (7).

    SNR = SNR0 * (tau / tau0) * (r / r0)^-4

    `tau` is the radar dwell in seconds and defaults to the slot's fixed sensing
    sub-slot TAU_SENSE. It is exposed as an argument only so reachability can be
    queried for a target the UAV is NOT measuring this slot (see
    MTTEnv._compute_set_snr); it is not a control variable.
    """
    r = _slant_range(uav, target_pos2d)
    return SNR0 * (tau / TAU0) * (r / R0) ** (-4)


def measure(uav, target_pos2d):
    """Range-and-bearing measurement of a target — Eqs. (8)-(10).

    Returns (z, R, snr): the noisy measurement, its noise covariance, and the
    radar SNR that produced it.
    """
    dx = target_pos2d[0] - uav.x
    dy = target_pos2d[1] - uav.y
    # The EKF tracks a 2D state, so range and bearing are the horizontal ones;
    # only the noise magnitude comes from the 3D slant-range SNR.
    r = max(np.sqrt(dx**2 + dy**2), 1e-3)

    snr = radar_snr(uav, target_pos2d)
    if snr <= 0.0:
        return np.array([r, np.arctan2(dy, dx)]), np.eye(2) * 1e10, 0.0

    var_r     = SIGMA_R2_0     / snr
    var_theta = SIGMA_THETA2_0 / snr
    R = np.diag([var_r, var_theta])
    # R is diagonal — sample each component directly, avoiding the SVD inside
    # multivariate_normal.
    z = np.array([
        r                  + np.sqrt(var_r)     * np.random.randn(),
        np.arctan2(dy, dx) + np.sqrt(var_theta) * np.random.randn(),
    ])
    return z, R, snr


# ------------------------------------------------------------------- uplink
def uplink_snr_batch(positions: np.ndarray, bs_pos: np.ndarray) -> np.ndarray:
    """Uplink SNR at the BS for N UAVs at once — Eqs. (11)-(12).

    LoS air-to-ground channel power gain h = Gt*Gc*lambda^2 / ((4*pi)^2 * d^2)
    with d the 3D UAV-BS distance, then gamma = Ptx*h / N0.

    positions: (N, 2), bs_pos: (2,) -> (N,)
    """
    diffs = positions - bs_pos
    d2d   = np.maximum(np.sqrt((diffs * diffs).sum(axis=1)), 1.0)
    d3d   = np.sqrt(d2d * d2d + H * H)
    h     = GT * GC * LAMBDA**2 / ((4.0 * np.pi) ** 2 * d3d ** 2)
    return PTX * h / N0


def uplink_snr(uav, bs) -> float:
    """Scalar form of uplink_snr_batch for a single UAV."""
    return float(uplink_snr_batch(uav.pos2d[None, :], bs.pos)[0])


def uplink_rate(gamma) -> float:
    """Achievable uplink rate over the communication sub-slot — Eq. (13).

    R = (1 - tau) * B * log2(1 + gamma), with the sensing/communication split
    fixed by the ISAC sensing fraction.
    """
    return (TAU_COMM / DT) * B * np.log2(1.0 + gamma)
