import numpy as np
from config.params import (
    MAP_SIZE, V_MAX, V_MAX_TARGET, DT, SNR_MIN, R_MIN,
    NUM_UAVS, NUM_TARGETS,
)

# Local observation per agent:
#   o^t_i = [s^t_i, mu^t_k, rho^t_i, tau^t_i, SNR^t_i,k, R^t_i, sigma^t_k,
#            rel-pos of the K nearest other UAVs]
# The neighbour block makes the collision penalty actionable: without it the shared
# actor cannot see another UAV closing in, so d < D_MIN is unavoidable for it.
# Exposing the *nearest K* (not all UAVs) keeps OBS_DIM fixed as the swarm scales.
_N_NEIGH = 2                    # nearest other UAVs exposed to the actor
OBS_DIM  = 13 + 2 * _N_NEIGH

_SIGMA_REF = 500.0
# Neighbour relative position is scaled by a collision-relevant radius (not MAP_SIZE)
# and clipped, so the signal is strong near D_MIN and saturates to "far, ignore".
_NEIGH_REF = 500.0             # m, 5x D_MIN

# Covariance normalisation references for the critic's global state.
_POS_VAR_REF = _SIGMA_REF ** 2            # m²    (position variance)
_VEL_VAR_REF = V_MAX_TARGET ** 2          # (m/s)² (velocity variance)
_POSVEL_REF  = _SIGMA_REF * V_MAX_TARGET  # position-velocity covariance

# Global (centralised) state for the critic — privileged, training-only.
#   S^t = { per UAV: [s_i, A_i (one-hot), rho_i, gamma_i],
#           per target: [mu_k, Sigma_k (full 4x4)] }
GLOBAL_OBS_DIM = NUM_UAVS * (6 + NUM_TARGETS) + NUM_TARGETS * 14


def local_obs(state: dict, uav_id: int, info: dict = None) -> np.ndarray:
    x, y, vx, vy = state["uavs"][uav_id]
    k             = state["assignments"].get(uav_id)
    if k is None:
        return np.zeros(OBS_DIM, dtype=np.float32)
    mu, Sigma = state["targets"][k]
    rho       = state["rho"][uav_id]
    tau       = state["tau"].get(uav_id, 0.0) if isinstance(state["tau"], dict) else state["tau"][uav_id]

    snr_lin = info["snr_linear"][uav_id] if info is not None else 0.0
    rate    = info["rate_Mbps"][uav_id] * 1e6 if info is not None else 0.0

    sigma_pos  = float(np.sqrt(np.trace(Sigma[:2, :2])))
    sigma_norm = min(sigma_pos / _SIGMA_REF, 1.0)

    # Nearest-K other UAVs: relative position, scaled to the collision radius and
    # clipped. Missing slots (fewer than K others) pad to the "far" sentinel so the
    # block is a no-op. Relative position + shared policy breaks the collision
    # symmetry: if A sees B east and B sees A west, both steer apart.
    # Only UAVs the BS still considers active are included: once a UAV has been
    # silent for N_FAIL_DETECT slots the BS declares it failed, and it must stop
    # appearing as a phantom obstacle frozen at its last position (state["active"]).
    active = state.get("active", {})
    neigh = []
    others = [
        (state["uavs"][j][0] - x, state["uavs"][j][1] - y)
        for j in state["uavs"]
        if j != uav_id and active.get(j, True)
    ]
    others.sort(key=lambda d: d[0] * d[0] + d[1] * d[1])
    for dxn, dyn in others[:_N_NEIGH]:
        neigh.append(float(np.clip(dxn / _NEIGH_REF, -1.0, 1.0)))
        neigh.append(float(np.clip(dyn / _NEIGH_REF, -1.0, 1.0)))
    neigh.extend([1.0] * (2 * _N_NEIGH - len(neigh)))  # pad missing → far

    return np.array([
        x / MAP_SIZE,
        y / MAP_SIZE,
        vx / V_MAX,
        vy / V_MAX,
        (mu[0] - x) / MAP_SIZE,
        (mu[1] - y) / MAP_SIZE,
        mu[2] / V_MAX_TARGET,
        mu[3] / V_MAX_TARGET,
        rho,
        tau / DT,
        min(snr_lin / SNR_MIN, 20.0) / 20.0,
        min(rate / R_MIN, 50.0) / 50.0,
        sigma_norm,
        *neigh,
    ], dtype=np.float32)


def _gamma_norm(gamma: float) -> float:
    """Uplink SNR (linear) → dB, normalised to ~[-2, 2] for the critic."""
    g_db = 10.0 * np.log10(max(gamma, 1e-10))
    return float(np.clip(g_db / 50.0, -2.0, 2.0))


def global_state(state: dict, info: dict = None) -> np.ndarray:
    """
    Centralised BS state S^t for the critic (CTDE). Privileged, training-only.

    Per UAV i : [x, y, vx, vy, rho_i, gamma_i, one-hot assignment A_i]
    Per target k: [mu_k (pos+vel), Sigma_k (10 unique entries of the 4x4 covariance)]

    Unlike local_obs, this is scenario-sized (depends on NUM_UAVS/NUM_TARGETS),
    which is fine: the critic is discarded at execution and only used in training.
    """
    gammas = state.get("gamma", {})
    parts: list = []

    for i in range(NUM_UAVS):
        x, y, vx, vy = state["uavs"][i]
        rho   = state["rho"][i]
        gamma = gammas.get(i, 0.0) if isinstance(gammas, dict) else 0.0
        k     = state["assignments"].get(i)
        onehot = [0.0] * NUM_TARGETS
        if k is not None and 0 <= k < NUM_TARGETS:
            onehot[k] = 1.0
        parts.extend([
            x / MAP_SIZE, y / MAP_SIZE, vx / V_MAX, vy / V_MAX,
            rho, _gamma_norm(gamma), *onehot,
        ])

    for k in range(NUM_TARGETS):
        mu, Sigma = state["targets"][k]
        parts.extend([
            mu[0] / MAP_SIZE, mu[1] / MAP_SIZE,
            mu[2] / V_MAX_TARGET, mu[3] / V_MAX_TARGET,
        ])
        # 10 unique entries of the symmetric 4x4 covariance, block-normalised.
        s = Sigma
        ent = [
            s[0, 0] / _POS_VAR_REF, s[1, 1] / _POS_VAR_REF, s[0, 1] / _POS_VAR_REF,  # position block
            s[2, 2] / _VEL_VAR_REF, s[3, 3] / _VEL_VAR_REF, s[2, 3] / _VEL_VAR_REF,  # velocity block
            s[0, 2] / _POSVEL_REF,  s[0, 3] / _POSVEL_REF,                            # position-velocity
            s[1, 2] / _POSVEL_REF,  s[1, 3] / _POSVEL_REF,
        ]
        parts.extend([float(np.clip(e, -5.0, 5.0)) for e in ent])

    return np.array(parts, dtype=np.float32)
