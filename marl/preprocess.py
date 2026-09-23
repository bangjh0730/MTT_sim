import numpy as np
from config.params import (MAP_SIZE, V_MAX, V_MAX_TARGET, NUM_UAVS, NUM_TARGETS,
                           MAX_TARGETS, SIGMA_FEATURE, LAMBDA_RESCUE)

# Local observation, Eq. (24):
#   ego (4)            x/MAP, y/MAP, vx/V_MAX, vy/V_MAX
#   per target (4M)    dx/MAP, dy/MAP, sigma-tilde, valid   -- padded to M
#   set summary (4)    |A|/K_MAX, centroid dx/MAP, dy/MAP, spread/MAP
#
# Targets are sorted by DISTANCE, a neutral key: sorting by urgency would hand
# the policy the priority ranking it is supposed to learn, and the slot a target
# occupies would jump around as its uncertainty evolved. Priority is still
# visible through sigma-tilde in each slot.
M_TARGETS = NUM_TARGETS
K_MAX     = NUM_TARGETS
OBS_DIM   = 4 + 4 * M_TARGETS + 4

# Critic state (CTDE, training only). Every id slot in [0, MAX_TARGETS) has a
# fixed position with an alive flag, since the live id set changes on every
# birth and rescue; packing only live targets would shift every feature.
GLOBAL_OBS_DIM = NUM_UAVS * 5 + MAX_TARGETS * (6 + NUM_UAVS)


def _sigma_scalar(Sigma) -> float:
    """Per-target uncertainty scalar of Eq. (24), normalised to [0, 1]."""
    tr = float(np.trace(Sigma[:2, :2]))
    if SIGMA_FEATURE == "log":
        return float(np.clip(1.0 - (np.log10(max(tr, 1e-4)) + 4.0) / 10.0, 0.0, 1.0))
    return float(LAMBDA_RESCUE / (LAMBDA_RESCUE + max(tr, 0.0)))


def _members_by_distance(state: dict, uav_id: int, x: float, y: float) -> list:
    ks = [k for k in state["assignments"].get(uav_id, ()) if k in state["targets"]]
    def _d2(k):
        mu = state["targets"][k][0]
        return (mu[0] - x) ** 2 + (mu[1] - y) ** 2
    return sorted(ks, key=_d2)


def local_obs(state: dict, uav_id: int, info: dict = None) -> np.ndarray:
    x, y, vx, vy = state["uavs"][uav_id]

    obs = np.zeros(OBS_DIM, dtype=np.float32)
    obs[0] = x / MAP_SIZE
    obs[1] = y / MAP_SIZE
    obs[2] = vx / V_MAX
    obs[3] = vy / V_MAX

    members = _members_by_distance(state, uav_id, x, y)

    # Only the M nearest members get a slot; any surplus is still reflected in
    # the set summary below.
    for j, k in enumerate(members[:M_TARGETS]):
        mu = state["targets"][k][0]
        b = 4 + 4 * j
        obs[b + 0] = (mu[0] - x) / MAP_SIZE
        obs[b + 1] = (mu[1] - y) / MAP_SIZE
        obs[b + 2] = _sigma_scalar(state["targets"][k][1])
        obs[b + 3] = 1.0

    b = 4 + 4 * M_TARGETS
    obs[b + 0] = min(len(members) / K_MAX, 2.0)
    if members:
        pos = np.array([state["targets"][k][0][:2] for k in members])
        centroid = pos.mean(axis=0)
        obs[b + 1] = (centroid[0] - x) / MAP_SIZE
        obs[b + 2] = (centroid[1] - y) / MAP_SIZE
        obs[b + 3] = float(np.sqrt(((pos - centroid) ** 2).sum(axis=1).mean())) / MAP_SIZE

    return obs


def global_state(state: dict, info: dict = None) -> np.ndarray:
    """Centralised state for the critic. Privileged, training only."""
    parts: list = []

    for i in range(NUM_UAVS):
        x, y, vx, vy = state["uavs"][i]
        load = len(state["assignments"].get(i, ()) or ())
        parts.extend([x / MAP_SIZE, y / MAP_SIZE, vx / V_MAX, vy / V_MAX,
                      min(load / K_MAX, 2.0)])

    pr_map = state.get("rescue_prob", {})
    holder = {}
    for i in range(NUM_UAVS):
        for k in state["assignments"].get(i, ()) or ():
            holder[k] = i

    for k in range(MAX_TARGETS):
        if k not in state["targets"]:
            parts.extend([0.0] * (6 + NUM_UAVS))
            continue
        mu = state["targets"][k][0]
        onehot = [0.0] * NUM_UAVS
        if k in holder:
            onehot[holder[k]] = 1.0
        parts.extend([1.0,
                      mu[0] / MAP_SIZE, mu[1] / MAP_SIZE,
                      mu[2] / V_MAX_TARGET, mu[3] / V_MAX_TARGET,
                      pr_map.get(k, 0.0), *onehot])

    return np.array(parts, dtype=np.float32)
