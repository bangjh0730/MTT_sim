import numpy as np
from config.params import (MAP_SIZE, V_MAX, V_MAX_TARGET, NUM_UAVS,
                           SIGMA_FEATURE, LAMBDA_RESCUE,
                           SNR0, TAU_SENSE, TAU0, R0, SNR_MIN, H)

# Local observation, Eq. (24), as a SET - no padding and no ceiling on |A_i|:
#   ego (EGO_DIM)          x/MAP, y/MAP, vx/V_MAX, vy/V_MAX, |A|/SET_SCALE
#   one row per member     dx/MAP, dy/MAP, dist/MAP, vx/VT, vy/VT, sigma-tilde,
#   (MEM_DIM)              in-range
#
# in-range is 1 when the member's estimate lies inside the detection footprint
# (SNR >= SNR_min) - information the UAV has from its own position and the
# estimate, not a sensing rule. Which member to sense is the actor's choice.
# Rows are ordered by distance only for determinism; the network is
# permutation-equivariant over them.
EGO_DIM   = 5
MEM_DIM   = 7
SET_SCALE = 10.0   # feature scale for set sizes, not a bound

# Critic state (CTDE, training only), also as sets, from agent i's viewpoint:
#   self (EGO_DIM)            agent i's ego features
#   one row per UAV (U_DIM)   dx/MAP, dy/MAP rel. to i, vx, vy, |A_j|/SET_SCALE, is_i
#   one row per target (T_DIM) dx/MAP, dy/MAP rel. to i, x/MAP, y/MAP, vx, vy,
#                             sigma-tilde, held_by_i, held_by_other, unassigned,
#                             dist-to-holder/MAP, in-range-of-holder
U_DIM = 6
T_DIM = 12

# Horizontal radius of the detection footprint, from SNR(r_slant) = SNR_min in
# Eq. (7). Squared, for a cheap comparison.
_R_SLANT_DET = R0 * (SNR0 * (TAU_SENSE / TAU0) / SNR_MIN) ** 0.25
R_DETECT_2   = max(_R_SLANT_DET ** 2 - H ** 2, 0.0)


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


def _ego(state: dict, uav_id: int) -> np.ndarray:
    x, y, vx, vy = state["uavs"][uav_id]
    n = len(state["assignments"].get(uav_id, ()) or ())
    return np.array([x / MAP_SIZE, y / MAP_SIZE, vx / V_MAX, vy / V_MAX,
                     n / SET_SCALE], dtype=np.float32)


def local_obs(state: dict, uav_id: int, info: dict = None):
    """Returns (ego [EGO_DIM], members [|A_i|, MEM_DIM], member ids [|A_i|]).

    Row j of `members` is target member_ids[j]; the actor's sensing choice is
    an index into these rows.
    """
    x, y = state["uavs"][uav_id][:2]
    ids  = _members_by_distance(state, uav_id, x, y)

    mem = np.zeros((len(ids), MEM_DIM), dtype=np.float32)
    for j, k in enumerate(ids):
        mu, Sigma = state["targets"][k]
        dx, dy = mu[0] - x, mu[1] - y
        d2 = dx * dx + dy * dy
        mem[j] = (dx / MAP_SIZE, dy / MAP_SIZE, np.sqrt(d2) / MAP_SIZE,
                  mu[2] / V_MAX_TARGET, mu[3] / V_MAX_TARGET,
                  _sigma_scalar(Sigma), 1.0 if d2 <= R_DETECT_2 else 0.0)
    return _ego(state, uav_id), mem, ids


def critic_obs(state: dict, uav_id: int):
    """Returns (self [EGO_DIM], uavs [|U|, U_DIM], targets [|K|, T_DIM])."""
    x, y = state["uavs"][uav_id][:2]

    holder = {}
    for i in state["uavs"]:
        for k in state["assignments"].get(i, ()) or ():
            holder[k] = i

    uav_rows = np.zeros((len(state["uavs"]), U_DIM), dtype=np.float32)
    for r, j in enumerate(sorted(state["uavs"])):
        xj, yj, vxj, vyj = state["uavs"][j]
        n = len(state["assignments"].get(j, ()) or ())
        uav_rows[r] = ((xj - x) / MAP_SIZE, (yj - y) / MAP_SIZE, vxj / V_MAX,
                       vyj / V_MAX, n / SET_SCALE, 1.0 if j == uav_id else 0.0)

    ks = sorted(state["targets"])
    tgt_rows = np.zeros((len(ks), T_DIM), dtype=np.float32)
    for r, k in enumerate(ks):
        mu, Sigma = state["targets"][k]
        h = holder.get(k)
        if h is not None:
            hx, hy = state["uavs"][h][:2]
            dh2 = (mu[0] - hx) ** 2 + (mu[1] - hy) ** 2
            dh, in_rng = np.sqrt(dh2) / MAP_SIZE, 1.0 if dh2 <= R_DETECT_2 else 0.0
        else:
            dh, in_rng = 0.0, 0.0
        tgt_rows[r] = ((mu[0] - x) / MAP_SIZE, (mu[1] - y) / MAP_SIZE,
                       mu[0] / MAP_SIZE, mu[1] / MAP_SIZE,
                       mu[2] / V_MAX_TARGET, mu[3] / V_MAX_TARGET,
                       _sigma_scalar(Sigma),
                       1.0 if h == uav_id else 0.0,
                       1.0 if (h is not None and h != uav_id) else 0.0,
                       1.0 if h is None else 0.0,
                       dh, in_rng)
    return _ego(state, uav_id), uav_rows, tgt_rows


def pad_sets(rows: list, dim: int):
    """List of [n_b, dim] arrays -> (padded [B, n_max, dim], mask [B, n_max]).

    n_max is the largest set in this batch, at least 1 so empty sets still
    produce a tensor; masked rows are ignored by the networks.
    """
    n_max = max(1, max((len(r) for r in rows), default=0))
    out  = np.zeros((len(rows), n_max, dim), dtype=np.float32)
    mask = np.zeros((len(rows), n_max), dtype=bool)
    for b, r in enumerate(rows):
        if len(r):
            out[b, :len(r)] = r
            mask[b, :len(r)] = True
    return out, mask
