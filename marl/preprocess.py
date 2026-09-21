import numpy as np
from config.params import (
    MAP_SIZE, V_MAX, V_MAX_TARGET, DT, SNR_MIN, R_MIN,
    NUM_UAVS, NUM_TARGETS, MAX_TARGETS,
)

# ---------------------------------------------------------------------------
# Local observation per agent — SET-CENTRIC.
#
# The UAV is not tracking a target. It is serving a SET, one member per slot,
# and every member it does not sense that slot decays: its EKF runs predict-only,
# tr(Sigma) grows and its rescue probability lambda/(lambda + tr Sigma) falls. So
# the UAV's real job is to hold a POSITION from which the whole set stays
# reachable, and to keep cycling, rather than to close on any single target.
#
# The previous observation could not express that job. It exposed only the
# current sensing target's relative position, so the one control signal the actor
# had was "the selected target is over there" — and the optimal response to that
# signal alone is to fly straight at it. Every slot the scheduler then picked a
# different (now-neediest) member, so the actor was pulled back and forth between
# members and commuted instead of covering. The set was invisible to it.
#
# What replaces it:
#   * the set CENTROID and SPREAD, so the actor can find a vantage point;
#   * a fixed-size block of the K neediest members, each with relative position,
#     uncertainty, and how reachable it is FROM HERE (would-be radar SNR), so the
#     actor can see directly whether its current spot serves them or not;
#   * aggregates (load, mean/max uncertainty, reachable fraction) that summarise
#     whatever does not fit in K.
#
# The current sensing target is still flagged, because that is the one
# measurement actually being taken this slot and its SNR is a real reward term —
# but it is one feature among the set's, not the whole observation.
# ---------------------------------------------------------------------------

_K_MEMBERS = 4                  # set members exposed individually to the actor
_N_NEIGH   = 2                  # nearest other UAVs exposed (collision avoidance)

# 4 own-state + 6 set-aggregate + 4 link/energy + K*4 member block + 2*N neighbour
OBS_DIM = 14 + 4 * _K_MEMBERS + 2 * _N_NEIGH

_SIGMA_REF = 500.0
_NEIGH_REF = 500.0             # m, collision-relevant radius
_LOAD_REF  = 6.0               # set size at which the load feature saturates
_SPREAD_REF = 700.0            # m, ~ the sensing radius: a set spread wider than
                               # this cannot be served from one spot

# Reachability: how far above/below the detection floor a member sits, in octaves
# of SNR, squashed to [-1, 1]. +1 means comfortably detectable from here, -1 means
# hopeless without moving. This is the feature that makes "park where you can
# serve them all" learnable at all.
_REACH_OCTAVES = 4.0


def _reach(snr: float) -> float:
    """Would-be radar SNR -> normalised reachability in [-1, 1]."""
    if snr <= 0.0:
        return -1.0
    return float(np.clip(np.log2(snr / SNR_MIN) / _REACH_OCTAVES, -1.0, 1.0))


# Global (centralised) state for the critic — privileged, training-only.
#   S^t = { per UAV: [s_i, rho_i, gamma_i, load_i, A_i (MULTI-hot over the id space)],
#           per target: [alive, mu_k, Sigma_k (10 unique entries)] }
#
# The assignment block is MULTI-hot, not one-hot: a UAV holds a set, so a single
# hot bit cannot represent it. It is sized over the whole id space [0, MAX_TARGETS)
# rather than NUM_TARGETS, because with births outnumbering the initial target
# count the live ids routinely exceed NUM_TARGETS.
GLOBAL_OBS_DIM = NUM_UAVS * (7 + MAX_TARGETS) + MAX_TARGETS * 15


def _members(state: dict, uav_id: int) -> list:
    """Live members of this UAV's set, neediest (highest tr(Sigma)) first."""
    ks = [k for k in state["assignments"].get(uav_id, ()) if k in state["targets"]]
    return sorted(ks, key=lambda k: -float(np.trace(state["targets"][k][1][:2, :2])))


def local_obs(state: dict, uav_id: int, info: dict = None) -> np.ndarray:
    x, y, vx, vy = state["uavs"][uav_id]
    members = _members(state, uav_id)

    if not members:
        # No set: the UAV has nothing to serve. A zero observation is honest here
        # (there is no geometry to reason about) and the reward is correspondingly
        # flat, so the actor is free to loiter until the BS gives it work.
        return np.zeros(OBS_DIM, dtype=np.float32)

    rho = state["rho"][uav_id]
    tau = state["tau"].get(uav_id, 0.0) if isinstance(state["tau"], dict) else state["tau"][uav_id]
    cur = state.get("sensing", {}).get(uav_id)
    snr_map = state.get("set_snr", {}).get(uav_id, {}) or {}

    pos = np.array([[state["targets"][k][0][0], state["targets"][k][0][1]]
                    for k in members])
    centroid = pos.mean(axis=0)
    # Spread as RMS distance from the centroid: the single number that says
    # whether this set can be served from one place or demands a commute.
    spread = float(np.sqrt(((pos - centroid) ** 2).sum(axis=1).mean()))

    sigmas = np.array([float(np.sqrt(np.trace(state["targets"][k][1][:2, :2])))
                       for k in members])
    reaches = np.array([_reach(snr_map.get(k, 0.0)) for k in members])

    rate = info["rate_Mbps"][uav_id] * 1e6 if info is not None else 0.0
    snr_cur = info["snr_linear"][uav_id] if info is not None else 0.0

    obs = [
        # ---- own kinematic state ----
        x / MAP_SIZE,
        y / MAP_SIZE,
        vx / V_MAX,
        vy / V_MAX,
        # ---- the set as a whole ----
        (centroid[0] - x) / MAP_SIZE,          # vector to the set's centre of mass
        (centroid[1] - y) / MAP_SIZE,
        min(spread / _SPREAD_REF, 2.0) / 2.0,  # can one spot serve this set?
        min(len(members) / _LOAD_REF, 1.0),    # how many mouths to feed
        float(np.clip(sigmas.mean() / _SIGMA_REF, 0.0, 1.0)),
        float(np.clip(sigmas.max() / _SIGMA_REF, 0.0, 1.0)),
        # ---- link / energy ----
        rho,
        tau / DT,
        min(snr_cur / SNR_MIN, 20.0) / 20.0,
        min(rate / R_MIN, 50.0) / 50.0,
    ]

    # ---- per-member block, neediest first ----
    # Relative position tells the actor where to go; sigma tells it how urgent the
    # member is; reach tells it whether standing here already serves that member.
    # Together these are what let one vantage point be preferred over another.
    for j in range(_K_MEMBERS):
        if j < len(members):
            k = members[j]
            mu = state["targets"][k][0]
            obs.extend([
                float(np.clip((mu[0] - x) / _SPREAD_REF, -2.0, 2.0)) / 2.0,
                float(np.clip((mu[1] - y) / _SPREAD_REF, -2.0, 2.0)) / 2.0,
                float(np.clip(sigmas[j] / _SIGMA_REF, 0.0, 1.0)),
                reaches[j],
            ])
        else:
            # Padding for an absent member reads as "nothing here, already fine":
            # zero offset and full reach, so it exerts no pull on the actor.
            obs.extend([0.0, 0.0, 0.0, 1.0])

    # ---- nearest other UAVs (collision avoidance) ----
    # Relative position + a shared policy breaks the collision symmetry: if A sees
    # B east and B sees A west, both steer apart. The whole fleet is always
    # present, so every other UAV is a real obstacle.
    others = [
        (state["uavs"][j][0] - x, state["uavs"][j][1] - y)
        for j in state["uavs"]
        if j != uav_id
    ]
    others.sort(key=lambda d: d[0] * d[0] + d[1] * d[1])
    neigh = []
    for dxn, dyn in others[:_N_NEIGH]:
        neigh.append(float(np.clip(dxn / _NEIGH_REF, -1.0, 1.0)))
        neigh.append(float(np.clip(dyn / _NEIGH_REF, -1.0, 1.0)))
    neigh.extend([1.0] * (2 * _N_NEIGH - len(neigh)))   # pad missing -> far
    obs.extend(neigh)

    return np.array(obs, dtype=np.float32)


def _gamma_norm(gamma: float) -> float:
    """Uplink SNR (linear) -> dB, normalised to ~[-2, 2] for the critic."""
    g_db = 10.0 * np.log10(max(gamma, 1e-10))
    return float(np.clip(g_db / 50.0, -2.0, 2.0))


def global_state(state: dict, info: dict = None) -> np.ndarray:
    """
    Centralised BS state S^t for the critic (CTDE). Privileged, training-only.

    Per UAV i : [x, y, vx, vy, rho_i, gamma_i, load_i, multi-hot set A_i]
    Per target k: [alive, mu_k (pos+vel), Sigma_k (10 unique entries of the 4x4)]

    Every slot in the id space [0, MAX_TARGETS) gets a fixed position with an
    explicit `alive` flag, because the live id set changes every time a target is
    born or rescued. Packing only the live targets would shift every downstream
    feature whenever the backlog changed, which the critic cannot learn through.

    Unlike local_obs this is scenario-sized, which is fine: the critic is
    discarded at execution and used only in training.
    """
    gammas = state.get("gamma", {})
    parts: list = []

    for i in range(NUM_UAVS):
        x, y, vx, vy = state["uavs"][i]
        rho   = state["rho"][i]
        gamma = gammas.get(i, 0.0) if isinstance(gammas, dict) else 0.0
        ks    = state["assignments"].get(i, set()) or set()
        multihot = [0.0] * MAX_TARGETS
        for k in ks:
            if 0 <= k < MAX_TARGETS:
                multihot[k] = 1.0
        parts.extend([
            x / MAP_SIZE, y / MAP_SIZE, vx / V_MAX, vy / V_MAX,
            rho, _gamma_norm(gamma), min(len(ks) / _LOAD_REF, 1.0), *multihot,
        ])

    for k in range(MAX_TARGETS):
        if k not in state["targets"]:
            parts.extend([0.0] * 15)       # 1 alive flag + 4 mean + 10 covariance
            continue
        mu, Sigma = state["targets"][k]
        parts.extend([
            1.0,
            mu[0] / MAP_SIZE, mu[1] / MAP_SIZE,
            mu[2] / V_MAX_TARGET, mu[3] / V_MAX_TARGET,
        ])
        s = Sigma
        ent = [
            s[0, 0] / (_SIGMA_REF ** 2), s[1, 1] / (_SIGMA_REF ** 2), s[0, 1] / (_SIGMA_REF ** 2),
            s[2, 2] / (V_MAX_TARGET ** 2), s[3, 3] / (V_MAX_TARGET ** 2), s[2, 3] / (V_MAX_TARGET ** 2),
            s[0, 2] / (_SIGMA_REF * V_MAX_TARGET), s[0, 3] / (_SIGMA_REF * V_MAX_TARGET),
            s[1, 2] / (_SIGMA_REF * V_MAX_TARGET), s[1, 3] / (_SIGMA_REF * V_MAX_TARGET),
        ]
        parts.extend([float(np.clip(e, -5.0, 5.0)) for e in ent])

    return np.array(parts, dtype=np.float32)
