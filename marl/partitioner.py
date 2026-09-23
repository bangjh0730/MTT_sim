"""Training-time target partitioner: the stand-in for the agentic allocator.

No LLM runs during training. The surrogate is geometric, because with uniform
target spawns the travel-minimising partition is spatial, which is also what the
planner mostly converges on. It randomises style, reachability, size skew and
untidiness around that geometry so the actor does not co-adapt to one rule --
otherwise it would be quietly tuned to the greedy baseline used as the
"w/o agentic reasoning" ablation.

Assignment is responsibility over time, not simultaneous coverage: a UAV senses
one member per slot, so most of its set is outside the footprint at any instant.
Sets containing far members are therefore correct and necessary to train on.
"""

import itertools
import numpy as np

from config.params import V_MAX, DT
from marl.preprocess import R_DETECT_2

# One-way flight budget in slots for a UAV holding a SINGLE target, drawn per
# episode. The affordable radius divides this by the holder's load (see
# _affordable_radius): a UAV senses one member per slot, so every member it
# adds dilutes the others, and a loaded UAV cannot afford a detour that a light
# one can.
_FLIGHT_BUDGET_SLOTS = (20, 96)

_R_DETECT = float(np.sqrt(R_DETECT_2))


def _travel(uav, p) -> float:
    """Flight needed to bring p into the detection footprint: the distance
    beyond the footprint radius, not the distance to p itself. With a ~660 m
    footprint most of the map is sensable from a central position, so charging
    the full distance would make the reach budget refuse targets that need
    little or no flight."""
    return max(0.0, float(np.linalg.norm(uav.pos2d - p)) - _R_DETECT)

_HOLD_SLOTS = (8, 120)      # partition lifetime between timer-driven re-solves

# What keeping an already-held target is worth, in metres of extra travel, when
# matching clusters to UAVs. Stops a re-solve from shuffling sets wholesale.
_STICKINESS_M = 350.0

_STEP_M = V_MAX * DT


def _best_matching(cost: np.ndarray) -> list:
    """Min-cost assignment. Exact by permutation for |U| <= 8 (scipy is not a
    dependency); greedy above that."""
    n = cost.shape[0]
    if n <= 8:
        best, best_c = None, np.inf
        for perm in itertools.permutations(range(n)):
            c = sum(cost[i, perm[i]] for i in range(n))
            if c < best_c:
                best, best_c = perm, c
        return list(best)

    order = np.dstack(np.unravel_index(np.argsort(cost, axis=None), cost.shape))[0]
    out, used_r, used_c = [-1] * n, set(), set()
    for r, c in order:
        if r not in used_r and c not in used_c:
            out[r] = int(c); used_r.add(r); used_c.add(c)
    return out


class TrainingPartitioner:
    """Forms and maintains assignment sets during training."""

    def __init__(self, rng=None):
        self.rng = rng if rng is not None else np.random.default_rng()
        self._next_resolve = 0
        self.n_resolves = 0
        self.n_event_updates = 0

    def reset(self, env) -> None:
        """Draw this episode's difficulty, then partition once."""
        r = self.rng
        self.mode = "voronoi" if r.random() < 0.5 else "kmeans"
        self.reach_m = float(r.uniform(*_FLIGHT_BUDGET_SLOTS) * _STEP_M)
        self.skew = r.choice(["natural", "balanced", "concentrated"],
                             p=[0.40, 0.40, 0.20])
        self.noise_frac = float(r.uniform(0.10, 0.40)) if r.random() < 0.35 else 0.0
        self._next_resolve = int(r.integers(*_HOLD_SLOTS))
        self.resolve(env)

    def update(self, env, born=None, rescued=None) -> bool:
        """Advance one slot; True if the partition changed.

        Triggers are birth, rescue and the hold timer. Between them the
        partition is held while everything drifts, and serving that staleness is
        most of what the actor learns.
        """
        if env.t >= self._next_resolve:
            self.resolve(env)
            self._next_resolve = env.t + int(self.rng.integers(*_HOLD_SLOTS))
            return True

        changed = False
        if born:
            self._absorb_births(env, born)
            changed = True
        if rescued:
            self._rebalance_after_rescue(env)
            changed = True
        if changed:
            self.n_event_updates += 1
        return changed

    # ------------------------------------------------------------------
    def resolve(self, env) -> None:
        live = sorted(env.targets)
        if not live:
            return
        uav_ids = sorted(env.uavs)
        pos = {k: env.ekf_state[k][0][:2] for k in live}

        sets = (self._voronoi(env, live, pos, uav_ids) if self.mode == "voronoi"
                else self._kmeans(env, live, pos, uav_ids))

        # k-means matches whole clusters, so a member can land past the cap even
        # when a nearer UAV exists; repair either side of the modifiers.
        self._enforce_reach(env, sets, pos, uav_ids)
        self._apply_skew(env, sets, pos, uav_ids)
        self._inject_noise(sets, pos, env, uav_ids)
        self._enforce_reach(env, sets, pos, uav_ids)
        self._fill_idle(env, sets, pos, uav_ids)

        env.apply_assignments(sets)
        self.n_resolves += 1

    def _voronoi(self, env, live, pos, uav_ids) -> dict:
        sets = {i: set() for i in uav_ids}
        for k in live:
            i = min(uav_ids, key=lambda i: float(np.linalg.norm(env.uavs[i].pos2d - pos[k])))
            sets[i].add(k)
        return sets

    def _kmeans(self, env, live, pos, uav_ids) -> dict:
        """Cluster targets, then match clusters to UAVs on travel minus a
        stickiness credit for members already held."""
        U = len(uav_ids)
        pts = np.array([pos[k] for k in live])
        if len(live) <= U:
            return self._voronoi(env, live, pos, uav_ids)

        cen = np.array([env.uavs[i].pos2d for i in uav_ids], dtype=float)
        lab = np.zeros(len(pts), dtype=int)
        for _ in range(12):
            d = ((pts[:, None, :] - cen[None, :, :]) ** 2).sum(-1)
            new = d.argmin(1)
            if np.array_equal(new, lab):
                break
            lab = new
            for c in range(U):
                if (lab == c).any():
                    cen[c] = pts[lab == c].mean(0)

        cost = np.zeros((U, U))
        for a, i in enumerate(uav_ids):
            held = env.uavs[i].assignment_set
            for c in range(U):
                members = {live[j] for j in range(len(live)) if lab[j] == c}
                cost[a, c] = (float(np.linalg.norm(env.uavs[i].pos2d - cen[c]))
                              - _STICKINESS_M * len(held & members))

        match = _best_matching(cost)
        return {i: {live[j] for j in range(len(live)) if lab[j] == match[a]}
                for a, i in enumerate(uav_ids)}

    # ------------------------------------------------------------------
    def _affordable_radius(self, load: int) -> float:
        """How far this UAV can afford to fly (beyond its footprint) given
        what it already holds.

        Adding a member at distance d lengthens the tour by ~2d, paid by every
        other member, so the affordable radius scales inversely with load. This
        is what makes the partition self-correcting: light UAVs clear members
        fast, which widens their reach, which lets them take targets that were
        affordable to nobody at the last re-solve.
        """
        return self.reach_m / max(1, load)

    def _enforce_reach(self, env, sets, pos, uav_ids) -> None:
        """Hand any member past its holder's cap to a UAV that can afford it.

        Never orphans a target: if nobody can afford it, the nearest keeps it,
        since a real mission still has to go and get it. What this forbids is a
        target held by a distant UAV while a nearer one was free.
        """
        # Bounded passes -- a move changes both loads and both radii, but
        # iterating to a fixed point can cycle a target between two UAVs.
        for _ in range(3):
            moved = False
            for i in uav_ids:
                for k in list(sets[i]):
                    if _travel(env.uavs[i], pos[k]) <= self._affordable_radius(len(sets[i])):
                        continue
                    cand = [j for j in uav_ids if j != i
                            and _travel(env.uavs[j], pos[k])
                            <= self._affordable_radius(len(sets[j]) + 1)]
                    if cand:
                        j = min(cand, key=lambda j: (
                            float(np.linalg.norm(env.uavs[j].pos2d - pos[k])),
                            len(sets[j])))
                    else:
                        j = min(uav_ids, key=lambda j: float(
                            np.linalg.norm(env.uavs[j].pos2d - pos[k])))
                        if j == i:
                            continue
                    if j != i:
                        sets[i].discard(k); sets[j].add(k); moved = True
            if not moved:
                break

    def _fill_idle(self, env, sets, pos, uav_ids) -> None:
        """Give every UAV work while targets wait. An empty set is a zero
        observation and a zero reward, so it teaches nothing -- and no sensible
        allocation idles a UAV with a backlog outstanding. Skipped in the
        concentrated regime, where an idle UAV is the point of the draw."""
        if self.skew == "concentrated":
            return
        n_live = sum(len(v) for v in sets.values())
        for _ in range(len(uav_ids)):
            idle = [i for i in uav_ids if not sets[i]]
            if not idle or n_live < len(uav_ids):
                return
            i = idle[0]
            donors = [j for j in uav_ids if len(sets[j]) >= 2]
            if not donors:
                return
            cand = [(j, k) for j in donors for k in sets[j]]
            j, k = min(cand, key=lambda jk: (
                float(np.linalg.norm(env.uavs[i].pos2d - pos[jk[1]]))
                - _STICKINESS_M * (len(sets[jk[0]]) - 1)))
            sets[j].discard(k); sets[i].add(k)

    def _apply_skew(self, env, sets, pos, uav_ids) -> None:
        """Push set sizes toward the drawn regime, honouring reachability."""
        if self.skew == "natural":
            return

        def reachable(i, k):
            return _travel(env.uavs[i], pos[k]) <= self._affordable_radius(len(sets[i]) + 1)

        n_live = sum(len(s) for s in sets.values())
        if self.skew == "balanced":
            cap = int(np.ceil(n_live / max(len(uav_ids), 1)))
            for _ in range(n_live):
                over = [i for i in uav_ids if len(sets[i]) > cap]
                under = [i for i in uav_ids if len(sets[i]) < cap]
                if not over or not under:
                    break
                src = max(over, key=lambda i: len(sets[i]))
                k = max(sets[src], key=lambda k: float(
                    np.linalg.norm(env.uavs[src].pos2d - pos[k])))
                cand = [i for i in under if reachable(i, k)]
                if not cand:
                    break
                dst = min(cand, key=lambda i: float(
                    np.linalg.norm(env.uavs[i].pos2d - pos[k])))
                sets[src].discard(k); sets[dst].add(k)
        else:
            heavy = uav_ids[int(self.rng.integers(0, len(uav_ids)))]
            for i in uav_ids:
                if i == heavy:
                    continue
                for k in list(sets[i]):
                    if len(sets[i]) <= 1:
                        break
                    if reachable(heavy, k) and self.rng.random() < 0.6:
                        sets[i].discard(k); sets[heavy].add(k)

    def _inject_noise(self, sets, pos, env, uav_ids) -> None:
        """Rehome a fraction of targets at random, within the reach cap."""
        if self.noise_frac <= 0.0:
            return
        allk = [k for s in sets.values() for k in s]
        n = int(round(self.noise_frac * len(allk)))
        if n <= 0:
            return
        for k in self.rng.permutation(allk)[:n]:
            k = int(k)
            cand = [i for i in uav_ids
                    if _travel(env.uavs[i], pos[k]) <= self._affordable_radius(len(sets[i]) + 1)]
            if not cand:
                continue
            dst = int(cand[self.rng.integers(0, len(cand))])
            for s in sets.values():
                s.discard(k)
            sets[dst].add(k)

    # ------------------------------------------------------------------
    def _absorb_births(self, env, born) -> None:
        table = env.assignment_table()
        for k in born:
            if k not in env.targets:
                continue
            p = env.ekf_state[k][0][:2]
            cand = [i for i in table
                    if _travel(env.uavs[i], p) <= self._affordable_radius(len(table[i]) + 1)]
            pool = cand or list(table)
            i = min(pool, key=lambda i: (float(np.linalg.norm(env.uavs[i].pos2d - p)),
                                         len(table[i])))
            table[i].add(k)
        env.apply_assignments(table)

    def _rebalance_after_rescue(self, env) -> None:
        """A rescue frees capacity, so the freed UAV absorbs load. Its lower
        load widens its affordable radius, so targets nobody could reach at the
        last re-solve may be reachable now."""
        table = env.assignment_table()
        if len(table) < 2:
            return
        for _ in range(3):
            light = min(table, key=lambda i: (len(table[i]), i))
            heavy = max(table, key=lambda i: (len(table[i]), i))
            if light == heavy or len(table[heavy]) - len(table[light]) < 2:
                break
            budget = self._affordable_radius(len(table[light]) + 1)
            cand = [k for k in table[heavy] if k in env.targets and
                    _travel(env.uavs[light], env.ekf_state[k][0][:2]) <= budget]
            if not cand:
                break
            k = min(cand, key=lambda k: float(
                np.linalg.norm(env.uavs[light].pos2d - env.ekf_state[k][0][:2])))
            table[heavy].discard(k); table[light].add(k)

        pos = {k: env.ekf_state[k][0][:2] for k in env.targets}
        self._fill_idle(env, table, pos, sorted(env.uavs))
        env.apply_assignments(table)
