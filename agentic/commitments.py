"""Conditional pre-commitments: an assignment staged now, applied on an event.

This is the capability the baselines lack. They resolve each decision from the
current state; a commitment is written in advance and fires when an anticipated
event occurs, so a UAV can be moving toward its next set before the trigger.

Lives here rather than in the environment on purpose. MTTEnv knows nothing about
commitments, so the baselines run on the same simulator without inheriting the
mechanism, and `then` is applied through the ordinary apply_assignments path.

Usage, once per slot, after stepping the environment:

    fired = store.step(env.t, rescued=info["rescued"], born=born_this_slot)
    if fired is not None:
        env.apply_assignments(fired)
"""
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional, Tuple


@dataclass
class Commitment:
    """Apply `then` once every condition in `when` has been met.

    when_rescued : every one of these targets must have been rescued. The set is
                   what makes a trigger compound - "7 and 11 are both done" is a
                   different condition from either alone.
    after_birth  : a target must have been born since the commitment was added.
    not_before   : absolute slot before which it may not fire.
    then         : {uav_id: target ids} to install as the partition.
    expires      : absolute slot after which it is discarded unfired. Not
                   optional: a stale commitment fires against a state that has
                   moved on, which is the obvious failure of the idea.
    note         : free text from whoever wrote it, carried for the log only.
    """
    then:         Dict[int, FrozenSet[int]]
    expires:      int
    when_rescued: FrozenSet[int] = frozenset()
    after_birth:  bool = False
    not_before:   int = 0
    note:         str = ""

    # progress, filled in by the store
    _seen_rescued: set = field(default_factory=set, repr=False)
    _seen_birth:   bool = field(default=False, repr=False)
    _added_slot:   int = field(default=0, repr=False)

    def satisfied(self, slot: int) -> bool:
        return (slot >= self.not_before
                and self.when_rescued <= self._seen_rescued
                and (self._seen_birth or not self.after_birth))


def early_assignment(c: "Commitment", current: Dict[int, Iterable[int]]
                     ) -> Dict[int, set]:
    """The partition to install at STAGING time, before the trigger fires.

    `then` hands the receiver its future targets and takes them off the donor.
    Applying that early would strip the receiver of the target it is still
    finishing, so this keeps any currently-held target that `then` does not
    mention - those are the ones expected to be rescued - and moves everything
    else immediately:

        current  uav0 {A}      uav1 {B, C, D}
        then     uav0 {B}      uav1 {C, D}        (after A is rescued)
        early    uav0 {A, B}   uav1 {C, D}

    So the receiver holds {finishing, future} through the lead window and can
    position for B while still sensing A, and the donor drops B at once rather
    than carrying a target it will not serve. Still a partition: no target is
    held twice.

    This is the step that turns a commitment into early MOVEMENT rather than an
    early decision, and it is only expressible because sets are one-to-many.
    """
    cur = {int(i): set(int(k) for k in ks) for i, ks in current.items()}
    promised = {k for ks in c.then.values() for k in ks}
    out: Dict[int, set] = {}
    for i in sorted(set(cur) | set(c.then)):
        finishing = {k for k in cur.get(i, set()) if k not in promised}
        out[i] = set(c.then.get(i, set())) | finishing
    return out


class CommitmentStore:
    """Holds staged commitments and fires at most one per slot.

    Earliest-added wins, and firing one clears the rest: they were written
    against a different predicted future, so applying a second on top would be
    acting on a plan that is already void. That is a simplification worth stating
    rather than discovering - a planner that wants a chain must re-stage it.
    """

    def __init__(self, num_uavs: int):
        self.num_uavs = num_uavs
        self._items: List[Commitment] = []
        self.n_added = 0
        self.n_fired = 0
        self.n_expired = 0
        self.n_voided = 0      # dropped because another commitment fired
        self.n_rejected = 0
        self.fired_log: List[Tuple[int, Commitment]] = []

    # ------------------------------------------------------------------ writing
    def add(self, c: Commitment, slot: int = 0) -> bool:
        """Stage a commitment. Returns False if it is malformed or already dead."""
        if c.expires <= slot:
            self.n_rejected += 1
            return False
        then = {}
        for i, ks in c.then.items():
            try:
                i = int(i)
            except (TypeError, ValueError):
                self.n_rejected += 1
                return False
            if not 0 <= i < self.num_uavs:
                self.n_rejected += 1
                return False
            then[i] = frozenset(int(k) for k in ks)
        # One target cannot be held by two UAVs; keep the lowest id that claims it.
        claimed: set = set()
        for i in sorted(then):
            then[i] = frozenset(k for k in sorted(then[i])
                                if k not in claimed and not claimed.add(k))
        c.then = then
        c.when_rescued = frozenset(int(k) for k in c.when_rescued)
        c._added_slot = slot
        c._seen_rescued = set()
        c._seen_birth = False
        self._items.append(c)
        self.n_added += 1
        return True

    def clear(self) -> None:
        self.n_voided += len(self._items)
        self._items.clear()

    # ------------------------------------------------------------------ reading
    @property
    def pending(self) -> List[Commitment]:
        return list(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def staged_members(self) -> Dict[int, set]:
        """{uav_id: targets promised to it by a PENDING commitment}.

        For the allocator's own bookkeeping and for logging - the UAV policy is
        not told which of its members are promised rather than current, and does
        not need to be. Drifting toward a distant promise costs p_r on the target
        in hand for a couple of slots against a transit of a hundred, so serving
        the nearest member is already right. What matters is which UAV holds the
        promise, which is decided here.
        """
        out: Dict[int, set] = {}
        for c in self._items:
            for i, ks in c.then.items():
                out.setdefault(i, set()).update(ks)
        return out

    # ------------------------------------------------------------------ driving
    def step(self, slot: int, rescued: Iterable[int] = (),
             born: Iterable[int] = (), live: Optional[Iterable[int]] = None
             ) -> Optional[Dict[int, set]]:
        """Record this slot's events, then fire at most one commitment.

        rescued / born : target ids from this slot.
        live           : target ids still alive, used to drop members of `then`
                         that have since been rescued. Pass env.targets.

        Returns the assignment to install, or None.
        """
        rescued = [int(k) for k in rescued]
        any_birth = bool(list(born))
        for c in self._items:
            c._seen_rescued.update(rescued)
            if any_birth:
                c._seen_birth = True

        before = len(self._items)
        self._items = [c for c in self._items if c.expires > slot]
        self.n_expired += before - len(self._items)

        for c in self._items:
            if not c.satisfied(slot):
                continue
            keep = set(live) if live is not None else None
            out = {i: set(ks if keep is None else (ks & keep))
                   for i, ks in c.then.items()}
            for i in range(self.num_uavs):
                out.setdefault(i, set())
            self.n_fired += 1
            self.fired_log.append((slot, c))
            self._items.remove(c)
            self.clear()          # the rest were written against a dead future
            return out
        return None

    # ------------------------------------------------------------------ summary
    def summary(self) -> dict:
        return {
            "added":    self.n_added,
            "fired":    self.n_fired,
            "expired":  self.n_expired,
            "voided":   self.n_voided,
            "rejected": self.n_rejected,
            "pending":  len(self._items),
        }
