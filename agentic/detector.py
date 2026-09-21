"""
Event Detector — tier 1 of the assignment stack. DETERMINISTIC, runs every slot.

This replaces the old LLM Monitor Agent entirely. The Monitor asked an LLM, every
20 slots, "is every target covered by an active UAV?" — a question with no meaning
in the overloaded regime, where 3 UAVs cannot cover 8+ targets and the answer is
trivially "no" forever. Worse, it put a language model on the per-slot path to
answer something that is pure bookkeeping, which is why it needed hallucinated-id
guards bolted on.

What actually invalidates a partition is small, discrete, and observable:

  * BIRTH  — a new target exists and belongs in somebody's set.
  * RESCUE — a target left, so the UAV that held it has spare capacity that an
             overloaded peer could use.

Those are the only two events that change the load. The fleet is fixed and never
loses a UAV, so arrivals and departures of TARGETS are the whole story. Everything else the old design
wanted to trigger on is handled elsewhere and deliberately does NOT wake the BS:

  * A target drifting out of a UAV's instantaneous sensing radius is a
    FAST-TIMESCALE event. The UAV shifts its position and picks it up again on a
    later slot; the MARL policy owns that, and waking the BS for it would replan
    over a condition that fixes itself in a few slots.
  * Load and SPREAD imbalance is real but SLOW and gradual. A set the planner made
    compact drifts apart as its targets wander, until no single position serves it
    and the UAV starts commuting. That degrades continuously rather than crossing
    a line, so it gets the periodic tick (DETECTOR_TICK) rather than an alarm —
    a backstop, not the workhorse. The digest carries per-UAV spread so the judge
    can see it when the tick does fire.

The detector makes no decisions. It raises alarms and maintains a running digest —
what has fired since the last replan, and how the load sits across the fleet — so
the judge can reason over ACCUMULATED events rather than reacting to each in
isolation. That accumulation is what lets a burst of four events inside ten slots
produce one considered replan instead of four thrashing ones.
"""

import numpy as np

from config.params import DETECTOR_TICK, LAMBDA_RESCUE


def _spread(state: dict, members: list) -> float:
    """RMS distance of a UAV's set members from their centroid, in metres."""
    if len(members) < 2:
        return 0.0
    pos = np.array([state["targets"][k][0][:2] for k in members])
    return float(np.sqrt(((pos - pos.mean(axis=0)) ** 2).sum(axis=1).mean()))


class EventDetector:
    """Deterministic alarm source and digest builder. No LLM, no state guessing."""

    def __init__(self, tick: int = DETECTOR_TICK):
        self.tick = int(tick)
        self._last_replan_slot = 0
        self._last_alarm_slot  = 0
        # Events accumulated since the last replan. Cleared by note_replan(), NOT
        # by raising an alarm: if the judge declines to replan, the events it
        # declined on are still pending and must still be visible next time.
        self.pending_births:  list = []   # (slot, target_id)
        self.pending_rescues: list = []   # (slot, target_id, delay_slots)
        self.n_alarms = 0

    # ------------------------------------------------------------------
    def note_replan(self, slot: int) -> None:
        """Called when the planner has actually installed a new partition."""
        self._last_replan_slot = int(slot)
        self.pending_births.clear()
        self.pending_rescues.clear()

    @property
    def slots_since_replan_at(self):
        return self._last_replan_slot

    # ------------------------------------------------------------------
    def observe(self, slot: int, born: list, info: dict) -> bool:
        """Record this slot's events and report whether to raise an alarm.

        Returns True when the judge should be woken — on any discrete event, or on
        the periodic tick if nothing has fired for a while.
        """
        fired = False

        for k in born or []:
            self.pending_births.append((slot, int(k)))
            fired = True

        rescued = info.get("rescued", []) if info else []
        delays  = info.get("rescue_delays", []) if info else []
        for k, d in zip(rescued, delays):
            self.pending_rescues.append((slot, int(k), int(d)))
            fired = True

        # Periodic backstop for the slow imbalance that no event announces.
        if not fired and slot - max(self._last_replan_slot, self._last_alarm_slot) >= self.tick:
            fired = True

        if fired:
            self._last_alarm_slot = slot
            self.n_alarms += 1
        return fired

    # ------------------------------------------------------------------
    def digest(self, slot: int, state: dict, info: dict) -> dict:
        """The compact situation summary the judge reads.

        Deliberately COARSER than the planner's view: per-UAV load and uncertainty
        aggregates plus event attribution, but no per-target geometry table. That
        asymmetry is the whole justification for having two tiers — a judge that
        needed full state to decide would just be a serial LLM call in front of
        every replan, buying nothing. It is richer than a single global spread
        number, though, because the judge has to be able to LOCALISE the problem
        ("UAV 2 was freed, UAV 0 is carrying the backlog") for its brief to be
        worth anything to the planner.
        """
        targets = state.get("targets", {})
        asgn    = state.get("assignments", {})

        tr = {k: float(np.trace(S[:2, :2])) for k, (mu, S) in targets.items()}
        pr = {k: LAMBDA_RESCUE / (LAMBDA_RESCUE + max(v, 0.0)) for k, v in tr.items()}

        per_uav = {}
        for i in sorted(state.get("uavs", {})):
            members = sorted(k for k in asgn.get(i, ()) if k in targets)
            per_uav[i] = {
                "load":       len(members),
                "members":    members,
                # Mean rescue rate across the set: the honest per-UAV measure of
                # how fast this UAV is clearing what it holds. A UAV with a big
                # set AND a low mean rate is genuinely stretched; a big set with a
                # high mean rate is coping and should be left alone.
                "mean_pr":    float(np.mean([pr[k] for k in members])) if members else 0.0,
                "worst_pr":   float(min([pr[k] for k in members])) if members else 0.0,
                "max_stale":  int(max([state.get("time_since_sensed", {}).get(k, 0)
                                       for k in members], default=0)),
                # RMS distance of the members from their own centroid. A UAV
                # serves its set from one position, so spread — not set size —
                # is what says whether that is even possible: a tight set is
                # refreshed every few slots, a spread one forces a commute during
                # which every member decays. This is the geometric half of "is
                # this partition still good", and it is cheap enough to put in
                # the digest without handing the judge the full target table.
                "spread_m":   _spread(state, members),
            }

        assigned   = {k for i in per_uav for k in per_uav[i]["members"]}
        unassigned = sorted(set(targets) - assigned)
        loads      = [v["load"] for v in per_uav.values()]

        return {
            "slot":               slot,
            "backlog":            len(targets),
            "n_uavs":             len(per_uav),
            "unassigned":         unassigned,
            "load_spread":        int(max(loads) - min(loads)) if loads else 0,
            "fleet_mean_pr":      float(np.mean(list(pr.values()))) if pr else 0.0,
            "per_uav":            per_uav,
            "slots_since_replan": slot - self._last_replan_slot,
            "births_since":       list(self.pending_births),
            "rescues_since":      list(self.pending_rescues),
        }

    # ------------------------------------------------------------------
    @staticmethod
    def format_digest(d: dict) -> str:
        """Render the digest as the judge's prompt body."""
        lines = [
            f"Slot {d['slot']}.  Backlog |K| = {d['backlog']} targets, "
            f"|U| = {d['n_uavs']} UAVs.",
            f"Slots since last re-partition: {d['slots_since_replan']}.",
            "",
            "Per-UAV load. mean_rescue_rate is the mean per-slot rescue probability "
            "across that UAV's set (higher means it is clearing its set faster). "
            "set_spread is how far apart that UAV's targets are: a UAV serves its set "
            "from one position and sensing reaches about 700 m, so a spread much beyond "
            "that means it cannot cover its own set and is losing time commuting:",
        ]
        for i, v in sorted(d["per_uav"].items()):
            lines.append(
                f"  UAV {i}: holds {v['load']} target(s) {v['members']}  "
                f"mean_rescue_rate={v['mean_pr']:.3f}  "
                f"worst_member_rate={v['worst_pr']:.3f}  "
                f"longest_unsensed={v['max_stale']} slots  "
                f"set_spread={v['spread_m']:.0f} m"
            )
        if d["unassigned"]:
            lines.append(f"  UNASSIGNED targets (held by nobody): {d['unassigned']}")
        lines += [
            "",
            f"Load spread across UAVs: {d['load_spread']} targets between the "
            f"heaviest and lightest.",
            f"Fleet mean rescue rate: {d['fleet_mean_pr']:.3f}.",
            "",
            "Events since the last re-partition:",
        ]
        if d["births_since"]:
            lines.append("  births:  " + ", ".join(
                f"target {k} at slot {s}" for s, k in d["births_since"]))
        if d["rescues_since"]:
            lines.append("  rescues: " + ", ".join(
                f"target {k} at slot {s} (waited {dl} slots)"
                for s, k, dl in d["rescues_since"]))
        if not (d["births_since"] or d["rescues_since"]):
            lines.append("  (none — this is a periodic check, not an event)")
        return "\n".join(lines)
