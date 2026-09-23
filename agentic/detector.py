"""Tier 1: deterministic event detector, every slot.

A partition goes stale for two observable reasons -- a BIRTH (a new target needs
a holder) and a RESCUE (its holder now has spare capacity). Both are bookkeeping,
so no LLM is involved. A target drifting out of sensing range is fast-timescale
and the MARL policy handles it; load/spread imbalance is gradual and gets the
periodic tick rather than an alarm.

The detector decides nothing. It raises alarms and keeps a running digest so the
judge can reason over ACCUMULATED events instead of reacting to each in
isolation, which is what lets a burst produce one considered replan.
"""

import numpy as np

from config.params import DETECTOR_TICK, LAMBDA_RESCUE


def _spread(state: dict, members: list) -> float:
    """RMS distance of a set's members from their centroid, metres."""
    if len(members) < 2:
        return 0.0
    pos = np.array([state["targets"][k][0][:2] for k in members])
    return float(np.sqrt(((pos - pos.mean(axis=0)) ** 2).sum(axis=1).mean()))


class EventDetector:

    def __init__(self, tick: int = DETECTOR_TICK):
        self.tick = int(tick)
        self._last_replan_slot = 0
        self._last_alarm_slot  = 0
        # Cleared by note_replan, not by raising an alarm: events the judge
        # declined to act on are still pending next time.
        self.pending_births:  list = []
        self.pending_rescues: list = []
        self.n_alarms = 0

    def note_replan(self, slot: int) -> None:
        self._last_replan_slot = int(slot)
        self.pending_births.clear()
        self.pending_rescues.clear()

    def observe(self, slot: int, born: list, info: dict) -> bool:
        """Record this slot's events; True if the judge should be woken."""
        fired = False

        for k in born or []:
            self.pending_births.append((slot, int(k)))
            fired = True

        rescued = info.get("rescued", []) if info else []
        delays  = info.get("rescue_delays", []) if info else []
        for k, d in zip(rescued, delays):
            self.pending_rescues.append((slot, int(k), int(d)))
            fired = True

        if not fired and slot - max(self._last_replan_slot, self._last_alarm_slot) >= self.tick:
            fired = True

        if fired:
            self._last_alarm_slot = slot
            self.n_alarms += 1
        return fired

    def digest(self, slot: int, state: dict, info: dict) -> dict:
        """Per-UAV load and event attribution for the judge.

        Deliberately coarser than the planner's view: no per-target geometry
        table. That asymmetry is what justifies two tiers -- a judge needing full
        state would just be a serial LLM call in front of every replan.
        """
        targets = state.get("targets", {})
        asgn    = state.get("assignments", {})

        tr = {k: float(np.trace(S[:2, :2])) for k, (mu, S) in targets.items()}
        pr = {k: LAMBDA_RESCUE / (LAMBDA_RESCUE + max(v, 0.0)) for k, v in tr.items()}

        per_uav = {}
        for i in sorted(state.get("uavs", {})):
            members = sorted(k for k in asgn.get(i, ()) if k in targets)
            per_uav[i] = {
                "load":      len(members),
                "members":   members,
                "mean_pr":   float(np.mean([pr[k] for k in members])) if members else 0.0,
                "worst_pr":  float(min([pr[k] for k in members])) if members else 0.0,
                "max_stale": int(max([state.get("time_since_sensed", {}).get(k, 0)
                                      for k in members], default=0)),
                "spread_m":  _spread(state, members),
            }

        assigned = {k for i in per_uav for k in per_uav[i]["members"]}
        loads    = [v["load"] for v in per_uav.values()]

        return {
            "slot":               slot,
            "backlog":            len(targets),
            "n_uavs":             len(per_uav),
            "unassigned":         sorted(set(targets) - assigned),
            "load_spread":        int(max(loads) - min(loads)) if loads else 0,
            "fleet_mean_pr":      float(np.mean(list(pr.values()))) if pr else 0.0,
            "per_uav":            per_uav,
            "slots_since_replan": slot - self._last_replan_slot,
            "births_since":       list(self.pending_births),
            "rescues_since":      list(self.pending_rescues),
        }

    @staticmethod
    def format_digest(d: dict) -> str:
        lines = [
            f"Slot {d['slot']}.  Backlog |K| = {d['backlog']} targets, "
            f"|U| = {d['n_uavs']} UAVs.",
            f"Slots since last re-partition: {d['slots_since_replan']}.",
            "",
            "Per-UAV load. mean_rescue_rate is the mean per-slot rescue probability "
            "across that UAV's set. set_spread is how far apart its targets are: a "
            "UAV serves its set from one position and sensing reaches about 490 m, "
            "so a spread much beyond that means it is losing time commuting:",
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
            lines.append("  (none - this is a periodic check, not an event)")
        return "\n".join(lines)
