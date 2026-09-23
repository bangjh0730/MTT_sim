"""Tier 2: judge. Light LLM, woken only when the detector alarms.

Answers WHETHER a re-partition is worth doing and hands the planner a brief. It
never computes an assignment.

A numeric threshold on imbalance cannot do this job: raw imbalance means
opposite things in the two regimes. Under heavy overload a large spread is the
correct equilibrium, since moving a target between two saturated UAVs relocates
the backlog without shrinking it; under light load the same spread is genuinely
fixable. What separates them is the marginal value of intervening, which the
saturating rescue curve encodes for free.

The output is a brief, not a boolean, because it feeds the planner: regime,
locus, a synthesis of what changed, and which sets to leave alone. The line that
keeps the tiers distinct is that the judge says WHERE and WHAT REGIME; the
planner says the exact set membership.
"""

import json
import re
from typing import Optional

from config.params import JUDGE_MODEL
from agentic.llm   import LLMClient


class JudgeAgent:

    _MODEL = JUDGE_MODEL

    _SYSTEM_PROMPT = (
        "You are the Judge Agent in a multi-UAV search-and-rescue system. There are "
        "FEWER UAVs than targets, so full coverage is impossible by construction and is "
        "NOT your concern — never ask for it, never flag its absence. Each UAV holds a SET "
        "of targets and cycles its sensing among them; a target is rescued with "
        "probability lambda/(lambda + tr(Sigma)), so the faster a UAV revisits a member, "
        "the sooner that member is rescued. The mission objective is minimum average "
        "rescue delay.\n"
        "\n"
        "You decide WHETHER re-partitioning the targets across the UAVs is worth doing "
        "right now, and if so you write a short brief for the Planner, which will do the "
        "actual re-partitioning. You NEVER assign targets yourself.\n"
        "\n"
        "The judgement that matters is the marginal value of intervening, and it depends "
        "on the load regime:\n"
        "  * OVERLOADED (every UAV is carrying several targets, rescue rates are low "
        "across the board): a big spread in per-UAV load is the normal equilibrium, not a "
        "fault. Moving a target from the busiest UAV to the next-busiest just relocates "
        "the backlog — both sets stay slow — so the answer is usually NO. Say yes only "
        "when something concrete changed, such as a target that belongs to nobody.\n"
        "  * MANAGEABLE (some UAV has few or no targets while another is stretched): the "
        "same spread is now genuinely fixable, because the lightly-loaded UAV's members "
        "are already being rescued quickly and it has real spare capacity. Say YES.\n"
        "Unassigned targets are always worth acting on — nobody is sensing them, so their "
        "rescue rate is effectively zero and they wait forever.\n"
        "\n"
        "You may also DEFER: if events are still arriving and the situation has not "
        "settled, it is better to wait a few slots and re-partition once over the settled "
        "state than to re-partition now and again immediately after.\n"
        "\n"
        "Reply ONLY with a valid JSON object, no markdown and no text outside it:\n"
        "{\n"
        '  "decision": "replan" | "hold" | "defer",\n'
        '  "wait_slots": <int, only meaningful for "defer">,\n'
        '  "regime": "overloaded" | "manageable",\n'
        '  "direction": "tightening" | "easing" | "steady",\n'
        '  "focus_uavs": [<uav ids the change should be confined to>],\n'
        '  "focus_targets": [<target ids that need a new holder>],\n'
        '  "preserve_uavs": [<uav ids whose sets are working and should not be disturbed>],\n'
        '  "whats_changed": "<one sentence: the NET effect of everything since the last '
        're-partition, e.g. targets 5 and 7 born and target 2 rescued, so UAV 1 is '
        'overloaded while UAV 2 has freed capacity>",\n'
        '  "emphasis": "<one sentence of guidance for the Planner: concentrate to drain '
        'fast, or spread to keep more targets tracked>",\n'
        '  "rationale": "<one sentence: why this is or is not worth a re-partition now>"\n'
        "}\n"
        "Name only UAV and target ids that appear in the state you were given."
    )

    def __init__(self, log_path: str = "llm_log.jsonl"):
        self._llm = LLMClient(log_path=log_path, model=self._MODEL)

    @property
    def latencies(self) -> list:
        return self._llm.latencies

    @property
    def slots(self) -> list:
        return self._llm.slots

    # ------------------------------------------------------------------
    def assess(self, digest: dict, digest_text: str, slot: int = 0) -> Optional[dict]:
        """Decide whether to re-partition; return the brief, or None to hold.

        A returned brief always carries decision == "replan". "hold" and "defer"
        both return None — the difference is only in what the caller schedules
        next, which is carried on the brief's `wait_slots` when it defers.
        """
        raw = self._llm.generate(
            system=self._SYSTEM_PROMPT,
            user=digest_text + "\n\nShould the targets be re-partitioned across the UAVs now?",
            max_tokens=1024,
            agent="JA",
            slot=slot,
        )
        brief = self._parse(raw, digest)
        self.last_brief = brief
        return brief if brief.get("decision") == "replan" else None

    # ------------------------------------------------------------------
    def _parse(self, raw: str, digest: dict) -> dict:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{[\s\S]*\}", raw)
            try:
                data = json.loads(m.group()) if m else {}
            except json.JSONDecodeError:
                data = {}
        if not isinstance(data, dict):
            data = {}

        decision = str(data.get("decision", "hold")).lower().strip()
        if decision not in ("replan", "hold", "defer"):
            decision = "hold"

        live_targets = set(digest.get("unassigned", []))
        for v in digest["per_uav"].values():
            live_targets |= set(v["members"])
        live_uavs = set(digest["per_uav"])

        def _ids(key, universe):
            vals = data.get(key, [])
            if not isinstance(vals, list):
                return []
            out = []
            for v in vals:
                try:
                    v = int(v)
                except (TypeError, ValueError):
                    continue
                if v in universe:
                    out.append(v)
            return sorted(set(out))

        try:
            wait = int(data.get("wait_slots", 0))
        except (TypeError, ValueError):
            wait = 0

        return {
            "decision":      decision,
            "wait_slots":    max(0, min(wait, 100)),
            "regime":        str(data.get("regime", "")) or "unknown",
            "direction":     str(data.get("direction", "")) or "steady",
            "focus_uavs":    _ids("focus_uavs", live_uavs),
            "focus_targets": _ids("focus_targets", live_targets),
            "preserve_uavs": _ids("preserve_uavs", live_uavs),
            "whats_changed": str(data.get("whats_changed", "")),
            "emphasis":      str(data.get("emphasis", "")),
            "rationale":     str(data.get("rationale", "")),
        }

    # ------------------------------------------------------------------
    @staticmethod
    def format_brief(brief: dict) -> str:
        """Render the brief as the leading section of the planner's prompt."""
        lines = [
            "== BRIEF FROM THE JUDGE ==",
            f"Load regime: {brief['regime']} ({brief['direction']}).",
            f"What changed since the last re-partition: {brief['whats_changed']}",
            f"Why now: {brief['rationale']}",
            f"Emphasis for you: {brief['emphasis']}",
        ]
        if brief["focus_targets"]:
            lines.append(f"Targets that need a holder: {brief['focus_targets']}")
        if brief["focus_uavs"]:
            lines.append(
                f"Confine the change to UAVs {brief['focus_uavs']} where you can — "
                "the rest of the partition is working and reshuffling it costs travel "
                "time for no gain."
            )
        if brief["preserve_uavs"]:
            lines.append(
                f"Leave UAVs {brief['preserve_uavs']} as they are unless you have a "
                "concrete reason not to."
            )
        return "\n".join(lines)
