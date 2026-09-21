"""
Judge Agent (JA) — tier 2. Light LLM, woken only when the detector alarms.

The judge answers WHETHER a re-partition is worth doing now, and hands the planner
a BRIEF characterising the situation. It never computes an assignment.

Why this tier exists at all
---------------------------
The obvious alternative is a numeric trigger: fire the planner when some imbalance
measure exceeds a threshold θ. That fails, and not for a tuning reason. Raw
imbalance means OPPOSITE things in the two load regimes. Under heavy overload
every UAV is saturated, so a large spread in per-UAV backlog is the correct
equilibrium — moving a target from the worst UAV to the next-worst relocates the
backlog without shrinking it, and firing is wasted work. Under light load the same
spread means one UAV has slack while another is stretched, which is genuinely
fixable. No single θ separates those, and hand-coding θ(load) just moves the
arbitrariness into a second knob.

What actually distinguishes them is the MARGINAL VALUE of re-planning, and the
rescue curve encodes it for free: because p_r = λ/(λ + tr Σ) saturates, its
marginal return is steep at low tr(Σ) and flat at high tr(Σ). A UAV cycling a set
of size m refreshes each member every ~m slots, so steady-state tr(Σ) grows with
m, and shifting one target between two large sets barely moves total rescue rate
(both sides sit on the flat tail) while the same shift between small sets moves it
a lot. Regime-adaptivity is therefore a property of the system, not a threshold to
tune — and judging it is a qualitative reading of load, uncertainty and what has
changed, which is what an LLM is actually good at.

Why the output is not a boolean
-------------------------------
The judge's answer feeds the planner, so a bare yes/no throws away the most
valuable thing the judge produced. It emits a brief with four parts:

  regime   — overloaded vs manageable, and which way it is moving. Sets the
             planner's emphasis: concentrate to drain fast under heavy load,
             spread to keep more targets tracked as it eases.
  focus    — which UAVs and targets are implicated. Under overload most of the
             partition is fine; naming the locus lets the planner do a LOCAL
             repartition instead of re-solving the whole map, which means a
             smaller problem and far less gratuitous reshuffling.
  changed  — the synthesised net effect of everything accumulated since the last
             replan, not a raw event log. This synthesis across a burst of alarms
             is real work and is the reason the tier is a model and not a rule.
  preserve — the stable sets the planner should leave alone. Anti-oscillation
             expressed as intent rather than hoped for.

The line that keeps the two tiers non-redundant: the judge says WHERE and WHAT
REGIME; the planner says the exact set membership. If the brief ever became
prescriptive enough to be the answer, the tiers should be collapsed — the standing
test is that the judge never needs the full per-target geometry table.

It may also answer "not yet": with the tiers decoupled in time, deferring lets a
burst settle so one considered replan lands over the settled state instead of four
reactive ones.
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
