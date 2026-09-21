"""
Planner Agent (PA) — tier 3. Heavier LLM, invoked only when the judge approves.

Reads the full system state plus the judge's brief and emits a PARTITION of the
live targets into per-UAV SETS.

What changed from the coverage-era planner, and why none of it could be kept
----------------------------------------------------------------------------
The old planner's entire objective was unsatisfiable here. It carried a hard
constraint — "every live target must have a UAV assigned" — and minimised the time
until ALL targets were being sensed. With 3 UAVs and 8+ targets there is no layout
that senses everything, so the constraint rejects every candidate and the
objective is a maximum over an always-infinite set. The hand-along chains, the
"cover" inversion field and the all_sensed_after_s scoring were all machinery for
that constraint, and they go with it.

The objective now is rescue delay under p_r = λ/(λ + tr Σ), and the real decision
is a RATE-versus-BREADTH tradeoff that simply did not exist when every target had
its own UAV:

  * A small set is revisited often, so its members hold low tr(Σ), draw a high
    rescue probability every slot, and clear FAST — which frees the UAV sooner,
    compounding into the next targets.
  * A large set spreads attention: every member gets some tracking, but each is
    revisited rarely, tr(Σ) climbs between visits and p_r falls for all of them.

Neither dominates. Concentrating drains the backlog fastest but leaves some
targets untouched for long stretches; spreading keeps everything weakly tracked
but may rescue nothing quickly. Which is right depends on the load regime, and
that judgement arrives in the brief rather than being rediscovered here.

Geometry matters through the SPREAD of a set, not the absolute position of its
members. A UAV takes one measurement per slot and its policy parks it where it
can serve the whole set, so a set whose members lie within about a sensing radius
(~700 m) of a common point is refreshed every few slots from that one spot. A set
spread wider than that admits no such spot: the UAV commutes, and every member
decays during the flight. So a compact set of four can beat a scattered set of
two. The distance table below is there for that comparison, not for a coverage
race.
"""

import re
import json

import numpy as np

from config.params import NUM_UAVS, V_MAX, DT, LAMBDA_RESCUE, ONE_TO_MANY
from agentic.llm   import LLMClient


class PlannerAgent:
    """
    LLM-based Planner Agent — emits {uav_id: [target ids]}.

    Uses the stronger model of the cascade. The tiering is what makes that
    affordable: the planner runs only on an approved alarm, perhaps a few dozen
    times in a 1000-slot episode, while the frequent call is the judge's cheap
    one. The old code was forced onto the light model because the planner ran on
    a fixed 20-slot grid and its latency was charged directly to an open coverage
    gap; with no coverage to lose and a judge debouncing the alarms, a slower and
    better answer is the right trade.
    """

    _SYSTEM_PROMPT = (
        "You are the Planner Agent in a multi-UAV search-and-rescue system. There are "
        "FEWER UAVs than targets. Full coverage is impossible and is NOT a goal — do not "
        "try to give every target its own UAV, and do not treat a crowded UAV as a fault. "
        "Each UAV is given a SET of targets and cycles its radar among them, sensing "
        "exactly ONE of them per slot: the member whose position estimate has decayed "
        "most. Every other member of that set goes unmeasured that slot and its "
        "uncertainty grows, so a bigger set means every member is refreshed less "
        "often. A target is rescued with "
        "probability lambda/(lambda + tr(Sigma)) each slot, so a member that was just "
        "sensed is very likely to be rescued soon, and one that has not been sensed for a "
        "long time is almost certainly not. Your objective is minimum AVERAGE RESCUE "
        "DELAY across all targets.\n"
        "\n"
        "Your output is a partition: every live target goes to AT MOST ONE UAV, and a UAV "
        "may hold any number of targets including none.\n"
        "Reply ONLY with a valid JSON object — no markdown, no text outside the JSON."
    )

    def __init__(self, log_path: str = "llm_log.jsonl", model: str = None):
        from config.params import PLANNER_MODEL
        self._model = model or PLANNER_MODEL
        self._llm = LLMClient(log_path=log_path, model=self._model)

    @property
    def latencies(self) -> list:
        return self._llm.latencies

    @property
    def slots(self) -> list:
        return self._llm.slots

    # ------------------------------------------------------------------
    def plan(self, state: dict, brief: dict, slot: int = 0, info: dict = None):
        """
        Generate a revised partition.

        Returns
        -------
        assignments : {uav_id: set(target ids)}
        reasoning   : the PA's justification
        """
        prompt = self._build_prompt(state, brief or {}, info or {})
        raw = self._llm.generate(
            system=self._SYSTEM_PROMPT,
            user=prompt,
            max_tokens=4096,
            agent="PA",
            slot=slot,
        )
        return self._parse(raw, state)

    # -- prompt ----------------------------------------------------------------

    def _build_prompt(self, state: dict, brief: dict, info: dict) -> str:
        from agentic.judge import JudgeAgent

        uav_ids = list(range(NUM_UAVS))
        live    = sorted(state["targets"])
        stale      = state.get("time_since_sensed", {})

        lines = []
        if brief:
            lines += [JudgeAgent.format_brief(brief), ""]

        lines += [
            "== SYSTEM STATE ==",
            f"UAVs: {len(uav_ids)} {uav_ids}   Live targets: {len(live)} {live}",
            f"Each UAV must therefore average {len(live) / max(len(uav_ids), 1):.1f} "
            f"targets. That is the situation, not a problem to fix.",
            "",
            "UAVs:",
        ]
        gamma_map = state.get("gamma", {})
        for i in range(NUM_UAVS):
            x, y, vx, vy = state["uavs"][i]
            speed  = float(np.sqrt(vx**2 + vy**2))
            rho    = float(state["rho"].get(i, 1.0))
            cur    = sorted(state["assignments"].get(i, ()))
            gamma = gamma_map.get(i)
            snr_str = f"{10.0 * np.log10(gamma):.1f} dB" if gamma else "n/a"
            lines.append(
                f"  UAV {i}: pos=({x:.0f},{y:.0f}) m  vel=({vx:.1f},{vy:.1f}) m/s  "
                f"speed={speed:.1f} m/s  uplinkSNR={snr_str}  energy={rho*100:.1f}%  "
                f"currently holds {cur}"
            )

        lines += [
            "",
            "Targets. rescue_rate is the CURRENT per-slot rescue probability implied by "
            "this target's uncertainty — read it as how close this target is to being "
            "saved. unsensed_for is how many slots since anyone measured it:",
        ]
        for k in live:
            mu, Sigma = state["targets"][k]
            tr_val = float(np.trace(Sigma[:2, :2]))
            pr     = LAMBDA_RESCUE / (LAMBDA_RESCUE + max(tr_val, 0.0))
            holder = [i for i in range(NUM_UAVS) if k in state["assignments"].get(i, ())]
            waited = state.get("t", 0) - state.get("birth_slot", {}).get(k, 0)
            lines.append(
                f"  Target {k}: est_pos=({mu[0]:.0f},{mu[1]:.0f}) m  "
                f"est_vel=({mu[2]:.1f},{mu[3]:.1f}) m/s  "
                f"rescue_rate={pr:.3f}  unsensed_for={stale.get(k, 0)} slots  "
                f"waiting={waited} slots  held_by={holder if holder else 'NOBODY'}"
            )

        # Distance table. Under overload this is about REVISIT COST — how far a UAV
        # must fly to cycle between the members it holds — not a race to cover.
        lines += [
            "",
            f"Distance from each UAV to each target, and the flight time at top speed "
            f"({V_MAX:.0f} m/s, {DT:.1f} s per slot). Sensing works out to roughly 700 m, "
            f"so a UAV does not have to arrive on top of a target to measure it:",
        ]
        for k in live:
            mu, _ = state["targets"][k]
            row = []
            for i in uav_ids:
                x, y = state["uavs"][i][0], state["uavs"][i][1]
                d = float(np.hypot(mu[0] - x, mu[1] - y))
                row.append(f"UAV{i}={d:.0f}m/{max(0.0, d - 700.0) / V_MAX / DT:.0f}slots")
            lines.append(f"  Target {k}: " + "  ".join(row))

        lines += [
            "",
            "== HOW TO CHOOSE ==",
            "The decision is a tradeoff between RATE and BREADTH, and it has no fixed "
            "answer — it depends on the load.",
            "  * Give a UAV FEWER targets and it revisits each one often, so their "
            "uncertainty stays low, their rescue_rate stays high, and they are saved "
            "quickly — which frees that UAV sooner to take on the next ones.",
            "  * Give a UAV MORE targets and each is revisited rarely, so every member's "
            "uncertainty climbs between visits and every member's rescue_rate falls. More "
            "targets get some attention, but none is close to rescue.",
            "Concentrating drains the backlog fastest but leaves some targets untouched "
            "for a long time. Spreading keeps everything weakly tracked but may rescue "
            "nothing soon. The judge's brief tells you which way the load is pointing; "
            "follow it.",
            "",
            "Also weigh:",
            "  * REVISIT COST, usually the deciding factor. A UAV takes ONE measurement "
            "per slot and parks where it can serve its whole set, so what matters is "
            "whether ONE POSITION EXISTS that reaches every member. Sensing reaches about "
            "700 m, so a set whose members sit within roughly 700 m of a common point can "
            "all be served from that point and each is refreshed every few slots. A set "
            "spread wider than that has no such point: the UAV must physically commute "
            "between members at about 12 m per slot, every member waits out that flight, "
            "and their uncertainty climbs the whole time. Two targets 1500 m apart in one "
            "set is far worse than the same two in different sets, even if that makes the "
            "set sizes uneven. Judge a set by its SPREAD, not only its size — but do not "
            "chase perfect geometry, since the targets keep moving.",
            "  * NEARLY DONE. A target with a high rescue_rate is about to be saved. "
            "Taking it away from the UAV that got it there throws away that progress and "
            "it starts over. Leave those alone.",
            "  * STRANDED. A target held by NOBODY has a rescue_rate that only falls. It "
            "waits forever unless you give it to someone.",
            "  * CHURN. Moving a target to a different UAV costs the new UAV a flight to "
            "reach it. A small predicted improvement is not worth that; leave settled sets "
            "settled, and change only what the brief points at.",
            "",
            "You MAY deliberately leave a target unassigned if the fleet is genuinely "
            "saturated and adding it to any set would slow that set's existing members "
            "more than it helps. That is a real choice under overload — but it means the "
            "target waits, so make it consciously and say so in your reasoning.",
            "",
        ]

        if not ONE_TO_MANY:
            lines += [
                "ABLATION CONSTRAINT: each UAV may hold AT MOST ONE target this run. "
                "Give each UAV a list of length 0 or 1.",
                "",
            ]

        _uav_hint = ", ".join(f'"{i}": [<target ids>]' for i in uav_ids)
        lines += [
            "== OUTPUT FORMAT (JSON only) ==",
            'Two fields. "reasoning": how you split the targets and why that split '
            "minimises total waiting — name the tradeoff you made. Then "
            '"assignments": the partition, one LIST of target ids per UAV '
            "(an empty list is allowed). A target id must appear under AT MOST ONE UAV.",
            "{",
            '  "reasoning": "<why this split>",',
            f'  "assignments": {{{_uav_hint}}}',
            "}",
        ]
        return "\n".join(lines)

    # -- response parsing ------------------------------------------------------

    def _parse(self, raw: str, state: dict):
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

        reasoning = data.get("reasoning", "")
        raw_asgn  = data.get("assignments", {})

        # Robustness fallback: the reasoning field is free-form text and can carry
        # an unescaped quote that breaks the outer JSON. The assignments block has
        # a rigid "i": [ints] shape and is emitted after it, so pull it out by
        # regex when the structured parse yielded nothing.
        if not (isinstance(raw_asgn, dict) and raw_asgn):
            block = re.search(r'"assignments"\s*:\s*\{([\s\S]*?)\}', raw)
            if block:
                raw_asgn = {
                    m.group(1): [int(n) for n in re.findall(r"\d+", m.group(2))]
                    for m in re.finditer(r'"(\d+)"\s*:\s*\[([^\]]*)\]', block.group(1))
                }

        live = set(state["targets"])

        assignments = {i: set() for i in range(NUM_UAVS)}
        claimed: set = set()

        for uav_str, val in (raw_asgn.items() if isinstance(raw_asgn, dict) else []):
            try:
                i = int(uav_str)
            except (TypeError, ValueError):
                continue
            if not (0 <= i < NUM_UAVS):
                continue
            # Accept a list, a bare int (a one-target set), or a dict wrapper.
            if isinstance(val, dict):
                val = val.get("targets", [])
            if isinstance(val, (int, float)):
                val = [val]
            if not isinstance(val, (list, tuple, set)):
                continue
            for v in val:
                try:
                    k = int(v)
                except (TypeError, ValueError):
                    continue
                # A dead id is dropped; a target the model listed twice goes to
                # the first UAV that claimed it, so the result is a partition.
                if k in live and k not in claimed:
                    assignments[i].add(k)
                    claimed.add(k)

        # NO coverage safety-net, by design. If the planner leaves a live target
        # in nobody's set, it stays unassigned: under overload that is a decision
        # the planner is explicitly allowed to make, and silently patching it here
        # would mean a repair rule, not the agent, was allocating. The detector
        # surfaces the unassigned target in the next digest and the judge decides
        # whether it is worth another replan — that delayed self-correction IS the
        # agentic loop.
        #
        # The one thing enforced is crash-safety: every UAV id carries a set,
        # empty if the planner gave it nothing.
        return assignments, reasoning
