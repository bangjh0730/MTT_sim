"""
Planner Agent (PA) — reasoning and action module.

Invoked by the Monitor Agent when a reassignment event is detected.
Receives S^t and the event description, reasons over the full system
configuration, and outputs revised assignments {A^{t+1}_i = k}.
"""

import re
import json

import numpy as np

from config.params import NUM_UAVS, NUM_TARGETS, V_MAX
from agentic.llm   import LLMClient

class PlannerAgent:
    """
    LLM-based Planner Agent.

    Uses the same lightweight model as the Monitor. A stronger model (3.5 Flash)
    was tried and rejected: its reasoning did not finish inside the token budget,
    and PA latency is charged to the system in slots — a PA that replans better
    but answers several slots later leaves the coverage gap open longer, which is
    the very thing the replan exists to shorten. The prompt therefore carries the
    load instead: the state is pre-digested (fleet census, per-target distances
    sorted nearest-first) so the PA reads numbers off rather than deriving them.
    """

    _MODEL = "gemini-3.1-flash-lite"

    _SYSTEM_PROMPT = (
        "You are the Planner Agent (PA) in a multi-UAV multi-target tracking (MTT) system. "
        "The Monitor Agent has detected an issue with the current UAV assignments. "
        "Your job: reason over the full system state and issue a revised assignment for "
        "every active UAV. The scenario is dynamic -- targets are born and die anywhere on "
        "the map, UAVs fail, and you will be asked to replan again as it evolves -- so the "
        "state you are given is a snapshot of an ongoing system, not a final picture. "
        "Reply ONLY with a valid JSON object -- no markdown, no text outside the JSON."
    )

    def __init__(self, log_path: str = "llm_log.jsonl"):
        self._llm = LLMClient(log_path=log_path, model=self._MODEL)

    @property
    def latencies(self) -> list:
        """Wall-clock seconds per PA reasoning call this run, in call order."""
        return self._llm.latencies

    @property
    def slots(self) -> list:
        """Sim slot of each PA reasoning call this run, aligned with latencies."""
        return self._llm.slots

    def plan(
        self,
        state: dict,
        event: str,
        slot:  int = 0,
        info:  dict = None,
    ) -> tuple[dict, str]:
        """
        Generate a revised assignment.

        Parameters
        ----------
        state : system state S^t (UAV states, assignments, uplink SNR, energy,
                per-target estimate and uncertainty)
        event : event description from the Monitor Agent
        slot  : current simulation slot (for logging)
        info  : env diagnostics for the slot — supplies per-target PCRLB

        Returns
        -------
        assignments : {uav_id: target_id}
        reasoning   : PA's one-sentence justification
        """
        prompt = self._build_prompt(state, event, info or {})
        # Roomy budget: the reasoning field now holds genuine step-by-step working
        # that precedes the assignments, so a tight cap would risk truncating the
        # response before the assignments block is emitted.
        raw    = self._llm.generate(
            system=self._SYSTEM_PROMPT,
            user=prompt,
            max_tokens=4096,
            agent="PA",
            slot=slot,
        )
        return self._parse(raw, state)

    # -- prompt ----------------------------------------------------------------

    def _build_prompt(self, state: dict, event: str, info: dict) -> str:
        gamma_map = state.get("gamma", {})

        # Census up front. The PA has been observed miscounting the fleet -- holding a
        # co-tracked pair together, counting the remainder, and concluding a shortage
        # that does not exist -- then sacrificing a covered target to "resolve" it.
        # These two numbers are the state it got wrong, so state them rather than
        # making it tally the lists below.
        active_ids = [i for i in range(NUM_UAVS)
                      if state.get("active", {}).get(i, True)]
        live_now   = sorted(state["targets"])

        lines = [
            "== EVENT ==",
            event,
            "",
            "== SYSTEM STATE ==",
            f"Active UAVs: {len(active_ids)} {active_ids}   Live targets: "
            f"{len(live_now)} {live_now}",
            "",
            "UAVs:",
        ]
        for i in range(NUM_UAVS):
            x, y, vx, vy = state["uavs"][i]
            speed  = float(np.sqrt(vx**2 + vy**2))
            rho    = float(state["rho"].get(i, 1.0))
            active = state.get("active", {}).get(i, True)
            k_cur  = state["assignments"].get(i, "?")
            status = "[FAILED] " if not active else ""
            gamma  = gamma_map.get(i)
            snr_str = f"{10.0 * np.log10(gamma):.1f} dB" if gamma else "n/a"
            # A failed UAV's last assignment is meaningless -- showing it invites the PA
            # to reason about it as a live tracker (and to re-emit it in the output).
            cur_str = "currently->(none - failed)" if not active else f"currently->Target {k_cur}"
            lines.append(
                f"  UAV {i}: {status}pos=({x:.0f},{y:.0f}) m  vel=({vx:.1f},{vy:.1f}) m/s  "
                f"speed={speed:.1f} m/s  uplinkSNR={snr_str}  energy={rho*100:.1f}%  "
                f"{cur_str}"
            )

        pcrlb_map = info.get("pcrlb_per_target", {})
        live_ids  = sorted(state["targets"])   # dynamic under the birth/death model
        lines += ["", "Targets (EKF estimates):"]
        for k in live_ids:
            mu, Sigma = state["targets"][k]
            tr_val    = float(np.trace(Sigma[:2, :2]))
            # Active-only: a failed UAV may still carry a stale assignment to k, and
            # counting it here would report a target as multiply-tracked when its real
            # coverage is zero or one -- corrupting the marginal-value reasoning.
            trackers  = [i for i in range(NUM_UAVS)
                         if state["assignments"].get(i) == k
                         and state.get("active", {}).get(i, True)]
            pcrlb     = pcrlb_map.get(k)
            pcrlb_str = f"{pcrlb:.3g} m2" if pcrlb is not None else "n/a"
            lines.append(
                f"  Target {k}: est_pos=({mu[0]:.0f},{mu[1]:.0f}) m  "
                f"est_vel=({mu[2]:.1f},{mu[3]:.1f}) m/s  PCRLB={pcrlb_str}  "
                f"tr(Sigma)={tr_val:.0f} m2  tracked_by={len(trackers)} UAV(s) {trackers}"
            )

        # Precomputed distances, TARGET-centric and sorted nearest-first, so
        # "which UAV should cover target k" is a direct read rather than an
        # error-prone mental column-scan across a UAV-indexed table. Each UAV is
        # tagged [Tk] with the target it currently tracks, so whether pulling it
        # would leave its target uncovered (vs. it being a spare) is also visible
        # at a glance. It is still the LLM's call what to do with the numbers.
        # Which targets are currently uncovered. Used only to require that the candidate
        # set tries more than one UAV on each gap -- deliberately NOT to name which UAV
        # should take it, since nearest-to-the-gap is not always best: pulling the
        # nearest can force an expensive backfill that costs more than it saves.
        _covered = {state["assignments"].get(i) for i in range(NUM_UAVS)
                    if state.get("active", {}).get(i, True)}
        _nearest_to_gap = {k for k in live_now if k not in _covered}

        # Time-to-sensing is precomputed alongside the raw distance. The PA has been
        # observed deriving it from a UAV's CURRENT speed rather than v_max, inflating
        # the figure, and -- more often -- never computing it at all for candidates it
        # did not already favour. It is the one number a candidate is judged on, so
        # give it directly rather than asking for (d - 800) / v_max in the head.
        lines += ["", "Distances to each target — nearest active UAV first. "
                      "t = time for that UAV to reach ~800 m sensing range at top speed "
                      f"({V_MAX:.0f} m/s); [Tk] = the target that UAV currently tracks:"]
        for k in live_ids:
            mu, _ = state["targets"][k]
            ranked = []
            for i in range(NUM_UAVS):
                if not state.get("active", {}).get(i, True):
                    continue
                x, y = state["uavs"][i][0], state["uavs"][i][1]
                d   = float(np.hypot(mu[0] - x, mu[1] - y))
                cur = state["assignments"].get(i)
                ranked.append((d, i, cur))
            ranked.sort()
            row = "  ".join(
                f"UAV{i}={d:.0f}m/{max(0.0, d - 800.0) / V_MAX:.0f}s[T{cur}]"
                for d, i, cur in ranked
            )
            lines.append(f"  Target {k}: {row}")

        lines += [
            "",
            "== OBJECTIVE ==",
            "Your job is a constrained choice. HARD CONSTRAINT, first and non-negotiable: every "
            "live target must have an active UAV assigned to it. A layout that leaves any target "
            "bare is not a slow option or a low-scoring one -- it is not a layout at all; never "
            "propose it, never rank it. THEN, among layouts that cover everything, pick the one "
            "that minimises the time until ALL targets are being sensed.",
            "Why that time is the objective: a target any UAV is sensing (within ~800 m) sits at "
            "the floor, ~1e-4 m2; a target no one senses climbs fast to ~1e4 m2 -- eight orders of "
            "magnitude higher -- so it dominates the average PCRLB you are minimising. The moment "
            "a UAV reaches sensing range the value drops back to the floor within one slot, with "
            "nothing carried over, so error accrues ONLY while a target sits unsensed. Minimising "
            "the average is thus the same as getting the LAST target sensed as soon as possible.",
            f"How to measure a layout's time -- go TARGET BY TARGET, never by which UAV moved. "
            f"Every UAV flies at once at the same top speed ({V_MAX:.0f} m/s) and senses once "
            "within ~800 m. For each target, its arrival time is the smallest t among the UAVs "
            "THAT LAYOUT ASSIGNS TO IT (0 if one is already in range); read that off the distance "
            "table only for UAVs the layout actually points at that target. The layout's time is "
            "the LARGEST of these over all targets. Measuring per moved-UAV instead is the classic "
            "error: it scores a target as sensed by a UAV that is really flying somewhere else, "
            "and silently strands it.",
            "",
            "COVERING A GAP -- the move that matters. Because every UAV flies at once and only "
            "the LARGEST t counts, moving TWO UAVs costs no more time than moving one, as long "
            "as both flights are short: a pair costs max(t1, t2), not t1 + t2. This is what lets "
            "you HAND COVERAGE ALONG. Send a UAV that is a short flight from the uncovered target "
            "-- even if it is some other target's only tracker -- and AT THE SAME TIME send "
            "another UAV to backfill the target it vacated. Both end up sensed, and the whole "
            "thing costs only the longer of the two short flights. Example: a UAV 25 s from the "
            "gap is target X's sole tracker, and another UAV is 20 s from X; handing along covers "
            "the gap AND refills X in max(25, 20) = 25 s, versus one long 55 s+ flight if you "
            "leave that UAV put and send a distant spare straight to the gap. It need NOT be the "
            "single closest UAV -- a few seconds of extra flight changes nothing; what decides "
            "which UAVs to use is which whole layout gives the smallest largest-t. So a sole "
            "tracker a short flight from a gap is a PRIME mover, not an untouchable one. A "
            "hand-along can run ANY number of hops: if the UAV you backfill with is itself some "
            "target's only tracker, hand along once more from its target, and so on. What matters "
            "is how the chain ENDS -- it must terminate without opening a new hole: at a target "
            "that has a SPARE (two or more UAVs, see the tracked_by counts) or at an idle UAV, "
            "and NEVER by leaving a target bare. Grabbing a UAV off a sole-tracked target and "
            "stopping there does not close the gap, it just MOVES it. Every hop flies at once, so "
            "a chain of any length still costs only its single longest flight; among chains that "
            "end cleanly, all_sensed_after_s picks the shortest.",
            "",
            "== CONSTRAINTS ==",
            f"  * Build every candidate TARGET-FIRST to satisfy the hard constraint by "
            f"construction: go through {live_ids} and give each target a UAV (reuse one from a "
            f"target that has more than one) BEFORE reading off any times. That way no layout can "
            f"strand a target. A target briefly unsensed while its assigned UAV is still in "
            f"transit is fine; a target with no assigned UAV at all is the forbidden case.",
            "  * Never assign a UAV that failed or that lacks the energy to finish.",
            "  * When two layouts tie on time (largest t within ~10 s), break it on what the t's "
            "do NOT already capture: prefer moving the UAV with more energy left (the first UAV "
            "to hit 0% loses its target permanently), and prefer leaving the fleet spread rather "
            "than bunched. Don't oscillate a UAV between targets on successive plans.",
            "",
            # Removing this sweep was tested and reverted: without it the PA stranded a
            # live target on three consecutive calls, each time asserting coverage was
            # intact. It is the only step that forces it to read its own assign table
            # back id by id rather than trusting what it meant to do.
            f"FINAL CHECK, before you answer: read your chosen assign table back, {live_ids} one "
            f"id at a time, and name the active UAV on each. Any id with none means the layout "
            f'strands a target -- its time is "never" and you must not submit it; give the freed '
            f"target a UAV and re-check. Read the table, not what you meant to do.",
            "",
            "== OUTPUT FORMAT (JSON only) ==",
            'Three fields, in this order. "candidates": multiple complete layouts. Offer more '
            "when genuinely distinct, not padding. "
            "The trap to avoid: a layout can give every UAV a target yet still leave a TARGET "
            "with none. To make that impossible to miss, each candidate carries a \"cover\" "
            "field you build by INVERTING your own assign map -- go through EVERY live target and "
            "list the UAV id(s) your assign actually points at it. This is coverage shown as "
            "identities, not times, so it cannot be faked: a target whose list is EMPTY is "
            "stranded, the hard constraint is broken, and that candidate is INVALID -- do not "
            "submit it. Do not write a time for a target from a UAV that is not in its list; that "
            "UAV is flying elsewhere. "
            "Only once every target's list is non-empty, give each target its arrival -- the "
            "smallest distance-table t among the UAVs IN ITS LIST (0 if one is already in range). "
            '"all_sensed_after_s" is the LARGEST of those arrivals over all targets. '
            + (
                "Your candidates must be genuinely different layouts, not the same idea twice -- "
                "two UAVs pulled off the SAME target start from the same place and barely differ, "
                "which is no comparison. A hand-along (COVERING A GAP) is usually the fastest way "
                "to close a gap, so it is normally worth having one among them. Let "
                '"all_sensed_after_s" decide which wins. '
                if _nearest_to_gap else
                "Your candidates must be genuinely different layouts, not one idea written twice. "
            )
            + 'Then "reasoning": why your chosen candidate has the smallest '
            '"all_sensed_after_s"; if it is not the smallest, name the faster one you passed '
            'over and why. Then "assignments": that same layout as BARE INTEGERS (not '
            '{"target": id}).',
            "{",
            '  "candidates": [',
        ]
        _ids_hint    = "|".join(str(k) for k in live_ids)
        # Only active UAVs appear in the skeleton: listing a failed one invites the PA
        # to fill it in, which is how "2": 0 kept reappearing for a dead UAV.
        _assign_hint = ", ".join(f'"{i}": <{_ids_hint}>' for i in active_ids)
        # One slot per live TARGET, naming the UAV(s) the layout assigns to it (inverted
        # from assign) and their arrival. Naming UAVs -- not just a time -- forces the
        # coverage check to be consistent with assign: a stranded target shows an empty
        # list, which a fabricated time used to hide.
        _cover_hint = ", ".join(
            f'"{k}": "<uav ids on T{k}, or NONE> @<s>"' for k in live_ids
        )
        lines += [
            f'    {{"moves": "<which UAVs move>", "assign": {{{_assign_hint}}}, '
            f'"cover": {{{_cover_hint}}}, "all_sensed_after_s": <largest arrival, or INVALID>}},',
            f'    {{"moves": "<...>", "assign": {{{_assign_hint}}}, '
            f'"cover": {{{_cover_hint}}}, "all_sensed_after_s": <largest arrival, or INVALID>}}',
            '    <optionally one or two more layouts in this same shape, only if distinct>',
            "  ],",
            '  "reasoning": "<why the chosen candidate beats the others>",',
            '  "assignments": {',
        ]
        for n, i in enumerate(active_ids):
            comma = "," if n < len(active_ids) - 1 else ""
            lines.append(f'    "{i}": <{_ids_hint}>{comma}')
        lines += ["  }", "}"]
        return "\n".join(lines)

    # -- response parsing ------------------------------------------------------

    def _parse(
        self,
        raw: str,
        state: dict,
    ) -> tuple[dict, str]:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r'\{[\s\S]*\}', raw)
            try:
                data = json.loads(m.group()) if m else {}
            except json.JSONDecodeError:
                data = {}

        reasoning   = data.get("reasoning", "")
        raw_asgn    = data.get("assignments", {})

        # Robustness fallback: the reasoning field now carries free-form step-by-step
        # text, which can contain an unescaped quote/newline that breaks the outer
        # JSON. The assignments block is emitted AFTER reasoning and has a rigid
        # "i": k shape, so pull it out directly by regex when the structured parse
        # yielded nothing -- this recovers the plan even if the reasoning broke JSON.
        if not (isinstance(raw_asgn, dict) and raw_asgn):
            block = re.search(r'"assignments"\s*:\s*\{([^}]*)\}', raw)
            if block:
                raw_asgn = {m.group(1): int(m.group(2))
                            for m in re.finditer(r'"(\d+)"\s*:\s*(\d+)', block.group(1))}

        assignments = {}

        # Valid targets are the currently-live ids (dynamic under birth/death).
        live   = set(state["targets"])
        active = state.get("active", {})

        for uav_str, val in (raw_asgn.items() if isinstance(raw_asgn, dict) else []):
            try:
                i = int(uav_str)
            except (TypeError, ValueError):
                continue
            if not (0 <= i < NUM_UAVS):
                continue
            # The prompt tells the PA never to assign a failed UAV, but it sometimes
            # emits one anyway (usually just echoing the stale assignment back). A
            # failed UAV cannot sense, so accepting it would book phantom coverage.
            if not active.get(i, True):
                continue
            # The LLM may emit a dict {"target": k}, a bare int, or null/garbage
            # (e.g. null for a [FAILED] UAV the prompt told it not to assign).
            if isinstance(val, dict):
                tgt = val.get("target")
            elif isinstance(val, (int, float)):
                tgt = val
            else:
                tgt = None
            if tgt is None:
                continue
            try:
                tgt = int(tgt)
            except (TypeError, ValueError):
                continue
            if tgt in live:            # ignore ids for targets that no longer exist
                assignments[i] = tgt

        # No coverage safety-net: the PA owns every assignment decision, including its
        # mistakes. If it leaves a target with zero active trackers, that target stays
        # uncovered -- the Monitor Agent flags it on the next poll and the Planner fixes
        # its own error. That delayed, MA-driven self-correction is the agentic behavior;
        # a mechanical rule that silently patched coverage here would mean the *rule*, not
        # the agent, was guaranteeing the mission.
        #
        # The only non-negotiable is crash-safety: the env indexes every UAV, and
        # local_obs dereferences state["targets"][k], so every UAV must carry an
        # entry that is either a live target id or None (unassigned). A UAV the LLM
        # omitted, or pointed at a now-dead target, is therefore left on its current
        # target if that is still live, else set unassigned -- never invented onto
        # some target to force coverage.
        for i in range(NUM_UAVS):
            if i not in assignments:
                if not active.get(i, True):
                    assignments[i] = None    # failed: never carry a stale target forward
                    continue
                cur = state["assignments"].get(i)
                assignments[i] = cur if cur in live else None

        return assignments, reasoning
