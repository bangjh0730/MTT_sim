"""
Monitor Agent (MA) — perception module.

Reads the full system state S^t every slot and decides, via LLM, whether
the current UAV-to-target assignments remain appropriate.  Returns a brief
event description when reassignment is warranted, or None to carry forward.
"""

import re
import numpy as np
from typing import Optional

from config.params    import NUM_UAVS, NUM_TARGETS
from agentic.llm      import LLMClient


class MonitorAgent:
    """
    LLM-based Monitor Agent.

    The LLM receives the formatted system state and decides autonomously
    whether the current assignments should be revised and why.
    Uses Gemini 3.1 Flash Lite — a lightweight, fast model — since it's a cheap
    yes/no coverage check run every active slot.
    """

    _MODEL = "gemini-3.1-flash-lite"

    _SYSTEM_PROMPT = (
        "You are the Monitor Agent (MA) in a multi-UAV multi-target tracking (MTT) system. "
        "The UAV flight controller (MARL policy) already handles collision avoidance, SNR "
        "optimisation, uplink scheduling, and flying each UAV to its assigned target — do "
        "NOT flag any of those. "
        "Your sole responsibility is UAV-to-target ASSIGNMENT coverage: whether every "
        "target still has an active UAV responsible for it. "
        "A target is COVERED as long as at least one active UAV (not [FAILED], energy > 0%) "
        "is assigned to it. Flag a reassignment ONLY when coverage is broken or about to "
        "break:\n"
        "  * a target has zero active assigned UAVs (its tracker failed, or it was left "
        "unassigned), or\n"
        "  * a target's only active tracker is about to drop out (energy nearly depleted), or\n"
        "  * an active UAV has no assigned target (assigned->Target None, e.g. its target was "
        "lost) — it should be put back to work on a target.\n"
        "Nothing else warrants a reassignment. In particular, do NOT flag:\n"
        "  * a target whose tr(Σ) / PCRLB is high or still rising, or whose assigned UAV is "
        "far away — that UAV is simply still flying toward it, and reassigning only restarts "
        "the travel and makes tracking worse;\n"
        "  * imbalance — there are more targets than UAVs can cover evenly, so one target "
        "will always have fewer UAVs and higher uncertainty. That is normal. Moving a UAV "
        "onto the worst-tracked target just makes the one it left the new worst, and the "
        "UAVs oscillate with no net gain.\n"
        "Judge coverage, not current distance or uncertainty. When every target has an "
        "active assigned UAV, the answer is NO. "
        "Reason ONLY about the target ids explicitly listed in the state. Target ids are "
        "NOT contiguous -- targets are born and die, so their ids have gaps (you may see "
        "e.g. 0, 2, 3, 5). A number that is absent from the list (e.g. 1 or 4) is NOT an "
        "uncovered target -- it means no such target exists. Never invent or flag a target "
        "id that is not in the listed set. "
        "Reply with exactly one line — no extra text:\n"
        '  "YES: Target <id> - <what happened to it>" — if reassignment is needed\n'
        '  "NO" — if every listed target is covered by an active UAV\n'
        "The Planner sees ONLY your one line as its description of the event, so a bare "
        'number tells it nothing. Always write the word "Target" followed by the id, then '
        "what changed. Name every affected target if there is more than one. For example: "
        '"YES: Target 3 - its only tracker UAV 2 failed", or "YES: Target 5 - newly born '
        'and unassigned", or "YES: Target 0 - its only tracker UAV 4 is nearly out of '
        'energy".'
    )

    def __init__(self, log_path: str = "llm_log.jsonl"):
        self._llm = LLMClient(log_path=log_path, model=self._MODEL)

    @property
    def latencies(self) -> list:
        """Wall-clock seconds per MA reasoning call this run, in call order."""
        return self._llm.latencies

    @property
    def slots(self) -> list:
        """Sim slot of each MA reasoning call this run, aligned with latencies."""
        return self._llm.slots

    def assess(self, state: dict, prev_traces: dict, slot: int = 0) -> Optional[str]:
        """
        Ask the LLM whether the current assignments should be revised.

        Parameters
        ----------
        state       : system state S^t from env._system_state()
        prev_traces : {k: tr(Σ[:2,:2])} from the previous slot (for trend info)
        slot        : current simulation slot (for logging)

        Returns
        -------
        Event description string if reassignment needed, else None.
        """
        prompt = self._build_prompt(state, prev_traces)
        answer = self._llm.generate(
            system=self._SYSTEM_PROMPT,
            user=prompt,
            max_tokens=256,
            agent="MA",
            slot=slot,
        )

        if answer.upper().startswith("YES"):
            # Extract the description after "YES:"
            parts  = answer.split(":", 1)
            reason = parts[1].strip() if len(parts) > 1 else "Reassignment required."

            # The reply spec asks for "Target <id> - <what happened>", but a terse
            # model sometimes answers "YES: 1". That reaches the Planner as the event
            # description "1", which tells it nothing and also slips past the id guard
            # below (the regex needs the word "Target"). Normalise a bare id into the
            # canonical form so the Planner gets a usable line and the guard applies.
            m_bare = re.fullmatch(r"[Tt]?\s*(\d+)\.?", reason)
            if m_bare:
                reason = f"Target {m_bare.group(1)} - lost or is losing its active tracker"

            # Guard against hallucinated target ids: the lightweight MA sometimes
            # flags ids that don't exist (e.g. gap numbers 1, 4 when live ids are
            # 0,2,3,5), inferring a contiguous id space. If the reason names target
            # ids and NONE of them are actually live, it's a false alarm -- don't
            # wake the Planner. (If it names no ids, or at least one real one, we
            # pass it through and let the PA reason on fresh state.)
            named = [int(n) for n in re.findall(r"[Tt]arget\s*(\d+)", reason)]
            if named and not any(k in state["targets"] for k in named):
                print(f"[Monitor] Ignored hallucinated event (no live target among "
                      f"{named}; live={sorted(state['targets'])}).")
                return None
            return reason
        return None

    def _build_prompt(self, state: dict, prev_traces: dict) -> str:
        lines = ["== CURRENT SYSTEM STATE ==", "", "UAVs:"]

        for i in range(NUM_UAVS):
            x, y, *_ = state["uavs"][i]
            rho    = float(state["rho"].get(i, 1.0))
            active = state.get("active", {}).get(i, True)
            k_cur  = state["assignments"].get(i)

            # A failed UAV keeps its last assignment in the env (the PA leaves an
            # omitted UAV on its current target), but showing that stale id here reads
            # as a live tracker that never moves and invites the MA to re-flag a
            # long-settled failure. Its assignment is meaningless once it is inactive,
            # so don't show one.
            if not active:
                status   = "[FAILED] "
                asgn_str = "assigned→(none - failed)"
                dist_str = ""
            else:
                status   = ""
                asgn_str = f"assigned→Target {k_cur}"
                if k_cur is not None and k_cur in state["targets"]:
                    mu, _ = state["targets"][k_cur]
                    dist  = float(np.sqrt((x - mu[0])**2 + (y - mu[1])**2))
                    dist_str = f"dist_to_target={dist:.0f}m"
                else:
                    dist_str = "no target assigned"

            lines.append(
                f"  UAV {i}: {status}pos=({x:.0f},{y:.0f}) m  "
                f"energy={rho*100:.1f}%  {asgn_str}  {dist_str}".rstrip()
            )

        lines += ["", "Targets (EKF estimates):"]
        for k in sorted(state["targets"]):
            mu, Sigma   = state["targets"][k]
            trace_now   = float(np.trace(Sigma[:2, :2]))
            trace_prev  = prev_traces.get(k, trace_now)
            delta_str   = f"Δ={trace_now - trace_prev:+.0f}" if k in prev_traces else "new target"
            trackers = [
                i for i in range(NUM_UAVS)
                if state["assignments"].get(i) == k
                and state.get("active", {}).get(i, True)
            ]
            lines.append(
                f"  Target {k}: pos=({mu[0]:.0f},{mu[1]:.0f}) m  "
                f"vel=({mu[2]:.1f},{mu[3]:.1f}) m/s  "
                f"tr(Σ)={trace_now:.0f} m² ({delta_str})  "
                f"tracked_by=UAVs{trackers}"
            )

        present_ids = sorted(state["targets"])
        lines += [
            "",
            "(UAVs marked [FAILED] / with 0% energy are inactive and cannot track.)",
            f"The targets currently present are EXACTLY {present_ids}. No other target "
            f"exists right now; ids not in this list (gaps in the numbering) are targets "
            f"that have died, not uncovered targets.",
            "",
            "Should the current assignments be revised?",
        ]
        return "\n".join(lines)
