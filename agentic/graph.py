"""Three-tier assignment stack.

  detector --[alarm?]--> judge --[replan?]--> planner --> END

  DETECTOR  deterministic, every slot, free.
  JUDGE     light LLM on an alarm; reads the digest, returns a scoping brief.
  PLANNER   heavy LLM on approval; reads full state, emits the sets.

The tiers are decoupled in time. A burst of events raises several alarms, but
the judge holds the accumulated digest across them and can approve a single
replan over the settled state instead of thrashing the partition. That is also
what separates this from a react-to-every-trigger baseline: there a trigger IS a
reallocation, here it is an observation a reasoning step may decline to act on.

Non-blocking: the judge/planner round trip runs in a daemon thread and the
simulation never waits for it.
"""

import threading
from typing import Optional, TypedDict

from langgraph.graph import StateGraph, END

from agentic.detector import EventDetector
from agentic.judge    import JudgeAgent
from agentic.planner  import PlannerAgent
from config.params    import NUM_UAVS, REPLAN_COOLDOWN


class _State(TypedDict):
    system_state: dict
    info:         dict
    digest:       dict
    digest_text:  str
    slot:         int
    brief:        Optional[dict]
    assignments:  Optional[dict]
    reasoning:    Optional[str]


class AgenticAI:
    """
    Agentic AI system for autonomous one-to-many task assignment.

    Public API is unchanged in shape: call step() once per slot with the current
    state and this slot's events; it returns the assignment table to install
    ({uav_id: set(target ids)}).
    """

    def __init__(self, log_path: str = "llm_log.jsonl"):
        self._detector = EventDetector()
        self._judge    = JudgeAgent(log_path=log_path)
        self._planner  = PlannerAgent(log_path=log_path)
        self._slot = 0
        self.log: list[dict] = []
        self.judge_log: list[dict] = []   # every judge verdict, including the holds

        self._graph = self._build_graph()

        self._thread:  threading.Thread | None = None
        self._pending: object = None
        self._lock = threading.Lock()

        self._last_replan_slot = -10**9
        self._hold_until_slot  = 0    # set when the judge answers "defer"

        # True when this slot's step() installed an LLM plan, False when it only
        # placed newborn targets (or returned nothing).
        self.last_step_replanned = False
        self.n_interim = 0            # newborn targets placed on the nearest UAV

    # ── latency accessors ─────────────────────────────────────────────────────
    @property
    def ja_latencies(self) -> list:
        return self._judge.latencies

    @property
    def pa_latencies(self) -> list:
        return self._planner.latencies

    @property
    def ja_slots(self) -> list:
        return self._judge.slots

    @property
    def pa_slots(self) -> list:
        return self._planner.slots

    @property
    def n_alarms(self) -> int:
        return self._detector.n_alarms

    # ── graph ─────────────────────────────────────────────────────────────────

    def _build_graph(self):
        builder = StateGraph(_State)
        builder.add_node("judge", self._judge_node)
        builder.add_node("planner", self._planner_node)
        builder.set_entry_point("judge")
        builder.add_conditional_edges(
            "judge",
            lambda s: "planner" if s["brief"] else END,
            {"planner": "planner", END: END},
        )
        builder.add_edge("planner", END)
        return builder.compile()

    def _judge_node(self, state: _State) -> dict:
        if state.get("brief"):
            return {}   # brief pre-filled (initial plan) — skip the judge call
        try:
            brief = self._judge.assess(state["digest"], state["digest_text"], state["slot"])
        except Exception as e:
            print(f"[Judge] LLM call failed ({e}); holding current partition.")
            brief = None
        verdict = getattr(self._judge, "last_brief", None)
        if verdict:
            self.judge_log.append({"slot": state["slot"], **verdict})
        return {"brief": brief}

    def _planner_node(self, state: _State) -> dict:
        new_asgn, reasoning = self._planner.plan(
            state["system_state"], state["brief"], state["slot"],
            info=state.get("info", {}),
        )
        return {"assignments": new_asgn, "reasoning": reasoning}

    # ── background worker ─────────────────────────────────────────────────────

    def _graph_worker(self, snapshot: dict):
        """Runs in a daemon thread; stores the result in self._pending when done."""
        seen = set(snapshot["system_state"].get("targets", {}))
        try:
            result = self._graph.invoke(snapshot)
            with self._lock:
                if result.get("brief") is not None and result.get("assignments") is not None:
                    self._pending = (result["assignments"], result["brief"],
                                     result.get("reasoning", ""), seen)
                else:
                    self._pending = _NO_REPLAN
        except Exception as e:
            print(f"[AAI] Worker failed: {e}")
            with self._lock:
                self._pending = _NO_REPLAN

    # ── interim placement ─────────────────────────────────────────────────────

    @staticmethod
    def _nearest_uav(system_state: dict, k: int) -> int:
        p = system_state["targets"][k][0][:2]
        return min(system_state["uavs"],
                   key=lambda i: float((system_state["uavs"][i][0] - p[0]) ** 2
                                       + (system_state["uavs"][i][1] - p[1]) ** 2))

    def _place(self, table: dict, system_state: dict, targets) -> None:
        """Put each target on the UAV nearest its estimate, in place."""
        for k in targets:
            table.setdefault(self._nearest_uav(system_state, k), set()).add(k)
            self.n_interim += 1

    # ── public API ────────────────────────────────────────────────────────────

    def step(self, system_state: dict, info: dict, born: list = None) -> Optional[dict]:
        """
        Advance one slot.

        Parameters
        ----------
        system_state : S^t from env._system_state()
        info         : the previous env.step() diagnostics (carries this slot's
                       rescues and any UAV-loss releases)
        born         : target ids born this slot

        Returns
        -------
        {uav_id: set(target ids)} to install, or None to carry the current
        partition forward. `last_step_replanned` says whether it is an LLM plan
        or only the interim placement of newborn targets.

        A newborn target is placed on the UAV nearest its estimate in the slot it
        appears, so no target waits unassigned while the LLMs think; the next
        plan decides where it really goes.
        """
        self._slot += 1
        result = None
        self.last_step_replanned = False
        live   = set(system_state.get("targets", {}))
        current = {i: set(ks) for i, ks in system_state.get("assignments", {}).items()}

        # ── 1. pick up a finished round trip ──────────────────────────────────
        with self._lock:
            pending = self._pending
            if pending is not None:
                self._pending = None

        if pending is not None and pending is not _NO_REPLAN:
            new_asgn, brief, reasoning, seen = pending
            # A plan computed in the background can be stale: targets may have been
            # rescued, or new ones born, while the LLMs were thinking. Drop the ids
            # that no longer exist rather than discarding the whole plan — under
            # this load one rescue during a round trip is routine, and throwing the
            # plan away for it would mean almost never landing one.
            clean = {i: {k for k in ks if k in live} for i, ks in new_asgn.items()}
            # Targets born after the plan's snapshot are not in it; they keep
            # their interim holder rather than being dropped. Targets the planner
            # saw and left out stay unassigned: that was its decision.
            planned = set().union(*clean.values()) if clean else set()
            for k in sorted(live - seen - planned):
                holder = next((i for i, ks in current.items() if k in ks), None)
                if holder is not None:
                    clean.setdefault(holder, set()).add(k)
                else:
                    self._place(clean, system_state, [k])
            self._last_replan_slot = self._slot
            self._detector.note_replan(self._slot)
            self.log.append({
                "slot":        self._slot,
                "brief":       brief,
                "assignments": {i: sorted(ks) for i, ks in clean.items()},
                "reasoning":   reasoning,
                "backlog":     info.get("backlog", 0) if info else 0,
            })
            result = clean
            self.last_step_replanned = True

        # ── 1b. interim placement of this slot's births ─────────────────────────
        base   = result if result is not None else current
        placed = set().union(*base.values()) if base else set()
        newborn = [k for k in (born or []) if k in live and k not in placed]
        if newborn:
            if result is None:
                result = {i: set(ks) for i, ks in current.items()}
            self._place(result, system_state, newborn)

        # ── 2. detector: record this slot's events, decide whether to alarm ───
        alarm = self._detector.observe(self._slot, born or [], info or {})

        # ── 3. launch the judge -> planner round trip if one is due ───────────
        thread_busy = self._thread is not None and self._thread.is_alive()
        cooling     = self._slot - self._last_replan_slot < REPLAN_COOLDOWN
        deferred    = self._slot < self._hold_until_slot

        if alarm and not thread_busy and not cooling and not deferred:
            digest = self._detector.digest(self._slot, system_state, info or {})
            snapshot = {
                "system_state": system_state,
                "info":         info or {},
                "digest":       digest,
                "digest_text":  self._detector.format_digest(digest),
                "slot":         self._slot,
                "brief":        None,
                "assignments":  None,
                "reasoning":    None,
            }
            self._thread = threading.Thread(
                target=self._graph_worker, args=(snapshot,), daemon=True)
            self._thread.start()

        # A "defer" verdict from the judge parks the next launch for a few slots,
        # so a burst of alarms is allowed to settle instead of each one re-arming
        # the round trip immediately.
        last = getattr(self._judge, "last_brief", None)
        if last and last.get("decision") == "defer" and last.get("wait_slots"):
            self._hold_until_slot = max(self._hold_until_slot,
                                        self._slot + int(last["wait_slots"]))
            self._judge.last_brief = None

        return result

    # ── initial assignment ────────────────────────────────────────────────────

    def initial_plan(self, system_state: dict, info: dict = None):
        """Launch the planner directly (skipping the judge) for the opening
        partition. There is nothing for the judge to weigh at slot 0 — every
        target needs a holder and no partition exists yet to preserve."""
        digest = self._detector.digest(0, system_state, info or {})
        snapshot = {
            "system_state": system_state,
            "info":         info or {},
            "digest":       digest,
            "digest_text":  self._detector.format_digest(digest),
            "slot":         0,
            "brief": {
                "decision": "replan", "wait_slots": 0,
                "regime": "overloaded", "direction": "steady",
                "focus_uavs": [], "focus_targets": [], "preserve_uavs": [],
                "whats_changed": "Mission start: no partition exists yet.",
                "emphasis": "Split the targets across the fleet for the fastest "
                            "overall clearance.",
                "rationale": "Opening allocation.",
            },
            "assignments":  None,
            "reasoning":    None,
        }
        self._thread = threading.Thread(
            target=self._graph_worker, args=(snapshot,), daemon=True)
        self._thread.start()


_NO_REPLAN = "no_replan"   # sentinel: round trip finished without a new partition
