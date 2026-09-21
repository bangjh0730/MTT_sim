"""
LangGraph two-node graph connecting the Monitor Agent and Planner Agent.

  monitor  ──[event?]──►  planner  ──► END
              NO ──────────────────────► END

Scheduling (non-blocking):
  - MA looks at the system every _MONITOR_INTERVAL slots. Each launch (tick
    or config-change) restarts that 20-slot clock from the launch slot
    itself, independent of how long the MA/PA round trip takes to return.
  - MA also looks immediately when the caller reports a configuration change
    (target birth/death, UAV failure), instead of waiting for the next tick.
  - Only one MA/PA round trip runs at a time. If a tick or a config-change
    request arrives while one is still in flight, it is remembered and fired
    the instant the in-flight one frees up.
  - The simulation never waits for the LLM — the graph runs in a daemon thread.
"""

import threading
import numpy as np
from typing import Optional, TypedDict

from langgraph.graph import StateGraph, END

from agentic.monitor import MonitorAgent
from agentic.planner import PlannerAgent
from config.params   import NUM_UAVS, NUM_TARGETS

_MONITOR_INTERVAL = 20   # slots between consecutive MA launches (both cases)

_NO_REASSIGNMENT = "no_reassignment"   # sentinel: MA finished, no action needed


class _State(TypedDict):
    system_state: dict
    info:         dict
    prev_traces:  dict
    assignments:  dict
    slot:         int
    event:        Optional[str]
    reasoning:    Optional[str]


class AgenticAI:
    """
    Agentic AI system for autonomous task assignment.

    Non-blocking: the LangGraph (MA → PA) runs in a daemon thread.
    The simulation loop is never stalled waiting for the LLM.

    Scheduling example
    ------------------
    Slot 20 : tick — thread launched; next tick set to 20 + 20 = 40.
    Slot 40 : tick — thread launched (whether or not slot-20's round trip has
              even finished; if it's still running, this tick is remembered
              and fired the moment it frees up); next tick set to 60.
    Slot 53 : target dies — config-change launch ahead of schedule; next
              tick reset to 53 + 20 = 73 (the slot-60 tick is absorbed by
              this launch, not fired separately).
    Slot 73 : tick — thread launched; next tick set to 93.
    """

    def __init__(self, log_path: str = "llm_log.jsonl"):
        self._ma  = MonitorAgent(log_path=log_path)
        self._pa  = PlannerAgent(log_path=log_path)
        self._slot = 0
        self.log: list[dict] = []

        self._graph = self._build_graph()

        self._next_monitor_slot = _MONITOR_INTERVAL   # fixed grid: 20, 40, 60, ...
        self._relaunch_pending  = False                # tick/change missed while busy

        self._thread:  threading.Thread | None = None
        self._pending: object = None   # None | _NO_REASSIGNMENT | (asgn, event, reasoning)
        self._lock = threading.Lock()

    @property
    def ma_latencies(self) -> list:
        """Wall-clock seconds per MA reasoning call this run, in call order."""
        return self._ma.latencies

    @property
    def pa_latencies(self) -> list:
        """Wall-clock seconds per PA reasoning call this run, in call order."""
        return self._pa.latencies

    @property
    def ma_slots(self) -> list:
        """Sim slot of each MA reasoning call this run, aligned with ma_latencies."""
        return self._ma.slots

    @property
    def pa_slots(self) -> list:
        """Sim slot of each PA reasoning call this run, aligned with pa_latencies."""
        return self._pa.slots

    # ── graph ─────────────────────────────────────────────────────────────────

    def _build_graph(self):
        builder = StateGraph(_State)
        builder.add_node("monitor", self._monitor_node)
        builder.add_node("planner", self._planner_node)
        builder.set_entry_point("monitor")
        builder.add_conditional_edges(
            "monitor",
            lambda s: "planner" if s["event"] else END,
            {"planner": "planner", END: END},
        )
        builder.add_edge("planner", END)
        return builder.compile()

    def _monitor_node(self, state: _State) -> dict:
        if state.get("event"):
            return {}   # event pre-filled (e.g. initial assignment) — skip LLM call
        try:
            event = self._ma.assess(state["system_state"], state["prev_traces"], state["slot"])
        except Exception as e:
            print(f"[Monitor] LLM call failed ({e}); skipping reassignment.")
            event = None
        return {"event": event}

    def _planner_node(self, state: _State) -> dict:
        new_asgn, reasoning = self._pa.plan(
            state["system_state"], state["event"], state["slot"],
            info=state.get("info", {}),
        )
        return {
            "assignments": new_asgn,
            "reasoning":   reasoning,
        }

    # ── initial assignment (non-blocking) ────────────────────────────────────

    def initial_plan(self, system_state: dict):
        """
        Launch the full MA→PA graph in a background thread for initial assignment.
        UAVs start with no assignments; MA detects "no UAV assigned to any target"
        and triggers PA through the normal graph flow.
        Suppress regular MA polling until the plan lands.
        """
        self._next_monitor_slot = float("inf")
        snapshot = {
            "system_state": system_state,
            "info":         {},
            "prev_traces":  {},
            "assignments":  {},
            "slot":         0,
            "event":        "INITIAL ASSIGNMENT: No UAVs are assigned. Assign all UAVs to targets.",
            "reasoning":    None,
        }
        self._thread = threading.Thread(
            target=self._graph_worker,
            args=(snapshot,),
            daemon=True,
        )
        self._thread.start()

    # ── background worker ─────────────────────────────────────────────────────

    def _graph_worker(self, snapshot: dict):
        """Runs in a daemon thread; stores result in self._pending when done."""
        try:
            result = self._graph.invoke(snapshot)
            with self._lock:
                if result.get("event") is not None:
                    self._pending = (
                        result["assignments"],
                        result["event"],
                        result.get("reasoning", ""),
                    )
                else:
                    self._pending = _NO_REASSIGNMENT
        except Exception as e:
            print(f"[AAI] Worker failed: {e}")
            with self._lock:
                self._pending = _NO_REASSIGNMENT   # unblock scheduling even on failure

    # ── public API ────────────────────────────────────────────────────────────

    def step(
        self,
        system_state:   dict,
        info:           dict,
        prev_traces:    dict,
        assignments:    dict,
        config_changed: bool = False,
    ) -> dict:
        """
        Parameters
        ----------
        config_changed : True the slot a target is born/dies or a UAV fails —
            makes the MA look now instead of waiting for the next fixed tick.
        """
        self._slot += 1
        result = None

        # ── pick up thread result if ready ────────────────────────────────────
        with self._lock:
            pending = self._pending
            if pending is not None:
                self._pending = None

        if self._next_monitor_slot == float("inf") and pending is not None:
            # The suppressed initial-plan poll just landed — start the fixed
            # 20-slot grid from here.
            self._next_monitor_slot = self._slot + _MONITOR_INTERVAL

        if pending is not None and pending is not _NO_REASSIGNMENT:
            # Reassignment ready. Under the birth/death model a target may have been
            # removed while the MA→PA ran in the background, leaving the result stale
            # (assigning a UAV to a target that no longer exists). Discard such a
            # result — the current assignment is still valid and the next poll will
            # re-plan on fresh state — so a dead-target assignment never reaches the
            # actor's observation.
            new_asgn, event, reasoning = pending
            # None is a legitimate value (UAV left unassigned by the LLM); only a
            # non-live target id makes the plan stale.
            live = set(system_state.get("targets", {}))
            if all(k is None or k in live for k in new_asgn.values()):
                self.log.append({
                    "slot":        self._slot,
                    "event":       event,
                    "assignments": dict(new_asgn),
                    "reasoning":   reasoning,
                    "mdp_reward":  -float(info.get("pcrlb", 0.0)),
                })
                result = dict(new_asgn)
            else:
                print(f"[AAI] Discarded stale plan (targets changed while planning): {event}")

        # ── is a launch due? ───────────────────────────────────────────────────
        due = self._slot >= self._next_monitor_slot
        thread_busy = self._thread is not None and self._thread.is_alive()

        if (due or config_changed) and thread_busy:
            # Can't run a second MA/PA round trip concurrently (single pending
            # slot) — fire the instant the in-flight one frees up instead of
            # waiting for the next tick.
            self._relaunch_pending = True

        if (due or config_changed or self._relaunch_pending) and not thread_busy:
            self._relaunch_pending = False
            # Every launch (tick or config-change) restarts the 20-slot clock
            # from *this* slot, not from whenever the round trip completes —
            # so a launch at 53 makes the next one due at 73, not "+20 after
            # the PA call happens to finish."
            self._next_monitor_slot = self._slot + _MONITOR_INTERVAL
            snapshot = {
                "system_state": system_state,
                "info":         info,
                "prev_traces":  dict(prev_traces),
                "assignments":  dict(assignments),
                "slot":         self._slot,
                "event":        None,
                "reasoning":    None,
            }
            self._thread = threading.Thread(
                target=self._graph_worker,
                args=(snapshot,),
                daemon=True,
            )
            self._thread.start()

        return result if result is not None else dict(assignments)
