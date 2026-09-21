import os
import time
import numpy as np

from config.params      import T_SLOTS, NUM_UAVS, MAX_TARGETS, DT
from evaluate.plot      import plot_trajectories
from evaluate.marl      import _finish, _log_slot
from evaluate.utils     import print_slot, print_assignments, seed_plots_dir


def eval_agentic(env, save_path: str, seed: int = None, births: bool = True):
    """MARL trajectory control + the three-tier agentic allocator.

    The allocator runs detector (every slot, deterministic) -> judge (light LLM,
    on an alarm) -> planner (heavy LLM, on approval), never blocking the
    simulation: the judge/planner round trip runs in a daemon thread and its
    result is installed whenever it lands.
    """
    from marl          import MAPPO
    from agentic       import AgenticAI
    from envs.births   import apply_births
    from envs.schedule import DisturbanceSchedule

    agent = MAPPO()
    agent.load(save_path)
    agent.actor.eval()

    plots_dir = os.path.join(seed_plots_dir(seed), "agentic")
    os.makedirs(plots_dir, exist_ok=True)
    agentic = AgenticAI(log_path=os.path.join(plots_dir, "llm_log.jsonl"))

    # Same seeding as marl mode: the ARRIVING LOAD (births, target motion,
    # measurement noise) is identical. Rescues are policy-dependent by
    # construction and will differ — that difference is the measurement.
    if seed is not None:
        np.random.seed(seed)
    schedule = DisturbanceSchedule(seed, env.T)
    env.seed_motion(seed)

    state = env.reset()
    info  = None

    # Both modes open from the SAME deterministic partition computed in
    # env.reset(), so the runs start identically and the only variable under
    # study is how the partition is revised from there.
    print("[AAI] Initial partition (deterministic, shared with the greedy baseline):")
    print_assignments(env)

    def _nan_pt():
        return np.array([np.nan, np.nan])

    uav_traj = {i: [env.uavs[i].pos2d.copy()] for i in range(NUM_UAVS)}
    tgt_traj = {k: [env.targets[k].pos.copy() if k in env.targets else _nan_pt()]
                for k in range(MAX_TARGETS)}

    print(f"[MAPPO + Agentic AI Eval] loaded from {save_path}")
    print("=" * 100)

    ev_born:    list = []
    ev_rescued: list = []

    backlog_log     = []
    rescued_cum_log = []
    delay_log       = []
    unassigned_log  = []
    mean_trace_log  = []
    rmse_log        = []
    pcrlb_log       = []
    pcrlb_per_target_log = {k: [] for k in range(MAX_TARGETS)}
    trace_per_target_log = {k: [] for k in range(MAX_TARGETS)}
    energy_log      = {i: [] for i in range(NUM_UAVS)}
    load_log        = {i: [] for i in range(NUM_UAVS)}
    assignment_log  = {i: [] for i in range(NUM_UAVS)}

    for t in range(T_SLOTS):
        born = apply_births(env, schedule) if births else []
        if born:
            ev_born.append((t + 1, list(born)))
            for k in born:
                tgt_traj[k] = [_nan_pt() for _ in tgt_traj[k]]

        state = env._system_state()

        # One call per slot. The detector runs inside it (deterministic, free);
        # the LLM tiers are woken only when it alarms and only when the judge
        # approves, and neither blocks this loop. `born` is passed explicitly
        # because a birth is the one event the env's own info cannot report — it
        # happens before the step, not inside it. Rescues come through `info`
        # from the previous step.
        new_asgn = agentic.step(state, info if info is not None else {}, born=born)

        if new_asgn is not None:
            env.apply_assignments(new_asgn)
            state = env._system_state()

            entry = agentic.log[-1]
            brief = entry["brief"]
            print(f"\n[Slot {t+1}] RE-PARTITION  (|K|={entry['backlog']})")
            print(f"  Regime   : {brief.get('regime')} ({brief.get('direction')})")
            print(f"  Changed  : {brief.get('whats_changed')}")
            print(f"  Judge    : {brief.get('rationale')}")
            print(f"  Planner  : {entry['reasoning']}")
            print_assignments(env)

        actions = agent.select_actions(state, info, deterministic=True)
        state, info = env.step(actions)

        for k, d in zip(info["rescued"], info["rescue_delays"]):
            ev_rescued.append((t + 1, k, d))

        _log_slot(info, env, backlog_log, rescued_cum_log, delay_log,
                  unassigned_log, mean_trace_log, pcrlb_log, rmse_log,
                  pcrlb_per_target_log, trace_per_target_log, energy_log,
                  load_log, assignment_log)

        # Real-time pacing so the LLM tiers experience the mission's actual
        # timescale: a judge/planner round trip that takes several seconds costs
        # several slots of stale partition, exactly as it would in deployment.
        time.sleep(DT)

        for i in range(NUM_UAVS):
            uav_traj[i].append(env.uavs[i].pos2d.copy())
        for k in range(MAX_TARGETS):
            tgt_traj[k].append(env.targets[k].pos.copy() if k in env.targets else _nan_pt())

        if (t + 1) % 100 == 0:
            print_slot(t, info, state)
            plot_trajectories(env, uav_traj, tgt_traj, plots_dir, t + 1)

    res = _finish("agentic", env, plots_dir, backlog_log, rescued_cum_log, delay_log,
                  unassigned_log, mean_trace_log, rmse_log, pcrlb_log,
                  pcrlb_per_target_log, trace_per_target_log, energy_log, load_log,
                  assignment_log, uav_traj, tgt_traj, ev_born, ev_rescued)

    # ---- tier accounting ---------------------------------------------------
    # The three numbers that justify the tiering: how many alarms the free
    # detector raised, how many of those the cheap judge actually looked at, and
    # how many reached the expensive planner. A judge that approved everything
    # would show n_replans == n_judge_calls and would not be earning its place.
    ja_lat, pa_lat = agentic.ja_latencies, agentic.pa_latencies
    res["n_alarms"]      = agentic.n_alarms
    res["n_judge_calls"] = len(ja_lat)
    res["n_replans"]     = len(agentic.log)

    print("=" * 100)
    print("Agentic tier accounting")
    print(f"  Detector alarms (free, deterministic) : {agentic.n_alarms}")
    print(f"  Judge calls  (light LLM)  : {len(ja_lat)}" +
          (f" | mean {np.mean(ja_lat):.2f}s | total {np.sum(ja_lat):.1f}s" if ja_lat else ""))
    print(f"  Planner calls (heavy LLM) : {len(pa_lat)}" +
          (f" | mean {np.mean(pa_lat):.2f}s | total {np.sum(pa_lat):.1f}s" if pa_lat else ""))
    print(f"  Re-partitions installed   : {len(agentic.log)}")
    if agentic.judge_log:
        from collections import Counter
        verdicts = Counter(v["decision"] for v in agentic.judge_log)
        regimes  = Counter(v["regime"]   for v in agentic.judge_log)
        print(f"  Judge verdicts : {dict(verdicts)}")
        print(f"  Regime reads   : {dict(regimes)}")
    print("=" * 100)

    np.savez(os.path.join(plots_dir, "llm_latency.npz"),
             ja_latencies=np.asarray(ja_lat, dtype=float),
             pa_latencies=np.asarray(pa_lat, dtype=float),
             ja_slots=np.asarray(agentic.ja_slots, dtype=int),
             pa_slots=np.asarray(agentic.pa_slots, dtype=int))

    if agentic.log:
        print("Re-partition log:")
        for entry in agentic.log:
            b = entry["brief"]
            print(f"  Slot {entry['slot']:4d} | |K|={entry['backlog']:2d} | "
                  f"{b.get('regime','?')}/{b.get('direction','?')} | "
                  f"{str(b.get('whats_changed',''))[:90]}")

    return res
