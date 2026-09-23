import os
import time
import numpy as np

from config.params  import T_SLOTS, NUM_UAVS, MAX_TARGETS, DT
from evaluate.plot  import plot_trajectories, plot_eval_rescue
from evaluate.utils import (print_slot, print_assignments, seed_plots_dir,
                            print_evolution_summary)


def eval_agentic(env, save_path: str, seed: int = None, births: bool = True,
                 realtime: bool = True):
    """MAPPO trajectory control plus the three-tier agentic allocator.

    Never blocks: the judge/planner round trip runs in a daemon thread.

    realtime : sleep DT per slot so the LLM tiers see the mission's real
        timescale, a slow round trip costing slots of stale partition. Turn off
        only for plumbing tests.
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

    # The seed fixes the spawns, target motion, measurement noise and the birth
    # schedule. RESCUES are deliberately not fixed: they depend on the tracking
    # quality the policy achieves, which is what the run measures.
    if seed is not None:
        np.random.seed(seed)
    schedule = DisturbanceSchedule(seed, env.T)
    env.seed_motion(seed)

    state = env.reset()
    info  = None

    print(f"[MAPPO + Agentic AI Eval] loaded from {save_path}")
    print("[AAI] Initial partition:")
    print_assignments(env)
    print("=" * 100)

    def _nan_pt():
        return np.array([np.nan, np.nan])

    uav_traj = {i: [env.uavs[i].pos2d.copy()] for i in range(NUM_UAVS)}
    tgt_traj = {k: [env.targets[k].pos.copy() if k in env.targets else _nan_pt()]
                for k in range(MAX_TARGETS)}

    ev_born:    list = []
    ev_rescued: list = []

    backlog_log     = []
    rescued_cum_log = []
    delay_log       = []
    unassigned_log  = []
    mean_trace_log  = []
    mean_pr_log     = []
    rmse_log        = []
    reward_log      = []
    trace_per_target_log = {k: [] for k in range(MAX_TARGETS)}
    load_log        = {i: [] for i in range(NUM_UAVS)}
    assignment_log  = {i: [] for i in range(NUM_UAVS)}

    from marl.reward import per_agent_rewards

    for t in range(T_SLOTS):
        born = apply_births(env, schedule) if births else []
        if born:
            ev_born.append((t + 1, list(born)))
            for k in born:
                tgt_traj[k] = [_nan_pt() for _ in tgt_traj[k]]

        state = env._system_state()

        # One call per slot. The detector runs inside it (deterministic, free);
        # the LLM tiers wake only when it alarms and only when the judge approves,
        # and neither blocks this loop. `born` is passed explicitly because a
        # birth is the one event the env's own info cannot report — it happens
        # before the step, not inside it.
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

        # ---- per-slot logging ----
        backlog_log.append(info["backlog"])
        rescued_cum_log.append(info["n_rescued_total"])
        delay_log.append(info["avg_rescue_delay_s"])
        unassigned_log.append(info["n_unassigned"])
        reward_log.append(float(per_agent_rewards(info, env.uavs).sum()))

        tr = info["trace_pos_per_target"]
        pr = info["rescue_prob"]
        mean_trace_log.append(float(np.mean(list(tr.values()))) if tr else np.nan)
        mean_pr_log.append(float(np.mean(list(pr.values()))) if pr else np.nan)
        for k in range(MAX_TARGETS):
            trace_per_target_log[k].append(tr.get(k, np.nan))
        for i in range(NUM_UAVS):
            load_log[i].append(info["load"][i])
            row = np.zeros(MAX_TARGETS, dtype=np.int8)
            for k in info["assignments"].get(i, ()):
                if 0 <= k < MAX_TARGETS:
                    row[k] = 1
            assignment_log[i].append(row)

        ekf_means = info["ekf_means"]
        sq = [float(np.sum((ekf_means[k] - env.targets[k].pos) ** 2)) for k in ekf_means]
        rmse_log.append(float(np.sqrt(np.mean(sq))) if sq else np.nan)

        if realtime:
            time.sleep(DT)

        for i in range(NUM_UAVS):
            uav_traj[i].append(env.uavs[i].pos2d.copy())
        for k in range(MAX_TARGETS):
            tgt_traj[k].append(env.targets[k].pos.copy() if k in env.targets else _nan_pt())

        if (t + 1) % 100 == 0:
            print_slot(t, info)
            plot_trajectories(env, uav_traj, tgt_traj, plots_dir, t + 1)

    # ---- end of episode ----
    plot_eval_rescue(backlog_log, rescued_cum_log, delay_log, unassigned_log, plots_dir)
    plot_trajectories(env, uav_traj, tgt_traj, plots_dir, T_SLOTS)
    print_assignments(env)

    print("=" * 100)
    print("MISSION RESULT")
    print(f"  Targets appeared      : {env.n_born_total}")
    print(f"  Targets rescued       : {len(env.rescue_delays)}")
    print(f"  Still awaiting rescue : {len(env.targets)}")
    print(f"  Avg rescue delay (Eq. 19, over all target-slots): "
          f"{env.avg_rescue_delay:.2f} s")
    print(f"  Mean delay of completed rescues                 : "
          f"{env.mean_completed_delay:.2f} s")
    print(f"  Mean backlog |K^t|    : {np.mean(backlog_log):.2f} "
          f"(max {int(np.max(backlog_log))})")
    print(f"  Mean tr(Sigma)        : {np.nanmean(mean_trace_log):.4g} m^2")
    print(f"  Mean p_r              : {np.nanmean(mean_pr_log):.4f}")
    print("=" * 100)

    print_evolution_summary(ev_born, ev_rescued, final_targets=len(env.targets))

    res = {
        "backlog":              backlog_log,
        "rescued_cum":          rescued_cum_log,
        "delay":                delay_log,
        "unassigned":           unassigned_log,
        "mean_trace":           mean_trace_log,
        "mean_pr":              mean_pr_log,
        "rmse":                 rmse_log,
        "reward":               reward_log,
        "trace_per_target":     trace_per_target_log,
        "load":                 load_log,
        "assignments":          assignment_log,
        "rescue_delays":        list(env.rescue_delays),
        "birth_events":         [(s, k) for s, ids in ev_born for k in ids],
        "rescue_events":        list(ev_rescued),
        "avg_rescue_delay":     env.avg_rescue_delay,
        "mean_completed_delay": env.mean_completed_delay,
        "n_rescued":            len(env.rescue_delays),
        "n_born":               env.n_born_total,
    }

    # Tier accounting: alarms -> judge calls -> planner calls. A judge that
    # approved everything would show n_replans == n_judge_calls.
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
        print(f"  Judge verdicts : {dict(Counter(v['decision'] for v in agentic.judge_log))}")
        print(f"  Regime reads   : {dict(Counter(v['regime'] for v in agentic.judge_log))}")
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
