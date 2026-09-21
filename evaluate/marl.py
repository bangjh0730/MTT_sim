import os
import numpy as np

from config.params      import T_SLOTS, NUM_UAVS, MAX_TARGETS, DT, E_MAX, ONE_TO_MANY
from evaluate.plot      import plot_trajectories, plot_eval_rescue
from evaluate.utils     import (print_slot, print_assignments, seed_plots_dir,
                                print_evolution_summary)

_REASSIGN_INTERVAL = 40   # slots between periodic greedy re-partitions


def greedy_repartition(env) -> bool:
    """
    Deterministic set allocator — the "w/o agentic reasoning" ablation, and the
    baseline the agentic stack is measured against.

    It does what a sensible hand-written rule does under overload: place every
    unheld target on the UAV that can serve it most cheaply, where cost trades
    flight distance against how loaded that UAV already is. Load enters through a
    penalty because a UAV's revisit interval grows with its set size, so handing a
    target to an already-busy UAV slows every member it holds.

    What it deliberately does NOT do is reason about whether re-partitioning is
    worth doing at all. It reacts to every trigger — which is exactly the
    behaviour of the event-driven reallocation baselines, and the contrast the
    three-tier stack exists to draw: there, a trigger IS a reallocation; here a
    trigger is an observation that a separate judgement step may decline to act on.

    Only UNHELD targets are placed. A full re-match every interval was tried and
    is wrong for the same reason it was wrong in the coverage regime: two UAVs in
    a crossing geometry would keep swapping sets and neither ever arrives.

    Returns True if any set changed.
    """
    live = list(env.targets.keys())
    if not live:
        return False
    uav_ids = sorted(env.uavs)

    table = {i: set(env.uavs[i].assignment_set) for i in uav_ids}
    held  = {k for ks in table.values() for k in ks}
    loose = [k for k in live if k not in held]
    if not loose:
        return False

    tgt_pos = {k: env.ekf_state[k][0][:2] for k in live}
    # The load penalty is in METRES of equivalent travel per target already held,
    # and it is the parameter that decides whether this baseline is competent or
    # a strawman, so it was measured rather than guessed. Adding a member to a
    # set of size m does not cost one flight — it lengthens the revisit cycle for
    # ALL m existing members, so its true cost grows with the set. Swept against
    # the pursuit controller over 3 seeds:
    #
    #   penalty:   100 m    300 m    600 m   1000 m   2000 m   5000 m
    #   mean |K|:  11.95     8.25     4.78     2.02     2.44     2.25
    #   rescued:    26.3     35.0     44.0     56.3     57.0     54.7
    #
    # A weak penalty makes placement distance-greedy, which piles targets onto
    # whichever UAV happens to be nearest and starves it — six times the backlog
    # of a balanced placement. On a 2 km map a 1000 m penalty dominates most
    # distance differences, so the rule is effectively "balance the load, break
    # ties on proximity". That is the right answer here because UAVs are slow
    # relative to the map (12.5 m per slot, so crossing 1 km takes ~80 slots):
    # revisit interval, not initial approach, is what sets rescue delay.
    load_penalty = 1000.0

    changed = False
    # Farthest-first, so the awkward targets are placed while every UAV still has
    # a light set to offer them.
    bs = env.bs.pos
    for k in sorted(loose, key=lambda k: -float(np.linalg.norm(tgt_pos[k] - bs))):
        if not ONE_TO_MANY:
            room = [i for i in uav_ids if len(table[i]) == 0]
            if not room:
                break   # ablation: no UAV is free to take another target
        else:
            room = uav_ids
        i = min(room, key=lambda i: (
            float(np.linalg.norm(env.uavs[i].pos2d - tgt_pos[k]))
            + load_penalty * len(table[i])
        ))
        table[i].add(k)
        changed = True

    if changed:
        env.apply_assignments(table)
    return changed


def eval_marl(env, save_path: str, seed: int = None, births: bool = True):
    """MARL trajectory control + deterministic greedy re-partitioning.

    `births` defaults to True: with targets leaving by rescue, an episode without
    arrivals drains to an empty map within ~150 slots and the remaining 850
    measure nothing. Arrivals are the load the system is being tested on.
    """
    from marl          import MAPPO
    from envs.births   import apply_births
    from envs.schedule import DisturbanceSchedule

    agent = MAPPO()
    agent.load(save_path)
    agent.actor.eval()

    print(f"[MAPPO + greedy re-partition Eval] loaded from {save_path}")

    # Re-seed the global stream (targets, spawns, motion, measurement noise) and
    # build the birth schedule from the same seed, so the ARRIVING LOAD is
    # identical across modes. Rescues are NOT scheduled and will differ between
    # modes — that difference is the measurement, not a confound.
    if seed is not None:
        np.random.seed(seed)
    schedule = DisturbanceSchedule(seed, env.T)
    env.seed_motion(seed)

    state = env.reset()
    info  = None

    assignment_history = {i: [set(env.uavs[i].assignment_set)] for i in range(NUM_UAVS)}
    print_assignments(env)

    def _nan_pt():
        return np.array([np.nan, np.nan])

    uav_traj  = {i: [env.uavs[i].pos2d.copy()] for i in range(NUM_UAVS)}
    tgt_traj  = {k: [env.targets[k].pos.copy() if k in env.targets else _nan_pt()]
                 for k in range(MAX_TARGETS)}
    plots_dir = os.path.join(seed_plots_dir(seed), "marl")
    os.makedirs(plots_dir, exist_ok=True)

    n_asgn_events: int = 0
    next_reassign_slot: int = _REASSIGN_INTERVAL
    ev_born:    list = []
    ev_rescued: list = []   # (slot, target_id, delay_slots)

    # ---- mission metric series --------------------------------------------
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
    # Set membership as a 0/1 row per slot: a UAV holds a set, which the old
    # single-id-per-slot log could not represent.
    assignment_log  = {i: [] for i in range(NUM_UAVS)}

    for t in range(T_SLOTS):
        born = apply_births(env, schedule) if births else []
        if born:
            ev_born.append((t + 1, list(born)))
            for k in born:
                tgt_traj[k] = [_nan_pt() for _ in tgt_traj[k]]

        # A target rescued in the previous slot has already left its holder's set,
        # freeing capacity the rule should redistribute.
        rescued_prev = info.get("rescued", []) if info else []

        state = env._system_state()

        # React to EVERY event — no judgement step. This is the baseline's
        # defining behaviour, and the cost it pays for it.
        event = bool(born) or bool(rescued_prev)
        due   = t >= next_reassign_slot
        if due or event:
            next_reassign_slot = t + _REASSIGN_INTERVAL
            if greedy_repartition(env):
                for i in range(NUM_UAVS):
                    assignment_history[i].append(set(env.uavs[i].assignment_set))
                n_asgn_events += 1
                state = env._system_state()

        actions     = agent.select_actions(state, info, deterministic=True)
        state, info = env.step(actions)

        for k, d in zip(info["rescued"], info["rescue_delays"]):
            ev_rescued.append((t + 1, k, d))

        _log_slot(info, env, backlog_log, rescued_cum_log, delay_log,
                  unassigned_log, mean_trace_log, pcrlb_log, rmse_log,
                  pcrlb_per_target_log, trace_per_target_log, energy_log,
                  load_log, assignment_log)

        for i in range(NUM_UAVS):
            uav_traj[i].append(env.uavs[i].pos2d.copy())
        for k in range(MAX_TARGETS):
            tgt_traj[k].append(env.targets[k].pos.copy() if k in env.targets else _nan_pt())

        if (t + 1) % 100 == 0:
            print_slot(t, info, state)
            plot_trajectories(env, uav_traj, tgt_traj, plots_dir, t + 1)

    res = _finish("greedy", env, plots_dir, backlog_log, rescued_cum_log, delay_log,
                  unassigned_log, mean_trace_log, rmse_log, pcrlb_log,
                  pcrlb_per_target_log, trace_per_target_log, energy_log, load_log,
                  assignment_log, uav_traj, tgt_traj, ev_born, ev_rescued)
    res["n_replans"] = n_asgn_events
    print(f"Greedy re-partition events: {n_asgn_events}")
    return res


# ---------------------------------------------------------------------------
def _log_slot(info, env, backlog_log, rescued_cum_log, delay_log, unassigned_log,
              mean_trace_log, pcrlb_log, rmse_log, pcrlb_per_target_log,
              trace_per_target_log, energy_log, load_log, assignment_log) -> None:
    """Append one slot's metrics to every series. Shared by both eval modes so
    the two runs are logged identically and stay comparable."""
    backlog_log.append(info["backlog"])
    rescued_cum_log.append(info["n_rescued_total"])
    delay_log.append(info["avg_rescue_delay_s"])
    unassigned_log.append(info["n_unassigned"])
    tr = info["trace_pos_per_target"]
    mean_trace_log.append(float(np.mean(list(tr.values()))) if tr else np.nan)
    pcrlb_log.append(info["pcrlb"])

    per_tgt = info["pcrlb_per_target"]
    for k in range(MAX_TARGETS):
        pcrlb_per_target_log[k].append(per_tgt.get(k, np.nan))
        trace_per_target_log[k].append(tr.get(k, np.nan))
    for i in range(NUM_UAVS):
        energy_log[i].append(E_MAX * (1.0 - info["rho"][i]) / 1000.0)
        load_log[i].append(info["load"][i])
        row = np.zeros(MAX_TARGETS, dtype=np.int8)
        for k in info["assignments"].get(i, ()):
            if 0 <= k < MAX_TARGETS:
                row[k] = 1
        assignment_log[i].append(row)

    ekf_means = info["ekf_means"]
    sq_errs = [float(np.sum((ekf_means[k] - env.targets[k].pos) ** 2)) for k in ekf_means]
    rmse_log.append(float(np.sqrt(np.mean(sq_errs))) if sq_errs else np.nan)


# ---------------------------------------------------------------------------
def _finish(tag, env, plots_dir, backlog_log, rescued_cum_log, delay_log,
            unassigned_log, mean_trace_log, rmse_log, pcrlb_log,
            pcrlb_per_target_log, trace_per_target_log, energy_log, load_log,
            assignment_log, uav_traj, tgt_traj, ev_born, ev_rescued) -> dict:
    """Shared end-of-episode reporting and result packaging for both eval modes."""
    plot_eval_rescue(backlog_log, rescued_cum_log, delay_log, unassigned_log,
                     plots_dir)
    plot_trajectories(env, uav_traj, tgt_traj, plots_dir, len(backlog_log))
    print_assignments(env)

    print("=" * 100)
    print(f"[{tag}] MISSION RESULT")
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
    print("=" * 100)

    print_evolution_summary(ev_born, ev_rescued, final_targets=len(env.targets))

    total_energy = np.sum([energy_log[i] for i in energy_log], axis=0).tolist()
    return {
        "backlog":              backlog_log,
        "rescued_cum":          rescued_cum_log,
        "delay":                delay_log,
        "unassigned":           unassigned_log,
        "mean_trace":           mean_trace_log,
        "rmse":                 rmse_log,
        "pcrlb":                pcrlb_log,
        "pcrlb_per_target":     pcrlb_per_target_log,
        "trace_per_target":     trace_per_target_log,
        "energy":               total_energy,
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
