import os
import numpy as np

from config.params import NUM_UAVS, NUM_TARGETS


def seed_plots_dir(seed: int = None) -> str:
    """plots/seed{N} — or plots/seed_unspecified if no seed was given."""
    tag = f"seed{seed}" if seed is not None else "seed_unspecified"
    return os.path.join("plots", tag)


def save_raw_eval(seed: int, mode: str, res: dict) -> str:
    """
    Persist one eval run's series to plots/seed{N}/{mode}/raw_eval.npz, so
    separate CLI invocations (different seeds, lambda values or flags) can later
    be overlaid on one plot via evaluate.compare_runs -- the in-memory return
    values don't otherwise survive past a single process.

    `res` is the dict returned by eval_marl / eval_agentic. It replaced a
    positional tuple when the metrics pivoted from PCRLB to rescue delay: the
    tuple had grown to ten fields and every consumer unpacked it by position, so
    adding the rescue series would have silently shifted all of them.

    The mission metrics it carries:
      backlog      (T,)  |K^t| per slot -- the series whose SUM is the objective,
                         since sum_t |K^t| counts the total slots targets spend
                         awaiting rescue (Eq. 19).
      rescued_cum  (T,)  cumulative rescues, so throughput reads as a slope.
      delay        (T,)  running D-bar in seconds.
      rescue_delays (R,) one completed delay per rescued target, in slots.
      unassigned   (T,)  live targets held by nobody. Under overload this is a
                         real allocation outcome, not an error, so it is logged.
      load         (U,T) per-UAV set size.
      assignments  (U,T,MAX_TARGETS) 0/1 set membership. A UAV holds a SET now,
                         so the old (U,T) array of single ids cannot express it;
                         this is the multi-hot form.

    Ground truth from the run itself; do NOT reconstruct birth slots by replaying
    envs.schedule.DisturbanceSchedule against config.params -- if config.params
    has changed since the run was generated, the replayed RNG stream desyncs from
    slot 1 onward.
    """
    d = os.path.join(seed_plots_dir(seed), mode)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "raw_eval.npz")

    arrays = {}
    for key in ("backlog", "rescued_cum", "delay", "unassigned", "pcrlb",
                "energy", "rmse", "mean_trace"):
        if res.get(key) is not None:
            arrays[key] = np.asarray(res[key], dtype=float)
    arrays["rescue_delays"] = np.asarray(res.get("rescue_delays", []), dtype=float)

    if res.get("load") is not None:
        n_uav = max(res["load"].keys()) + 1
        arrays["load"] = np.asarray([res["load"][i] for i in range(n_uav)], dtype=int)
    if res.get("assignments") is not None:
        n_uav = max(res["assignments"].keys()) + 1
        arrays["assignments"] = np.asarray(
            [res["assignments"][i] for i in range(n_uav)], dtype=np.int8)
    if res.get("pcrlb_per_target") is not None:
        n_t = max(res["pcrlb_per_target"].keys()) + 1
        arrays["pcrlb_per_target"] = np.asarray(
            [res["pcrlb_per_target"][k] for k in range(n_t)], dtype=float)
    if res.get("trace_per_target") is not None:
        n_t = max(res["trace_per_target"].keys()) + 1
        arrays["trace_per_target"] = np.asarray(
            [res["trace_per_target"][k] for k in range(n_t)], dtype=float)

    for name, ev in (("birth", res.get("birth_events")),
                     ("rescue", res.get("rescue_events"))):
        if ev is None:
            continue
        arrays[name + "_event_slots"]   = np.asarray([e[0] for e in ev], dtype=int)
        arrays[name + "_event_targets"] = np.asarray([e[1] for e in ev], dtype=int)
        if name == "rescue":
            arrays["rescue_event_delays"] = np.asarray([e[2] for e in ev], dtype=int)

    for scalar in ("avg_rescue_delay", "mean_completed_delay", "n_rescued",
                   "n_born", "n_replans", "n_alarms", "n_judge_calls"):
        if res.get(scalar) is not None:
            arrays[scalar] = np.asarray(res[scalar], dtype=float)

    np.savez(path, **arrays)
    return path


def load_raw_eval(path: str) -> dict:
    """Inverse of save_raw_eval: the saved arrays as a plain dict."""
    d = np.load(path)
    return {k: d[k] for k in d.files}


def print_evolution_summary(tgt_born, tgt_rescued, final_targets: int,
                            timeline: bool = False):
    """
    End-of-episode summary of how the backlog evolved.

    tgt_born    : list of (slot, [target_ids]) births
    tgt_rescued : list of (slot, target_id, delay_slots)

    The fleet no longer appears here: there is no UAV failure model, so the UAV
    count is a constant and reporting it every run said nothing. What changes
    over an episode is the TARGET set — arrivals and rescues — and the delay
    those rescues took.

    The per-event timeline is off by default. Targets are rescued on the order of
    every twenty slots, so printing every event buries the summary under a
    hundred lines; pass timeline=True when tracing a specific run.
    """
    n_born = sum(len(ids) for _, ids in tgt_born)
    n_resc = len(tgt_rescued)

    print("=" * 60)
    print("Episode evolution")
    print("-" * 60)
    print(f"UAVs: {NUM_UAVS} throughout (no failure model)")
    print(f"Targets: {NUM_TARGETS} initial, +{n_born} born, {n_resc} RESCUED "
          f"-> {final_targets} still awaiting rescue at end")
    if tgt_rescued:
        delays = [d for _, _, d in tgt_rescued]
        print(f"Completed rescue delays (slots): mean {np.mean(delays):.1f}  "
              f"median {np.median(delays):.1f}  max {max(delays)}")
    print("-" * 60)

    if timeline:
        events = ([(s, f"target {ids} born") for s, ids in tgt_born] +
                  [(s, f"target {i} RESCUED after {d} slots")
                   for s, i, d in tgt_rescued])
        for s, txt in sorted(events):
            print(f"  slot {s:4d}  {txt}")
        print("=" * 60)


def print_slot(t: int, info: dict, state: dict):
    """Per-slot line, led by the mission metrics rather than by PCRLB.

    Backlog and cumulative rescues come first because they ARE the objective:
    sum_t |K^t| is the quantity Eq. (19) averages. SNR/rate/energy follow as the
    sensing diagnostics behind them.
    """
    load_str = " ".join(f"{info['load'][i]}"          for i in range(NUM_UAVS))
    snr_str  = " ".join(f"{info['snr_db'][i]:6.1f}"   for i in range(NUM_UAVS))
    rate_str = " ".join(f"{info['rate_Mbps'][i]:.3f}" for i in range(NUM_UAVS))
    rho_str  = " ".join(f"{info['rho'][i]:.3f}"       for i in range(NUM_UAVS))
    pr = info.get("rescue_prob_per_target", {})
    pr_str = " ".join(f"k{k}={v:.2f}" for k, v in sorted(pr.items()))
    print(
        f"Slot {t+1:4d} | |K|={info['backlog']:2d} "
        f"rescued={info['n_rescued_total']:3d} "
        f"D={info['avg_rescue_delay_s']:7.1f}s "
        f"unassigned={info['n_unassigned']} | load=[{load_str}]"
    )
    print(
        f"           | p_r[{pr_str}]"
    )
    print(
        f"           | SNR(dB)=[{snr_str}] | Rate(Mbps)=[{rate_str}] | "
        f"Energy=[{rho_str}]"
    )


def print_assignments(env):
    """Print the assignment SETS, each member tagged with its current rescue
    probability -- the one number that says whether a UAV is actually clearing
    what it holds, rather than merely holding it."""
    from envs.rescue import rescue_prob
    print("=" * 100)
    print("UAV assignment sets:")
    for i in range(NUM_UAVS):
        uav = env.uavs[i]
        members = sorted(uav.assignment_set)
        if not members:
            print(f"  UAV {i}: (no targets)")
            continue
        detail = "  ".join(
            f"T{k}(p_r={rescue_prob(float(np.trace(env.ekf_state[k][1][:2, :2]))):.3f})"
            for k in members if k in env.ekf_state
        )
        print(f"  UAV {i}: {len(members)} target(s)  {detail}")
    held = {k for i in range(NUM_UAVS) for k in env.uavs[i].assignment_set}
    orphan = sorted(set(env.targets) - held)
    if orphan:
        print(f"  UNASSIGNED (nobody is sensing these): {orphan}")
    print("=" * 100)


def print_assignment_history(history: dict):
    print("=" * 100)
    for i, sets in history.items():
        chain = " -> ".join("{" + ",".join(str(k) for k in sorted(s)) + "}"
                            for s in sets)
        print(f"  UAV {i}: {chain}")
    print("=" * 100)
