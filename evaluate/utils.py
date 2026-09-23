import os
import numpy as np

from config.params import NUM_UAVS, NUM_TARGETS


def seed_plots_dir(seed: int = None) -> str:
    """plots/seed{N} — or plots/seed_unspecified if no seed was given."""
    tag = f"seed{seed}" if seed is not None else "seed_unspecified"
    return os.path.join("plots", tag)


def save_raw_eval(seed: int, mode: str, res: dict) -> str:
    """
    Persist one run's series to plots/seed{N}/{mode}/raw_eval.npz.

    backlog is the objective: sum_t |K^t| is the total slots targets spend
    awaiting rescue (Eq. 19). assignments is (U, T, MAX_TARGETS) multi-hot,
    since a UAV holds a set.
    """
    d = os.path.join(seed_plots_dir(seed), mode)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "raw_eval.npz")

    arrays = {}
    for key in ("backlog", "rescued_cum", "delay", "unassigned",
                "mean_trace", "mean_pr", "rmse", "reward"):
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
    End-of-episode backlog summary.

    tgt_born    : [(slot, [target_ids]), ...]
    tgt_rescued : [(slot, target_id, delay_slots), ...]
    """
    n_born = sum(len(ids) for _, ids in tgt_born)
    n_resc = len(tgt_rescued)

    print("=" * 60)
    print("Episode evolution")
    print("-" * 60)
    print(f"UAVs: {NUM_UAVS} throughout")
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


def print_slot(t: int, info: dict):
    """Per-slot line, led by the mission metrics."""
    load_str = " ".join(f"{info['load'][i]}" for i in range(NUM_UAVS))
    snr_str  = " ".join(f"{info['snr_db'][i]:6.1f}" for i in range(NUM_UAVS))
    pr = info.get("rescue_prob", {})
    pr_str = " ".join(f"k{k}={v:.2f}" for k, v in sorted(pr.items()))
    print(
        f"Slot {t+1:4d} | |K|={info['backlog']:2d} "
        f"rescued={info['n_rescued_total']:3d} "
        f"D={info['avg_rescue_delay_s']:7.1f}s "
        f"unassigned={info['n_unassigned']} | load=[{load_str}]"
    )
    print(f"           | p_r[{pr_str}]")
    print(f"           | sensed={info['sensed_now']} | SNR(dB)=[{snr_str}]")


def print_assignments(env):
    """Print the assignment SETS, each member tagged with its current rescue
    probability — the one number that says whether a UAV is actually clearing
    what it holds rather than merely holding it."""
    from envs.rescue import rescue_prob
    print("=" * 100)
    print("UAV assignment sets:")
    for i in range(NUM_UAVS):
        members = sorted(env.uavs[i].assignment_set)
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
