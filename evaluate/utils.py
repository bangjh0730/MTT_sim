import os
import numpy as np

from config.params import NUM_UAVS, NUM_TARGETS


def seed_plots_dir(seed: int = None) -> str:
    """plots/seed{N} — or plots/seed_unspecified if no seed was given."""
    tag = f"seed{seed}" if seed is not None else "seed_unspecified"
    return os.path.join("plots", tag)


def save_raw_eval(seed: int, mode: str, pcrlb_log, total_energy, failure_slots,
                  rmse_log=None, birth_slots=None, pcrlb_per_target=None,
                  assignments=None, birth_events=None, death_events=None,
                  failure_events=None) -> str:
    """
    Persist one eval_marl/eval_agentic run's raw series to
    plots/seed{N}/{mode}/raw_eval.npz, so separate CLI invocations (different
    seeds and/or disturbance flags) can later be overlaid on one plot via
    evaluate.compare_runs — the in-memory return values don't otherwise
    survive past a single process.

    rmse_log is optional (older callers/npz files won't have it) — per-slot
    RMSE of the EKF position estimate against ground truth, mean over live
    targets each slot.

    birth_slots is optional — slots (1-indexed) at which a target was actually
    born (apply_target_dynamics can silently reject a scheduled birth if there's
    no room), matching the failure_slots convention. This is ground truth from
    the run itself; do NOT reconstruct birth slots by replaying
    envs.schedule.DisturbanceSchedule against config.params — if config.params
    (NUM_UAVS/NUM_TARGETS/etc.) has changed since this run was generated, the
    replayed RNG stream desyncs from slot 1 onward and produces onset slots
    that don't correspond to what actually happened in this trajectory.

    pcrlb_per_target : {target_id: [per-slot PCRLB, ...]} for target_id in
        [0, MAX_TARGETS) — saved as a (MAX_TARGETS, T) array, NaN wherever
        that id wasn't alive. Slot ids are reused across births/deaths (a
        fixed-size id space, not a growing counter — see envs/birth_death.py),
        so a single id's row is NOT one continuous physical target: it's
        however many separate "lives" happened to land on that id, each
        surrounded by NaN. Segment on contiguous non-NaN runs to recover
        individual lives, or cross-reference birth_events/death_events below
        for the exact slot each life started/ended.

    assignments : {uav_id: [per-slot assigned target_id or None, ...]} for
        uav_id in [0, NUM_UAVS) — saved as a (NUM_UAVS, T) int array, -1 where
        unassigned/inactive. Aligned 1:1 with pcrlb_per_target's slot axis, so
        "which UAV was tracking target k when its PCRLB moved" is a direct
        lookup rather than a replay. Same target-id-reuse caveat applies: a
        given (uav, target_id) pair spans that id's current life only.

    birth_events : [(slot, target_id), ...] — one row per target actually
        born (ground truth, mirrors birth_slots but keeps which id was born,
        not just when). death_events : [(slot, target_id, cause), ...]; cause is
        always "died" now that the tracking-quality-driven "fleet" removal is
        gone (see envs/birth_death.py), but runs saved before that change still
        carry "fleet" rows. Together these give the exact [birth_slot,
        death_slot) span of each life on a reused id, for consumers that don't
        want to infer it from NaN runs alone.

    failure_events : [(slot, uav_id, cause), ...] — one row per UAV actually
        lost, cause in {"shock", "battery"}. Mirrors failure_slots but keeps
        WHICH UAV went down, which failure_slots alone cannot express when two
        UAVs fail in the same slot. This is what lets a failure be attributed to
        the target it hit (the UAV's assignment at that slot); without it,
        evaluate/recovery_events.py has to reconstruct the victim from the slot
        the BS cleared an assignment, which is inferable but not free of edge
        cases. Runs saved before this field existed simply lack the key.
    """
    d = os.path.join(seed_plots_dir(seed), mode)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "raw_eval.npz")
    arrays = dict(
        pcrlb=np.asarray(pcrlb_log, dtype=float),
        energy=np.asarray(total_energy, dtype=float),
        failure_slots=np.asarray(failure_slots, dtype=int),
    )
    if rmse_log is not None:
        arrays["rmse"] = np.asarray(rmse_log, dtype=float)
    if birth_slots is not None:
        arrays["birth_slots"] = np.asarray(birth_slots, dtype=int)
    if pcrlb_per_target is not None:
        n_targets = max(pcrlb_per_target.keys()) + 1
        arrays["pcrlb_per_target"] = np.asarray(
            [pcrlb_per_target[k] for k in range(n_targets)], dtype=float)
    if assignments is not None:
        n_uav = max(assignments.keys()) + 1
        arrays["assignments"] = np.asarray(
            [[-1 if v is None else v for v in assignments[i]] for i in range(n_uav)],
            dtype=int)
    if birth_events is not None:
        arrays["birth_event_slots"]   = np.asarray([s for s, _ in birth_events], dtype=int)
        arrays["birth_event_targets"] = np.asarray([k for _, k in birth_events], dtype=int)
    if death_events is not None:
        arrays["death_event_slots"]   = np.asarray([s for s, _, _ in death_events], dtype=int)
        arrays["death_event_targets"] = np.asarray([k for _, k, _ in death_events], dtype=int)
        arrays["death_event_causes"]  = np.asarray([c for _, _, c in death_events], dtype="<U8")
    if failure_events is not None:
        arrays["failure_event_slots"] = np.asarray([s for s, _, _ in failure_events], dtype=int)
        arrays["failure_event_uavs"]  = np.asarray([i for _, i, _ in failure_events], dtype=int)
        arrays["failure_event_causes"] = np.asarray([c for _, _, c in failure_events], dtype="<U8")
    np.savez(path, **arrays)
    return path


def load_raw_eval(path: str):
    """Inverse of save_raw_eval: -> (pcrlb_log, total_energy, failure_slots)."""
    d = np.load(path)
    return d["pcrlb"], d["energy"], d["failure_slots"].tolist()


def print_evolution_summary(uav_shock, uav_battery, tgt_born, tgt_death,
                            final_active, final_targets):
    """
    End-of-episode summary of how the UAV fleet and target set evolved, as a single
    time-ordered timeline of all events.

    uav_shock   : list of (slot, [uav_ids])   shock failures
    uav_battery : list of (slot, uav_id)       battery depletions
    tgt_born    : list of (slot, [target_ids]) births
    tgt_death   : list of (slot, target_id, cause)  removals (fleet/died/uncertain)
    """
    n_shock = sum(len(ids) for _, ids in uav_shock)
    n_batt  = len(uav_battery)
    n_born  = sum(len(ids) for _, ids in tgt_born)
    n_death = len(tgt_death)

    print("=" * 60)
    print("Episode evolution")
    print("-" * 60)
    print(f"UAVs: {NUM_UAVS} started -> {final_active} active at end "
          f"({n_shock + n_batt} lost: {n_shock} shock, {n_batt} battery)")
    print(f"Targets: {NUM_TARGETS} started, +{n_born} born, -{n_death} lost "
          f"-> {final_targets} live at end")
    print("-" * 60)

    events = ([(s, f"UAV {ids} shock-failed")      for s, ids in uav_shock] +
              [(s, f"UAV {i} battery-depleted")    for s, i in uav_battery] +
              [(s, f"target {ids} born")           for s, ids in tgt_born] +
              [(s, f"target {i} lost ({cause})")   for s, i, cause in tgt_death])
    for s, txt in sorted(events):
        print(f"  slot {s:4d}  {txt}")
    print("=" * 60)


def print_slot(t: int, info: dict, state: dict):
    rho_str  = " ".join(f"{info['rho'][i]:.4f}"       for i in range(NUM_UAVS))
    snr_str  = " ".join(f"{info['snr_db'][i]:6.1f}"   for i in range(NUM_UAVS))
    rate_str = " ".join(f"{info['rate_Mbps'][i]:.3f}" for i in range(NUM_UAVS))
    tau_str  = " ".join(f"{state['tau'][i]:.3f}"      for i in range(NUM_UAVS))
    per_tgt  = " ".join(
        f"k{k}={v:.3g}" for k, v in sorted(info["pcrlb_per_target"].items())
    )
    print(
        f"Slot {t+1:3d} | PCRLB={info['pcrlb']:.4g} m^2 [{per_tgt}] | "
        f"Energy=[{rho_str}] | SNR(dB)=[{snr_str}] | "
        f"Rate(Mbps)=[{rate_str}] | tau(s)=[{tau_str}]"
    )


def print_assignments(env):
    print("=" * 100)
    print("UAV assignments:")
    for i in range(NUM_UAVS):
        print(f"  UAV {i} → Target {env.uavs[i].assignment}")
    print("=" * 100)


def print_assignment_history(history: dict):
    print("=" * 100)
    for i, targets in history.items():
        chain = " → ".join(f"Target {k}" for k in targets)
        print(f"  UAV {i}: {chain}")
    print("=" * 100)
