import os
import time
import numpy as np

from config.params      import T_SLOTS, NUM_UAVS, NUM_TARGETS, MAX_TARGETS, DT, E_MAX
from evaluate.plot      import plot_trajectories, plot_eval_pcrlb
from evaluate.utils     import (print_slot, print_assignments, seed_plots_dir,
                                print_evolution_summary)


def eval_agentic(env, save_path: str, seed: int = None,
                 failures: bool = False, births: bool = False):
    from marl              import MAPPO
    from agentic           import AgenticAI
    from envs.failure      import apply_uav_failures
    from envs.birth_death  import apply_target_dynamics
    from envs.schedule     import DisturbanceSchedule

    agent   = MAPPO()
    agent.load(save_path)
    agent.actor.eval()

    # Computed early so the LLM transaction log lands next to this run's other
    # artifacts (plots/seed{N}/agentic/) instead of the cwd-relative default,
    # which would otherwise be overwritten by the next eval run.
    plots_dir = os.path.join(seed_plots_dir(seed), "agentic")
    os.makedirs(plots_dir, exist_ok=True)
    agentic = AgenticAI(log_path=os.path.join(plots_dir, "llm_log.jsonl"))

    # Re-seed the global stream (targets, spawns, motion, measurement noise) and build
    # the disturbance schedule from the same seed. The schedule pre-draws all births /
    # deaths / failures so they match marl mode exactly, regardless of the LLM's
    # non-determinism or how much of the global stream each mode consumes.
    if seed is not None:
        np.random.seed(seed)
    schedule = DisturbanceSchedule(seed, env.T)
    env.seed_motion(seed)   # dedicated RNG so target motion matches marl mode

    state = env.reset()
    info  = None

    # Both eval modes share the SAME deterministic angular initial assignment
    # (computed in env.reset), so the two runs start identically and the only
    # variable under study is the reassignment policy. The agentic AI is invoked
    # only for mid-episode reassignments (MA polling), not the initial plan.
    assignments = {i: env.uavs[i].assignment for i in range(NUM_UAVS)}
    print("[AAI] Initial assignment (deterministic, shared with MARL baseline):")
    print_assignments(env)

    prev_traces = {
        k: float(np.trace(state["targets"][k][1][:2, :2]))
        for k in state["targets"]
    }
    # BS-observable active set carried across slots. A UAV failure (shock or battery)
    # only enters state["active"] once the BS declares it lost — N_FAIL_DETECT silent
    # slots after the physical failure. That later slot, not the failure slot, is when
    # the coverage gap first becomes visible to the MA, so it is when we must poll.
    prev_bs_active = {i: True for i in range(NUM_UAVS)}
    failure_log:   dict = {}   # {uav_id: traj_index} — physical failure slot
    failure_slots: list = []   # slots (1-indexed) at which any UAV failed
    birth_slots:   list = []   # slots (1-indexed) at which a target was actually born
    # Evolution timeline for the end-of-episode summary.
    ev_shock:   list = []   # (slot, [uav_ids]) shock failures
    ev_battery: list = []   # (slot, uav_id)    battery depletions
    ev_born:    list = []   # (slot, [target_ids])
    ev_death:   list = []   # (slot, target_id, cause)

    def _nan_pt():
        return np.array([np.nan, np.nan])

    uav_traj  = {i: [env.uavs[i].pos2d.copy()] for i in range(NUM_UAVS)}
    # Per-target logs span every id slot [0, MAX_TARGETS); NaN whenever that target
    # is not alive, so births/deaths break the plotted line.
    tgt_traj  = {k: [env.targets[k].pos.copy() if k in env.targets else _nan_pt()]
                 for k in range(MAX_TARGETS)}

    print(f"[MAPPO + Agentic AI Eval] loaded from {save_path}")
    print("=" * 100)

    pcrlb_log            = []
    rmse_log             = []
    pcrlb_per_target_log = {k: [] for k in range(MAX_TARGETS)}
    energy_log           = {i: [] for i in range(NUM_UAVS)}
    # Which target (or None) each UAV is assigned to, per slot — aligned 1:1 with
    # pcrlb_log/pcrlb_per_target_log so a consumer can tell, at the slot a given
    # target's PCRLB was perturbed, which UAV (if any) was responsible for it.
    assignment_log       = {i: [] for i in range(NUM_UAVS)}
    pcrlb_sum:   float   = 0.0   # Σ_t Σ_k PCRLB_k^t
    pcrlb_count: int     = 0     # Σ_t |K^t|
    for t in range(T_SLOTS):
        # Physical failures take effect at the START of the slot, so a failed UAV
        # contributes no measurement from this slot onward.
        failed = apply_uav_failures(env, schedule) if failures else []
        if failed:
            print(f"\n[Slot {t+1}] UAV FAILURE: UAVs {failed} physically lost")
            failure_slots.append(t + 1)
            ev_shock.append((t + 1, list(failed)))
            for i in failed:
                if i not in failure_log:
                    failure_log[i] = len(uav_traj[i]) - 1   # last live trajectory index

        # Target birth/death. A resulting coverage gap or freed UAV wakes the MA→PA
        # this same slot (config_changed below), not just on the next 20-slot tick.
        born, removed = apply_target_dynamics(env, schedule) if births else ([], [])
        if born or removed:
            if born:
                print(f"\n[Slot {t+1}] TARGET BIRTH: targets {born} appeared")
                ev_born.append((t + 1, list(born)))
                birth_slots.append(t + 1)
                for k in born:
                    # Fixed-slot ids are reused, so blank the previous occupant's
                    # path — the reborn target is drawn only from its own birth,
                    # not as one line stretching back before it appeared. Keeps the
                    # legend to a bounded Target 0..MAX_TARGETS-1.
                    tgt_traj[k] = [_nan_pt() for _ in tgt_traj[k]]
            if removed:
                ids = [r[0] for r in removed]
                print(f"\n[Slot {t+1}] TARGET LOST: targets {ids} removed")
                for k, cause in removed:
                    ev_death.append((t + 1, k, cause))
            # A removed target left its trackers unassigned — resync so the snapshot
            # the MA reads reflects that.
            assignments = {i: env.uavs[i].assignment for i in range(NUM_UAVS)}

        # Rebuild the observation to reflect the current target set / assignments.
        state = env._system_state()

        # A UAV loss becomes actionable only when the BS observes it: state["active"]
        # flips the UAV inactive (and env clears its assignment) N_FAIL_DETECT slots
        # after the physical failure. Trigger the MA on that transition — reacting at
        # the physical-failure slot is futile because the loss isn't in the observed
        # state yet. This covers both shock failures and battery depletions, since both
        # go silent and are detected the same way; `failed` (shock only, and premature)
        # is kept purely for the trajectory/plot logging above.
        bs_active  = state["active"]
        newly_lost = [i for i in range(NUM_UAVS)
                      if prev_bs_active.get(i, True) and not bs_active.get(i, True)]
        prev_bs_active = dict(bs_active)
        if newly_lost:
            print(f"\n[Slot {t+1}] BS DETECTED UAV LOSS: UAVs {newly_lost} — reassignment poll")

        new_asgn = agentic.step(
            state, info if info is not None else {},
            prev_traces, assignments,
            config_changed=bool(newly_lost) or bool(born) or bool(removed),
        )

        if new_asgn != assignments:
            assignments = new_asgn
            for i in range(NUM_UAVS):
                env.uavs[i].assignment = assignments[i]
            state["assignments"] = dict(assignments)

            entry = agentic.log[-1]
            print(f"\n[Slot {t+1}] REASSIGNMENT")
            print(f"  Event    : {entry['event']}")
            print(f"  Reasoning: {entry['reasoning']}")
            print_assignments(env)

        actions = agent.select_actions(state, info, deterministic=True)

        state, info = env.step(actions)

        pcrlb_log.append(info["pcrlb"])
        per_tgt = info["pcrlb_per_target"]
        pcrlb_sum   += float(sum(per_tgt.values()))   # Σ_k PCRLB_k^t over live targets
        pcrlb_count += len(per_tgt)                    # |K^t|
        for k in range(MAX_TARGETS):
            pcrlb_per_target_log[k].append(per_tgt.get(k, np.nan))
        for i in range(NUM_UAVS):
            energy_log[i].append(E_MAX * (1.0 - info["rho"][i]) / 1000.0)   # consumed kJ
            assignment_log[i].append(info["assignments"].get(i))

        # Tracking RMSE: EKF position estimate vs. ground truth, mean over live targets.
        ekf_means = info["ekf_means"]
        sq_errs = [float(np.sum((ekf_means[k] - env.targets[k].pos) ** 2)) for k in ekf_means]
        rmse_log.append(float(np.sqrt(np.mean(sq_errs))) if sq_errs else 0.0)
        time.sleep(DT)

        prev_traces = {
            k: float(np.trace(state["targets"][k][1][:2, :2]))
            for k in state["targets"]
        }
        for i in range(NUM_UAVS):
            uav_traj[i].append(env.uavs[i].pos2d.copy())
        for k in range(MAX_TARGETS):
            tgt_traj[k].append(env.targets[k].pos.copy() if k in env.targets else _nan_pt())

        # A UAV that depletes its battery deactivates inside env.step; record it with
        # the same red-X marker as a shock failure (both set uav.active = False).
        for i in range(NUM_UAVS):
            if not env.uavs[i].active and i not in failure_log:
                failure_log[i] = len(uav_traj[i]) - 1   # last live (post-move) position
                failure_slots.append(t + 1)
                ev_battery.append((t + 1, i))
                print(f"\n[Slot {t+1}] UAV BATTERY DEPLETED: UAV {i} inactive")

        if (t + 1) % 50 == 0:
            print_slot(t, info, state)
            plot_trajectories(env, uav_traj, tgt_traj, plots_dir, t + 1,
                              failure_log=failure_log)

    plot_eval_pcrlb(pcrlb_log, pcrlb_per_target_log, plots_dir, failure_slots=failure_slots)
    # plot_eval_energy(energy_log, plots_dir, failure_log=failure_log)
    plot_trajectories(env, uav_traj, tgt_traj, plots_dir, T_SLOTS,
                      failure_log=failure_log)
    print_assignments(env)
    avg_pcrlb = pcrlb_sum / max(pcrlb_count, 1)   # Eq. (avg_pcrlb): mean over all target-slots
    print(f"Avg PCRLB (over all target-slots): {avg_pcrlb:.4g} m^2")

    # MA/PA reasoning latency — how long each LLM call took to come back, not how
    # long the simulation waited (the graph runs in a background thread, so the
    # sim never blocks on it). ma_slots/pa_slots give the sim slot of each call
    # (aligned with the latency arrays), so invocation counts can be resolved over
    # the episode without re-parsing llm_log.jsonl. Saved alongside the per-slot
    # logs for this seed.
    ma_lat, pa_lat   = agentic.ma_latencies, agentic.pa_latencies
    ma_slot, pa_slot = agentic.ma_slots, agentic.pa_slots
    print(f"MA reasoning calls: {len(ma_lat)}" +
          (f" | mean {np.mean(ma_lat):.2f}s | total {np.sum(ma_lat):.1f}s" if ma_lat else ""))
    print(f"PA reasoning calls: {len(pa_lat)}" +
          (f" | mean {np.mean(pa_lat):.2f}s | total {np.sum(pa_lat):.1f}s" if pa_lat else ""))
    np.savez(os.path.join(plots_dir, "llm_latency.npz"),
             ma_latencies=np.asarray(ma_lat, dtype=float),
             pa_latencies=np.asarray(pa_lat, dtype=float),
             ma_slots=np.asarray(ma_slot, dtype=int),
             pa_slots=np.asarray(pa_slot, dtype=int))

    print(f"Reassignments: {len(agentic.log)}")
    if agentic.log:
        print("Reassignment log:")
        for entry in agentic.log:
            print(f"  Slot {entry['slot']:3d} | {entry['event'][:100]}")
    print_evolution_summary(
        ev_shock, ev_battery, ev_born, ev_death,
        final_active=sum(1 for u in env.uavs.values() if u.active),
        final_targets=len(env.targets),
    )

    # Per-slot avg PCRLB and total fleet consumed energy (kJ) for mode comparison.
    total_energy = np.sum([energy_log[i] for i in range(NUM_UAVS)], axis=0).tolist()
    # Flattened (slot, target_id[, cause]) event lists — one entry per target,
    # not per slot, so a target born/lost alongside others in the same slot each
    # get their own row.
    birth_events = [(s, k) for s, ids in ev_born for k in ids]
    death_events = [(s, k, cause) for s, k, cause in ev_death]
    # (slot, uav_id, cause) — same shape, and the piece failure_slots alone can't
    # give: WHICH UAV was lost, hence which target lost a tracker. Analysis that
    # attributes a failure to the target it hit needs this (evaluate/recovery_events.py).
    failure_events = ([(s, i, "shock") for s, ids in ev_shock for i in ids] +
                      [(s, i, "battery") for s, i in ev_battery])
    return (pcrlb_log, total_energy, failure_slots, rmse_log, birth_slots,
            pcrlb_per_target_log, assignment_log, birth_events, death_events,
            failure_events)
