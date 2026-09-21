import os
import shutil
import numpy as np

from config.params      import T_SLOTS, NUM_UAVS, NUM_TARGETS, MAX_TARGETS, E_MAX
from evaluate.plot      import plot_trajectories, plot_eval_pcrlb
from evaluate.utils     import (print_slot, print_assignments, print_assignment_history,
                                seed_plots_dir, print_evolution_summary)

_REASSIGN_INTERVAL = 20   # slots between greedy reassignment checks


def _greedy_reassign(env) -> bool:
    """
    Coverage + orphan reassignment.

    A UAV freed by a target loss (its assignment cleared to None by remove_target)
    is placed onto a target: covering an uncovered target first, else doubling up on
    the nearest. If any target is still uncovered afterwards (e.g. its sole tracker
    failed), the nearest surplus UAV is pulled from a target that keeps >= 1 tracker.
    Every-target coverage is preserved and the change is logged by the caller.

    A full nearest-target re-match is deliberately avoided — two UAVs in a crossing
    geometry would keep swapping targets every interval and neither arrives. Only
    freed or genuinely uncovered UAVs are moved, so a settled swarm is a no-op.

    Returns True if any assignment changed, False otherwise.
    """
    live = list(env.targets.keys())
    if not live:
        return False
    active = [i for i, uav in env.uavs.items() if uav.active]

    tgt_pos = {k: env.ekf_state[k][0][:2] for k in live}
    def _dist(i, k):
        return float(np.linalg.norm(env.uavs[i].pos2d - tgt_pos[k]))

    cover   = {k: [] for k in live}
    orphans = []
    for i in active:
        k = env.uavs[i].assignment
        if k in cover:
            cover[k].append(i)
        else:
            orphans.append(i)   # freed by a target loss (None / removed target)

    changed = False

    # 1. Place freed UAVs: cover an uncovered target first, else double up nearest.
    uncovered = [k for k in live if not cover[k]]
    for i in orphans:
        if uncovered:
            k = min(uncovered, key=lambda k: _dist(i, k))
            uncovered.remove(k)
        else:
            k = min(live, key=lambda k: _dist(i, k))
        env.uavs[i].assignment = k
        cover[k].append(i)
        changed = True

    # 2. Any target still uncovered (no freed UAVs left): pull the nearest surplus.
    for k in [k for k in live if not cover[k]]:
        donors = [i for kc in live if len(cover[kc]) >= 2 for i in cover[kc]]
        if not donors:
            break
        best_i = min(donors, key=lambda i: _dist(i, k))
        kc = env.uavs[best_i].assignment
        env.uavs[best_i].assignment = k
        cover[kc].remove(best_i)
        cover[k].append(best_i)
        changed = True

    return changed


def eval_marl(env, save_path: str, seed: int = None,
              failures: bool = False, births: bool = False):
    from marl              import MAPPO
    from envs.failure      import apply_uav_failures
    from envs.birth_death  import apply_target_dynamics
    from envs.schedule     import DisturbanceSchedule

    agent = MAPPO()
    agent.load(save_path)
    agent.actor.eval()

    print(f"[MAPPO Eval] loaded from {save_path}")

    # Re-seed the global stream (targets, spawns, motion, measurement noise) and build
    # the disturbance schedule from the same seed. The schedule pre-draws all births /
    # deaths / failures so they are identical across marl and agentic, independent of
    # each policy's runtime state.
    if seed is not None:
        np.random.seed(seed)
    schedule = DisturbanceSchedule(seed, env.T)
    env.seed_motion(seed)   # dedicated RNG so target motion matches agentic mode

    state = env.reset()
    info  = None

    assignment_history = {i: [env.uavs[i].assignment] for i in range(NUM_UAVS)}
    print_assignments(env)

    # BS-observable active set carried across slots. A UAV failure (shock or battery)
    # only enters state["active"] once the BS declares it lost — N_FAIL_DETECT silent
    # slots after the physical failure. That later slot, not the failure slot, is when
    # the coverage gap first becomes visible to the greedy pass, so it is when we must
    # trigger reassignment (mirrors evaluate/agentic.py's MA-poll trigger).
    prev_bs_active = {i: True for i in range(NUM_UAVS)}

    def _nan_pt():
        return np.array([np.nan, np.nan])

    uav_traj  = {i: [env.uavs[i].pos2d.copy()] for i in range(NUM_UAVS)}
    # Per-target logs span every id slot [0, MAX_TARGETS); a slot reads NaN whenever
    # that target is not currently alive, so births/deaths break the plotted line.
    tgt_traj  = {k: [env.targets[k].pos.copy() if k in env.targets else _nan_pt()]
                 for k in range(MAX_TARGETS)}
    plots_dir = os.path.join(seed_plots_dir(seed), "marl")
    os.makedirs(plots_dir, exist_ok=True)

    failure_log:   dict = {}   # {uav_id: traj_index} — physical failure slot
    failure_slots: list = []   # slots (1-indexed) at which any UAV failed
    birth_slots:   list = []   # slots (1-indexed) at which a target was actually born
    n_asgn_events: int  = 0
    # Fixed 20-slot clock, same scheduling model as the agentic MA: every check
    # (periodic tick or config-change) restarts the clock from the check slot
    # itself, so a config-change check at t=53 makes the next tick due at t=73.
    next_reassign_slot: int = _REASSIGN_INTERVAL
    # Evolution timeline for the end-of-episode summary.
    ev_shock:   list = []   # (slot, [uav_ids]) shock failures
    ev_battery: list = []   # (slot, uav_id)    battery depletions
    ev_born:    list = []   # (slot, [target_ids])
    ev_death:   list = []   # (slot, target_id, cause)
    pcrlb_log            = []
    rmse_log             = []
    pcrlb_per_target_log = {k: [] for k in range(MAX_TARGETS)}
    energy_log           = {i: [] for i in range(NUM_UAVS)}
    # Which target (or None) each UAV is assigned to, per slot — aligned 1:1 with
    # pcrlb_log/pcrlb_per_target_log so a consumer can tell, at the slot a given
    # target's PCRLB was perturbed, which UAV (if any) was responsible for it.
    assignment_log       = {i: [] for i in range(NUM_UAVS)}
    pcrlb_sum:   float   = 0.0   # Σ_t Σ_k PCRLB_k^t
    pcrlb_count: int     = 0     # Σ_t |K^t|  (target-slot count)

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

        # Target birth/death. Freed UAVs (from a lost target) are reassigned below by
        # the greedy pass — which now triggers and is logged, rather than silently.
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

        # Rebuild the observation to reflect the current target set / assignments
        # (a birth adds a target the previous snapshot did not contain), so the
        # BS-observable active set below reflects this slot, not a stale one.
        state = env._system_state()

        # A UAV loss becomes actionable only when the BS observes it: state["active"]
        # flips the UAV inactive N_FAIL_DETECT slots after the physical failure.
        # Trigger reassignment on that transition, not the physical-failure slot —
        # reacting immediately is a lookahead the BS doesn't actually have.
        bs_active  = state["active"]
        newly_lost = [i for i in range(NUM_UAVS)
                      if prev_bs_active.get(i, True) and not bs_active.get(i, True)]
        prev_bs_active = dict(bs_active)
        if newly_lost:
            print(f"\n[Slot {t+1}] BS DETECTED UAV LOSS: UAVs {newly_lost} — reassignment check")

        # Coverage / orphan reassignment: fixed 20-slot tick, or immediately when
        # the BS detects a UAV loss / a target is born or dies. Either way the
        # 20-slot clock restarts from this check, not from t=0's multiples.
        config_changed = bool(newly_lost) or bool(born) or bool(removed)
        due = t >= next_reassign_slot
        if due or config_changed:
            next_reassign_slot = t + _REASSIGN_INTERVAL
            if _greedy_reassign(env):
                for i in range(NUM_UAVS):
                    assignment_history[i].append(env.uavs[i].assignment)
                n_asgn_events += 1
                print(f"\n[Slot {t+1}] GREEDY REASSIGNMENT (assignments changed)")
                print_assignment_history(assignment_history)
                # Assignments changed — refresh the snapshot so the actor sees them.
                state = env._system_state()

        actions     = agent.select_actions(state, info, deterministic=True)
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

    # Extra copy for cherry-picking the best run across seeds: alongside the
    # normal plots/seed{N}/marl/eval_pcrlb.png, keep a flat pcrlb/<mode>_seed{N}.png
    # tagged by disturbance mode (birth / fail / birth_fail) so runs don't collide
    # (does not replace or move the original save).
    os.makedirs("plots/pcrlb", exist_ok=True)
    mode_tag = "_".join(m for m, on in (("birth", births), ("fail", failures)) if on) or "none"
    seed_str = f"seed{seed}" if seed is not None else "seed_unspecified"
    shutil.copy2(os.path.join(plots_dir, "eval_pcrlb.png"),
                 os.path.join("plots/pcrlb", f"{mode_tag}_{seed_str}.png"))

    avg_pcrlb = pcrlb_sum / max(pcrlb_count, 1)   # Eq. (avg_pcrlb): mean over all target-slots
    print(f"Greedy reassignment events (assignments changed): {n_asgn_events}")
    print(f"Avg PCRLB (over all target-slots): {avg_pcrlb:.4g} m^2")
    print(f"Eval PCRLB / energy plots saved to {plots_dir}/")
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
