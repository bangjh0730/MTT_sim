import argparse
import numpy as np

import config.params as params
from config.params  import T_SLOTS, T_SLOTS_TRAIN, NUM_UAVS, NUM_TARGETS


def main():
    parser = argparse.ArgumentParser(description="MTT Simulation")
    parser.add_argument(
        "--mode", choices=["train", "marl", "agentic", "compare"],
        default="train",
        help="train: MAPPO training | marl: MARL + greedy re-partition eval | "
             "agentic: MARL + three-tier agentic allocator | "
             "compare: run both on the same scenario and plot them together",
    )
    parser.add_argument("--episodes", type=int, default=None,
                        help="Number of training episodes (train mode only)")
    parser.add_argument("--save", default="./results",
                        help="Directory to save / load model weights and plots")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed for eval modes — fixes spawns, target motion, "
                             "measurement noise and births. RESCUES are "
                             "deliberately NOT fixed: they depend on tracking quality, "
                             "which is what the comparison measures. The LLM tiers are "
                             "also non-deterministic regardless of seed.")
    parser.add_argument("--no-births", action="store_true",
                        help="Disable mid-mission target births. Off by default — with "
                             "targets leaving by rescue, a run without arrivals drains "
                             "to an empty map within ~150 slots.")
    parser.add_argument("--lam", type=float, default=None,
                        help="Override LAMBDA_RESCUE. This is the load-regime knob and "
                             "the experiment's main sweep axis: small values keep the "
                             "system in persistent heavy overload, large ones let it "
                             "repeatedly drain to a manageable state. See the calibration "
                             "notes in config/params.py before going outside "
                             "[0.002, 0.03].")
    parser.add_argument("--p-birth", type=float, default=None,
                        help="Override P_BIRTH. Sweep alongside --lam; the two jointly "
                             "set the load.")
    parser.add_argument("--one-to-one", action="store_true",
                        help="Ablation: cap every assignment set at one target, "
                             "reproducing single-target assignment.")
    args = parser.parse_args()

    # Overrides are applied to the config module before anything imports the env,
    # so a sweep runs without editing the file.
    if args.lam is not None:
        params.LAMBDA_RESCUE = args.lam
        import envs.rescue as _rescue
        _rescue.LAMBDA_RESCUE = args.lam
    if args.p_birth is not None:
        params.P_BIRTH = args.p_birth
        import envs.schedule as _sched
        _sched.P_BIRTH = args.p_birth
    if args.one_to_one:
        params.ONE_TO_MANY = False
        import components.UAV as _uav
        _uav.ONE_TO_MANY = False

    from envs.MTTEnv    import MTTEnv
    from marl           import MAPPO_run, DEFAULT_EPISODES
    from evaluate.utils import save_raw_eval

    births = not args.no_births
    print(f"[Config] lambda={params.LAMBDA_RESCUE}  p_birth={params.P_BIRTH}  "
          f"one_to_many={params.ONE_TO_MANY}  births={births}")

    if args.mode == "train":
        episodes = args.episodes if args.episodes is not None else DEFAULT_EPISODES
        env = MTTEnv(t_slots=T_SLOTS_TRAIN)
        print(f"[MAPPO Training] {NUM_UAVS} UAVs | {NUM_TARGETS} initial targets | "
              f"{T_SLOTS_TRAIN} slots/ep | {episodes} episodes")
        MAPPO_run(env, num_episodes=episodes, save_path=args.save)

    elif args.mode in ("marl", "agentic"):
        if args.seed is not None:
            print(f"[Seed] {args.seed}")
        env = MTTEnv(t_slots=T_SLOTS)
        if args.mode == "marl":
            from evaluate import eval_marl
            res = eval_marl(env, args.save, seed=args.seed, births=births)
        else:
            from evaluate import eval_agentic
            res = eval_agentic(env, args.save, seed=args.seed, births=births)
        raw_path = save_raw_eval(args.seed, args.mode, res)
        print(f"[Raw eval saved] {raw_path}")

    elif args.mode == "compare":
        # Both modes on the SAME scenario: reseeding before each run fixes the
        # spawns, target motion, measurement noise and births, so the only
        # difference is how the partition is revised.
        from evaluate       import eval_marl, eval_agentic, plot_mode_comparison
        from evaluate.utils import seed_plots_dir
        seed = args.seed if args.seed is not None else int(np.random.randint(1_000_000_000))
        print(f"[Compare] seed {seed}")

        env = MTTEnv(t_slots=T_SLOTS)
        print("\n============= MARL + greedy re-partition =============")
        marl_res = eval_marl(env, args.save, seed=seed, births=births)

        env = MTTEnv(t_slots=T_SLOTS)
        print("\n=================== Agentic mode ====================")
        agentic_res = eval_agentic(env, args.save, seed=seed, births=births)

        out_dir = seed_plots_dir(seed)
        plot_mode_comparison({"marl": marl_res, "agentic": agentic_res}, out_dir)

        print("\n" + "=" * 70)
        print(f"{'':10s}{'D-bar (s)':>12s}{'rescued':>10s}{'mean |K|':>10s}"
              f"{'re-plans':>10s}")
        for name, r in (("greedy", marl_res), ("agentic", agentic_res)):
            print(f"{name:10s}{r['avg_rescue_delay']:12.2f}{r['n_rescued']:10d}"
                  f"{np.mean(r['backlog']):10.2f}{r.get('n_replans', 0):10d}")
        print("=" * 70)
        print(f"\nPlots saved to {out_dir}/rescue.png and {out_dir}/delay_distribution.png")

        save_raw_eval(seed, "marl", marl_res)
        save_raw_eval(seed, "agentic", agentic_res)


if __name__ == "__main__":
    main()
