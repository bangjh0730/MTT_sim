import argparse
import numpy as np

from envs.MTTEnv    import MTTEnv
from marl           import MAPPO_run, DEFAULT_EPISODES
from config.params  import T_SLOTS, T_SLOTS_TRAIN, NUM_UAVS, NUM_TARGETS
from evaluate.utils import save_raw_eval


def main():
    parser = argparse.ArgumentParser(description="MTT Simulation")
    parser.add_argument(
        "--mode", choices=["train", "marl", "agentic", "compare"],
        default="train",
        help="train: MAPPO training | marl: MARL eval | agentic: MARL + Agentic AI eval "
             "| compare: run marl then agentic on the same scenario and plot both together",
    )
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES,
                        help="Number of training episodes (train mode only)")
    parser.add_argument("--save", default="./results",
                        help="Directory to save / load model weights and plots")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed for eval modes — fixes spawn, target motion, "
                             "measurement noise, and (via the dedicated failure RNG) "
                             "UAV failures. The LLM in agentic mode is still "
                             "non-deterministic regardless of seed.")
    parser.add_argument("--failures", action="store_true",
                        help="Enable the UAV failure model in eval (off by default).")
    parser.add_argument("--births", action="store_true",
                        help="Enable the target birth/death model in eval (off by default).")
    args = parser.parse_args()

    if args.mode == "train":
        env = MTTEnv(t_slots=T_SLOTS_TRAIN)
        print(f"[MAPPO Training] {NUM_UAVS} UAVs | {NUM_TARGETS} targets | "
              f"{T_SLOTS_TRAIN} slots/ep | {args.episodes} episodes")
        MAPPO_run(env, num_episodes=args.episodes, save_path=args.save)

    elif args.mode == "marl":
        if args.seed is not None:
            print(f"[Seed] {args.seed}")
        env = MTTEnv(t_slots=T_SLOTS)
        from evaluate import eval_marl
        res = eval_marl(env, args.save, seed=args.seed,
                        failures=args.failures, births=args.births)
        raw_path = save_raw_eval(args.seed, "marl", *res)
        print(f"[Raw eval saved] {raw_path}")

    elif args.mode == "agentic":
        if args.seed is not None:
            print(f"[Seed] {args.seed}")
        env = MTTEnv(t_slots=T_SLOTS)
        from evaluate import eval_agentic
        res = eval_agentic(env, args.save, seed=args.seed,
                           failures=args.failures, births=args.births)
        raw_path = save_raw_eval(args.seed, "agentic", *res)
        print(f"[Raw eval saved] {raw_path}")

    elif args.mode == "compare":
        # Run both modes on the SAME scenario: reseeding before each run fixes the
        # spawns, target motion, measurement noise, and failures identically, so the
        # only difference is the reassignment policy. Then plot the two together.
        from evaluate       import eval_marl, eval_agentic, plot_mode_comparison
        from evaluate.utils import seed_plots_dir
        seed = args.seed if args.seed is not None else int(np.random.randint(1_000_000_000))
        print(f"[Compare] seed {seed}")

        env = MTTEnv(t_slots=T_SLOTS)
        print("\n===================== MARL mode =====================")
        marl_res = eval_marl(env, args.save, seed=seed,
                             failures=args.failures, births=args.births)

        env = MTTEnv(t_slots=T_SLOTS)
        print("\n=================== Agentic mode ====================")
        agentic_res = eval_agentic(env, args.save, seed=seed,
                                   failures=args.failures, births=args.births)

        out_dir = seed_plots_dir(seed)
        plot_mode_comparison({"marl": marl_res, "agentic": agentic_res}, out_dir)
        print(f"\nPlots saved to {out_dir}/pcrlb.png and {out_dir}/energy.png")

        save_raw_eval(seed, "marl", *marl_res)
        save_raw_eval(seed, "agentic", *agentic_res)


if __name__ == "__main__":
    main()
