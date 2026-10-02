import argparse

import numpy as np

import config.params as params
from config.params import T_SLOTS, T_SLOTS_TRAIN, NUM_UAVS, NUM_TARGETS


def main():
    parser = argparse.ArgumentParser(
        description="Multi-UAV multi-target search-and-rescue simulation")
    parser.add_argument("--mode", choices=["train", "eval"], default="train",
                        help="train: MAPPO trajectory training (no LLM in the loop) | "
                             "eval: the full proposal - MAPPO + the agentic allocator",
    )
    parser.add_argument("--episodes", type=int, default=None,
                        help="Number of training episodes (train mode only)")
    parser.add_argument("--save", default="./results",
                        help="Directory to save / load model weights and plots")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed. In train mode it fixes network init, "
                             "action sampling and the environment, so a run is "
                             "reproducible and two configurations can be compared "
                             "at matched initialisation. In eval it fixes spawns, "
                             "target motion, measurement noise and births; RESCUES "
                             "are deliberately not fixed, since they depend on "
                             "tracking quality, which is what the run measures. The "
                             "LLM tiers are non-deterministic regardless of seed.")
    parser.add_argument("--fast", action="store_true",
                        help="Eval only: skip the real-time pacing. The LLM tiers "
                             "then see an unrealistically fast clock, so use this "
                             "for plumbing checks, not for reported results.")
    args = parser.parse_args()

    from envs.MTTEnv    import MTTEnv
    from marl           import MAPPO_run, DEFAULT_EPISODES
    from evaluate.utils import save_raw_eval
    import marl.reward as reward

    # Logged so a run's own output records what produced it. Edit the values in
    # config/params.py and marl/reward.py; they are not command-line flags.
    print(f"[Config] lambda={params.LAMBDA_RESCUE}  p_birth={params.P_BIRTH}  "
          f"w_track={reward.W_TRACK}  one_to_many={params.ONE_TO_MANY}  "
          f"map={params.MAP_SIZE:.0f}  seed={args.seed}")

    if args.mode == "train":
        # Seeding takes three steps: np.random and torch cover network init,
        # action sampling and the measurement noise; seed_motion covers the env's
        # per-target motion and rescue streams; MAPPO_run's own seed covers the
        # partitioner's per-episode draws and the birth schedule.
        if args.seed is not None:
            import torch
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
        episodes = args.episodes if args.episodes is not None else DEFAULT_EPISODES
        env = MTTEnv(t_slots=T_SLOTS_TRAIN)
        if args.seed is not None:
            env.seed_motion(args.seed)
        print(f"[MAPPO Training] {NUM_UAVS} UAVs | {NUM_TARGETS} initial targets | "
              f"{T_SLOTS_TRAIN} slots/ep | {episodes} episodes")
        MAPPO_run(env, num_episodes=episodes, save_path=args.save, seed=args.seed)

    else:
        from evaluate import eval_agentic
        env = MTTEnv(t_slots=T_SLOTS)
        res = eval_agentic(env, args.save, seed=args.seed, births=True,
                           realtime=not args.fast)
        raw_path = save_raw_eval(args.seed, "agentic", res)
        print(f"[Raw eval saved] {raw_path}")


if __name__ == "__main__":
    main()
