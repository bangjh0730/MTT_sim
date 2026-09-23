import argparse
import numpy as np

import config.params as params
from config.params import T_SLOTS, T_SLOTS_TRAIN, NUM_UAVS, NUM_TARGETS


def main():
    parser = argparse.ArgumentParser(
        description="Multi-UAV multi-target search-and-rescue simulation")
    parser.add_argument(
        "--mode", choices=["train", "eval"], default="train",
        help="train: MAPPO trajectory training (no LLM in the loop) | "
             "eval: the full proposal — MAPPO + the agentic allocator",
    )
    parser.add_argument("--episodes", type=int, default=None,
                        help="Number of training episodes (train mode only)")
    parser.add_argument("--save", default="./results",
                        help="Directory to save / load model weights and plots")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed for eval — fixes spawns, target motion, "
                             "measurement noise and births. RESCUES are "
                             "deliberately NOT fixed: they depend on tracking "
                             "quality, which is what the run measures. The LLM "
                             "tiers are non-deterministic regardless of seed.")
    parser.add_argument("--no-births", action="store_true",
                        help="Disable mid-mission target births. Off by default — "
                             "with targets leaving by rescue, a run without "
                             "arrivals drains to an empty map and measures nothing.")
    parser.add_argument("--lam", type=float, default=None,
                        help="Override LAMBDA_RESCUE. The load-regime knob and the "
                             "experiment's main sweep axis: small values hold the "
                             "system in heavy overload, large ones let it drain to "
                             "a manageable state. See the calibration notes in "
                             "config/params.py before going outside [0.002, 0.03].")
    parser.add_argument("--p-birth", type=float, default=None,
                        help="Override P_BIRTH. Sweep alongside --lam; the two "
                             "jointly set the load.")
    parser.add_argument("--one-to-one", action="store_true",
                        help="Ablation: cap every assignment set at one target, "
                             "reproducing single-target assignment.")
    parser.add_argument("--fast", action="store_true",
                        help="Eval only: skip the real-time pacing. The LLM tiers "
                             "then see an unrealistically fast clock, so use this "
                             "for plumbing checks, not for reported results.")
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

    else:
        if args.seed is not None:
            print(f"[Seed] {args.seed}")
        from evaluate import eval_agentic
        env = MTTEnv(t_slots=T_SLOTS)
        res = eval_agentic(env, args.save, seed=args.seed, births=births,
                           realtime=not args.fast)
        raw_path = save_raw_eval(args.seed, "agentic", res)
        print(f"[Raw eval saved] {raw_path}")


if __name__ == "__main__":
    main()
