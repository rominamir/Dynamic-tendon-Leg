"""Entry‑point script aligned with the streamlined constant‑growth PPO setup.

Usage (example):
    python train.py --growth_type constant_100 --lr 5e-4 --seed_start 100 --seed_end 109 --total_timesteps 100000
    python train.py --growth_type linear --stiffness_start 50 --stiffness_end 200 --seed_start 100 --seed_end 100 --total_timesteps 100000

"""

import argparse
import sys
from datetime import datetime
import traceback
import re
import os

from env import TrainingConfig, train, aggregate_and_save_results

sys.stdout.reconfigure(encoding="utf-8")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=5e-4, help="Constant learning rate for PPO")
    parser.add_argument("--seed_start", type=int, default=100)
    parser.add_argument("--seed_end", type=int, default=109)
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument("--stiffness_start", type=int, default=5_000)
    parser.add_argument("--stiffness_end", type=int, default=50_000)
    parser.add_argument("--max_episode_steps", type=int, default=1_000)
    parser.add_argument("--growth_type", type=str, default="constant_20k", help="Stiffness schedule type (e.g., constant_20k, linear)")
    parser.add_argument("--lr_schedule_type", type=str, default="constant", help="Learning rate schedule type")
    args = parser.parse_args()

    run_date = datetime.now().strftime("%b%d")

    # ----------------------------------------------
    # Parse growth_type: constant_<val> or linear
    # ----------------------------------------------
    if args.growth_type.startswith("constant_"):
        match = re.search(r"constant_(\d+(?:\.\d+)?)(k?)", args.growth_type.lower())
        if match:
            base_val = float(match.group(1))
            is_k = match.group(2) == "k"
            stiffness_val = int(base_val * 1000 if is_k else base_val)
            args.stiffness_start = stiffness_val
            args.stiffness_end = stiffness_val
            print(f"[INFO] Parsed constant stiffness: {stiffness_val} from growth_type '{args.growth_type}'")
        else:
            print(f"[WARN] Could not parse stiffness from growth_type: '{args.growth_type}'")
    else:
        print(f"[INFO] Using dynamic stiffness schedule: {args.growth_type}")
        if args.stiffness_end == args.stiffness_start:
            print(f"[WARN] stiffness_end == stiffness_start for dynamic growth; setting stiffness_end to 50000")
            args.stiffness_end = 50_000

    # ----------------------------------------------
    # Use global range for folder naming consistency
    # ----------------------------------------------
    folder_seed_start = int(os.environ.get("SEED_RANGE_START", args.seed_start))
    folder_seed_end = int(os.environ.get("SEED_RANGE_END", args.seed_end))

    cfg = TrainingConfig(
        stiffness_start=args.stiffness_start,
        stiffness_end=args.stiffness_end,
        num_seeds=args.seed_end - args.seed_start + 1,
        total_timesteps=args.total_timesteps,
        lr=args.lr,
        seed_start=args.seed_start,
        seed_end=args.seed_end,
        run_date=run_date,
        max_episode_steps=args.max_episode_steps,
        growth_type=args.growth_type,
        folder_seed_start=folder_seed_start,
        folder_seed_end=folder_seed_end
    )

    # ----------------------------------------------
    # Run training loop for all seeds
    # ----------------------------------------------
    for seed in range(args.seed_start, args.seed_end + 1):
        print(f"🚀 Training | Seed={seed} | stiffness schedule={args.growth_type} | LR={args.lr:.0e}")
        try:
            train(cfg, seed)
            print(f"✅ Finished training for Seed {seed}")
        except Exception as exc:
            print(f"❌ Failed training for Seed {seed}: {exc}")
            traceback.print_exc()

    print("✅ All seeds complete.")
    aggregate_and_save_results(cfg)

if __name__ == "__main__":
    main()
