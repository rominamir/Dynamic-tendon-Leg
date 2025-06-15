"""Entry‑point script aligned with the streamlined constant‑growth PPO setup.

Usage (example):
    python train.py --lr 5e-4 --seed_start 100 --seed_end 109 --total_timesteps 100000
"""

import argparse
import sys

# Import your environment/training utilities.
# Make sure the module name matches the cleaned file you saved earlier.
from env import TrainingConfig, train, aggregate_and_save_results # Rename if your file/module differs.

sys.stdout.reconfigure(encoding="utf-8")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=5e-4, help="Constant learning rate for PPO")
    parser.add_argument("--seed_start", type=int, default=100)
    parser.add_argument("--seed_end", type=int, default=109)
    parser.add_argument("--total_timesteps", type=int, default=100_000)
    parser.add_argument("--stiffness_start", type=int, default=5_000)
    parser.add_argument("--stiffness_end", type=int, default=50_000)
    args = parser.parse_args()

    # Shared configuration (one object reused across seeds)
    cfg = TrainingConfig(
        stiffness_start=args.stiffness_start,
        stiffness_end=args.stiffness_end,
        num_seeds=args.seed_end - args.seed_start + 1,
        total_timesteps=args.total_timesteps,
        lr=args.lr,
        seed_start=args.seed_start,
        seed_end=args.seed_end
    )

    for seed in range(args.seed_start, args.seed_end + 1):
        print(f"🚀 Training | Seed={seed} | constant stiffness | LR={args.lr:.0e}")
        try:
            train(cfg, seed)
            print(f"✅ Finished training for Seed {seed}")
        except Exception as exc:
            print(f"❌ Failed training for Seed {seed}: {exc}")

    print("✅ All seeds complete.")
    aggregate_and_save_results(cfg)


if __name__ == "__main__":
    main()
