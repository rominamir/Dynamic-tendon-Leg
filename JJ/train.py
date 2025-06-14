import argparse
from env import train_env, TrainingConfig, aggregate_and_save_results
import sys
sys.stdout.reconfigure(encoding='utf-8')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--growth_type', type=str, required=True)
    parser.add_argument('--lr_schedule_type', type=str, required=True)
    parser.add_argument('--seed_start', type=int, default=100)
    parser.add_argument('--seed_end', type=int, default=109)
    parser.add_argument('--total_timesteps', type=int, default=100000)
    args = parser.parse_args()

    # Create shared config once
    config = TrainingConfig(
        algorithm="PPO",
        growth_type=args.growth_type,
        growth_factor=3.0,
        stiffness_start=5000,
        stiffness_end=50000,
        num_seeds=(args.seed_end - args.seed_start + 1),
        lr_schedule_type=args.lr_schedule_type,
        lr_start=5e-4,
        lr_end=1e-5,
        total_timesteps=args.total_timesteps
    )

    for seed in range(args.seed_start, args.seed_end + 1):
        print(f"🚀 Training | Seed={seed} | Growth={args.growth_type} | LR={args.lr_schedule_type}")

        try:
            train_env(seed_value=seed, config=config)
            print(f"✅ Finished training for Seed {seed}")

        except Exception as e:
            print(f"❌ Failed training for Seed {seed}: {e}")

    # Perform aggregation after all seeds are done
    print("📊 Aggregating results from all seeds...")
    aggregate_and_save_results(config)
    print("✅ Aggregation complete.")

if __name__ == '__main__':
    main()
