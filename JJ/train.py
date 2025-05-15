import argparse
from env import train_env, TrainingConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--growth_type', type=str, required=True)
    parser.add_argument('--lr_schedule_type', type=str, required=True)
    parser.add_argument('--seed_start', type=int, default=100)
    parser.add_argument('--seed_end', type=int, default=109)
    args = parser.parse_args()

    for seed in range(args.seed_start, args.seed_end + 1):
        print(f"🚀 Training | Seed={seed} | Growth={args.growth_type} | LR={args.lr_schedule_type}")

        config = TrainingConfig(
            algorithm="PPO",
            growth_type=args.growth_type,
            growth_factor=3.0,
            stiffness_start=30000,
            stiffness_end=40000,
            num_seeds=(args.seed_end - args.seed_start + 1),
            lr_schedule_type=args.lr_schedule_type,
            lr_start=5e-4,
            lr_end=1e-5
        )

        try:
            train_env(seed_value=seed, config=config)
            print(f"✅ Finished training for Seed {seed}")

        except Exception as e:
            print(f"❌ Failed training for Seed {seed}: {e}")

if __name__ == '__main__':
    main()
