# train.py
import argparse
from env import train_env, TrainingConfig
import sys
sys.stdout.reconfigure(encoding='utf-8')

print("▶️  Entered train.py main()")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--growth_type', type=str, required=True)
    parser.add_argument('--lr_schedule_type', type=str, required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()

    print(f"🚀 Training | Seed={args.seed} | Growth={args.growth_type} | LR={args.lr_schedule_type}")

    config = TrainingConfig(
        algorithm="PPO",
        growth_type=args.growth_type,
        growth_factor=3.0,
        stiffness_start=5000,
        stiffness_end=50000,
        num_seeds=1,  # 单个种子
        lr_schedule_type=args.lr_schedule_type,
        lr_start=5e-4,
        lr_end=1e-5
    )

    try:
        train_env(seed_value=args.seed, config=config)
        print(f"✅ Finished training for Seed {args.seed}")
    except Exception as e:
        print(f"❌ Failed training for Seed {args.seed}: {e}")

if __name__ == '__main__':
    main()
