import os
import time
from datetime import datetime
from env import train_env

def run_training_with_multiple_schedulers_and_seeds():
    """
    Run training for multiple stiffness growth strategies and multiple seeds.
    """
    growth_types = [
        "constant:20k"
    ]
    growth_factor = 3.0  # You can adjust or ignore this for constant types
    algorithm = "PPO"
    num_seeds = 10

    for growth_type in growth_types:
        print(f"\n🌱 Starting training for growth_type = {growth_type}")

        for i in range(num_seeds):
            seed_value = 100 + i
            start_time = time.time()

            print(f"\n🔹 Training run {i+1}/{num_seeds} | Seed: {seed_value} | Growth Type: {growth_type}")

            try:
                train_env(
                    seed_value=seed_value,
                    algorithm=algorithm,
                    growth_factor=growth_factor,
                    growth_type=growth_type
                )
                elapsed_time = time.time() - start_time
                print(f"✅ Training run completed in {elapsed_time:.2f} seconds.")

            except Exception as e:
                print(f"❌ Training failed due to error: {str(e)}")

    print("\n🎉 All training for all growth strategies completed!")


if __name__ == '__main__':
    run_training_with_multiple_schedulers_and_seeds()
