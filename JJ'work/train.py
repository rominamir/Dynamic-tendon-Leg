import os
import time
from datetime import datetime
from env import train_env

def run_training_with_multiple_seeds():
    """
    Run training for a single growth strategy with 20 different random seeds.

    This function configures:
    - `growth_type`: The type of stiffness progression (e.g., 'exponential', 'linear', 'logarithmic', 'constant').
    - `growth_factor`: Fixed at 3.0 for all types.
    - `algorithm`: The reinforcement learning algorithm to use ('PPO' or 'A2C').
    - `num_seeds`: The number of different seeds for training (default: 20).

    It iterates through 20 different seeds and calls `train_env` for each.
    """

    growth_type = "exponential"  # Options: "linear", "logarithmic", "constant"
    growth_factor = 3.0  # Keep it fixed as you prefer
    algorithm = "PPO"  # Options: "PPO", "A2C"
    num_seeds = 20  # Run training with 20 different random seeds

    print(f"\n🚀 Starting {num_seeds} training runs with {growth_type} growth using {algorithm} (growth_factor = {growth_factor})")

    for i in range(num_seeds):
        seed_value = 100 + i  # Generate different seed values
        start_time = time.time()  # Track execution time

        print(f"\n🔹 Training run {i+1}/{num_seeds} | Seed: {seed_value} | Growth Type: {growth_type} | Growth Factor: {growth_factor}")

        try:
            train_env(
                seed_value=seed_value,
                algorithm=algorithm,
                growth_factor=growth_factor,
                growth_type=growth_type
            )
            elapsed_time = time.time() - start_time
            print(f"✅ Training run {i+1} completed in {elapsed_time:.2f} seconds.")

        except Exception as e:
            print(f"❌ Training run {i+1} failed due to error: {str(e)}")

    print("\n🎉 All training runs completed!")


if __name__ == '__main__':
    # Execute the function to run training with different seeds
    run_training_with_multiple_seeds()
