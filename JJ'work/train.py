import os
from datetime import datetime
from env import train_env


def run_training_with_multiple_seeds():
    """
    Run training for a single growth strategy with 20 different random seeds.

    This function configures:
    - `growth_type`: The type of stiffness progression (e.g., 'exponential', 'linear', 'logarithmic', 'constant').
    - `growth_factor`: The factor controlling the rate of stiffness change (set to 0.05 for exponential, 5 for logarithmic).
    - `algorithm`: The reinforcement learning algorithm to use ('PPO' or 'A2C').
    - `num_seeds`: The number of different seeds for training (default: 20).

    It iterates through 20 different seeds and calls `train_env` for each.
    """

    growth_type = "exponential"  # Change this to "linear", "logarithmic", or "constant" as needed
    growth_factor = 3.0
    algorithm = "PPO"  # Change to "A2C" if needed
    num_seeds = 20  # Run training with 20 different random seeds

    for i in range(num_seeds):
        seed_value = 100 + i  # Generate different seed values

        print(f"Starting training for seed {seed_value} using {growth_type} growth with {algorithm}")

        train_env(
            seed_value=seed_value,
            algorithm=algorithm,
            growth_factor=growth_factor,
            growth_type=growth_type
        )


if __name__ == '__main__':
    # Execute the function to run training with different seeds
    run_training_with_multiple_seeds()
