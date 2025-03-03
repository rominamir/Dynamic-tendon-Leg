import os
from datetime import datetime
from env import train_env

def run_multiple_schedulers_in_series():
    """
    Run training with different stiffness growth strategies and RL algorithms in series.

    This function iterates through predefined scheduler configurations, each specifying:
    - growth_type: The type of stiffness progression (e.g., 'exponential', 'linear', 'logarithmic', 'constant').
    - growth_factor: The factor controlling the rate of stiffness change (by default, it is suggested to set exponential at 0.05 and logarithmic at 5, 
    there is no affect on constant and linear).
    - algorithm: The reinforcement learning algorithm to use ('PPO' or 'A2C').

    It sets a fixed random seed for reproducibility and calls `train_env` for each configuration.
    """

    scheduler_configs = [{"growth_type": "exponential", "growth_factor": 5, "algorithm": "A2C"}]
    seed_value = 101  # Set a fixed seed for consistency

    for config in scheduler_configs:
        train_env(
            seed_value=seed_value,
            algorithm=config["algorithm"],
            growth_factor=config["growth_factor"],
            growth_type=config["growth_type"]
        )

if __name__ == '__main__':
    # Execute the function to run training for each scheduler configuration
    run_multiple_schedulers_in_series()
