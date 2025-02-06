import os
from datetime import datetime
from env import train_env

def run_multiple_schedulers_in_series():
    """
    Run the training multiple times in series,
    each with a distinct combination of growth_type and growth_factor.
    """

    # Example configurations:
    # 1) exponential with growth_factor=0.05
    # 2) linear with growth_factor=5
    # 3) logarithmic with growth_factor=5
    # 4) constant with growth_factor=5
    # scheduler_configs = [
    #     {"growth_type": "exponential", "growth_factor": 0.05},
    #     {"growth_type": "logarithmic", "growth_factor": 5},
    #     {"growth_type": "linear", "growth_factor": 5},
    #     {"growth_type": "constant", "growth_factor": 5}
    # ]
    scheduler_configs = [{"growth_type": "exponential", "growth_factor": 5}]
    seed_value = 101  # You can change or vary this if desired

    for config in scheduler_configs:
        g_type = config["growth_type"]
        g_factor = config["growth_factor"]
        print(f"Running training with growth_type={g_type}, growth_factor={g_factor}")
        # Call your environment training function
        train_env(
            seed_value=seed_value,
            train=True,
            growth_factor=g_factor,
            growth_type=g_type
        )
        print(f"Finished training for {g_type} with factor={g_factor}\n")


if __name__ == '__main__':
    # This will sequentially run train_env for each config
    run_multiple_schedulers_in_series()
