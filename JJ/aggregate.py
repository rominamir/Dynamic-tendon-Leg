# run this after each job is done 
from env import TrainingConfig, aggregate_and_save_results

growth_types = [f"constant:{k}k" for k in range(5, 51, 5)]

lr_schedule_types = ["constant"]  # or ["constant", "linear"]

seed_start = 100
seed_end = 124
num_seeds = seed_end - seed_start + 1

def run_aggregation():
    for growth in growth_types:
        for lr_schedule in lr_schedule_types:
            print(f"📊 Aggregating for Growth={growth}, LR={lr_schedule}")
            config = TrainingConfig(
                algorithm="PPO",
                growth_type=growth,
                growth_factor=3.0, 
                stiffness_start=5000,
                stiffness_end=50000,
                num_seeds=num_seeds,
                lr_schedule_type=lr_schedule,
                lr_start=5e-4,
                lr_end=1e-5
            )
            aggregate_and_save_results(config)

if __name__ == "__main__":
    run_aggregation()
