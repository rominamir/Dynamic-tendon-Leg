import subprocess

# You can modify this list to include other constant stiffness levels
growth_types = [f"constant:{k}k" for k in range(5, 51, 5)]
# growth_types = ["constant:30k"]  # ← Use this for testing single growth type

lr_schedule_types = ["constant"]

# Fixed test setup
seed_start = 100
seed_end = 124
total_timesteps = 1000000  # Small value for fast test

def submit_all_jobs():
    for growth in growth_types:
        for lr in lr_schedule_types:
            command = [
                "sbatch", "train.sh",
                growth,
                lr,
                str(seed_start),
                str(seed_end),
                str(total_timesteps)
            ]
            print(f"📤 Submitting: {' '.join(command)}")
            try:
                result = subprocess.run(command, check=True, capture_output=True, text=True)
                print(f"✅ Submitted job for {growth}, {lr}: {result.stdout.strip()}")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to submit job for {growth}, {lr}")
                print(f"Error: {e.stderr.strip()}")

if __name__ == "__main__":
    submit_all_jobs()
