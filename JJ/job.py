import subprocess

growth_types = ["linear", "curriculum_linear", "constant:30k", "constant:40k"]
lr_schedule_types = ["constant", "linear"]

def submit_all_jobs():
    for growth in growth_types:
        for lr in lr_schedule_types:
            command = ["sbatch", "train.sh", growth, lr]
            print(f"📤 Submitting: {' '.join(command)}")
            try:
                result = subprocess.run(command, check=True, capture_output=True, text=True)
                print(f"✅ Submitted job for {growth}, {lr}: {result.stdout.strip()}")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to submit job for {growth}, {lr}")
                print(f"Error: {e.stderr}")

if __name__ == "__main__":
    submit_all_jobs()
