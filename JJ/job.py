import subprocess

#growth_types = [f"constant:{k}k" for k in range(5, 51, 5)]
growth_types = ["constant:5k"]

lr_schedule_types = ["constant"]

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
