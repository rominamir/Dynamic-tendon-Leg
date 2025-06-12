# job.py
import subprocess

growth_types = [f"constant:{k}k" for k in range(5, 51, 5)]
lr_schedule_types = ["constant"]
seeds = range(100, 125)  # 25 seeds

def submit_all_jobs():
    for growth in growth_types:
        for lr in lr_schedule_types:
            for seed in seeds:
                command = ["sbatch", "train.sh", growth, lr, str(seed)]
                print(f"📤 Submitting: {' '.join(command)}")
                try:
                    result = subprocess.run(command, check=True, capture_output=True, text=True)
                    print(f"✅ Submitted job: {growth} {lr} seed={seed} -> {result.stdout.strip()}")
                except subprocess.CalledProcessError as e:
                    print(f"❌ Failed to submit job: {growth} {lr} seed={seed}")
                    print(f"Error: {e.stderr}")

if __name__ == "__main__":
    submit_all_jobs()
