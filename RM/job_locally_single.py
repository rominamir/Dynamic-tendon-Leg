import subprocess
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

growth_type =  "constant:25k"           # e.g., "curriculum_linear", "constant:30k"
lr_schedule_type = "constant"      # e.g., "constant" or "linear"
job_tag = "parallel_test2"          # tag output folders
train_script = r"C:\Users\User\Desktop\New folder\AimIV\Dynamic-tendon-Leg\Dynamic-tendon-Leg\JJ\train.py"
max_workers = 12                    # number of parallel jobs (adjust based on your CPU)
seed_range = range(100, 101)       # 10 seeds

def run_seed(seed):
    env = os.environ.copy()
    env["JOB_TAG"] = job_tag

    command = [
        "python", train_script,
        "--growth_type", growth_type,
        "--lr_schedule_type", lr_schedule_type,
        "--seed_start", str(seed),
        "--seed_end", str(seed)
    ]

    start_time = time.time()
    print(f"🚀 Starting Seed {seed} | {growth_type} | {lr_schedule_type}")
    try:
        #process = subprocess.run(command, env=env, shell=False, capture_output=True, text=True)
        process = subprocess.run(command, env=env, shell=False, capture_output=True, text=True, encoding='utf-8', errors='replace')

        duration = time.time() - start_time
        if process.returncode == 0:
            print(f"✅ Done Seed {seed} in {duration:.1f}s")
        else:
            print(f"❌ Failed Seed {seed} | Error:\n{process.stderr}")
    except Exception as e:
        print(f"🔥 Exception running Seed {seed}: {e}")

def main():
    print(f"🌱 Running seeds {seed_range.start}–{seed_range.stop - 1} in parallel")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_seed, seed) for seed in seed_range]
        for future in as_completed(futures):
            future.result()

if __name__ == "__main__":
    main()
