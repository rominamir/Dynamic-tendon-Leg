import subprocess

# ✅ Full path to your train.py file
train_script = r"C:\Users\User\Desktop\New folder\AimIV\Dynamic-tendon-Leg\Dynamic-tendon-Leg\JJ\train.py"

growth_types = ["linear", "curriculum_linear", "constant:30k", "constant:40k"]
lr_schedule_types = ["constant", "linear"]

def run_all_locally():
    for growth in growth_types:
        for lr in lr_schedule_types:
            command = [
                "python", train_script,
                "--growth_type", growth,
                "--lr_schedule_type", lr
            ]
            print(f"\n🔧 Running: {' '.join(command)}\n" + "-"*60)

            # ✅ Stream output live to terminal
            process = subprocess.Popen(command, shell=False)
            process.wait()

            if process.returncode == 0:
                print(f"✅ Finished run for {growth} | {lr}")
            else:
                print(f"❌ Failed run for {growth} | {lr} with return code {process.returncode}")

if __name__ == "__main__":
    run_all_locally()
