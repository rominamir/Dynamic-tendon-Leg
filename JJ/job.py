"""Batch‑submit SLURM jobs for constant‑stiffness PPO experiments.

Assumes `train.sh` accepts arguments in this order:
    1) learning‑rate (float)
    2) stiffness_start (int)
    3) stiffness_end   (int)
    4) seed_start (int)
    5) seed_end   (int)
    6) total_timesteps (int)

Example call constructed here:
    sbatch train.sh 5e-4 5000 5000 100 124 1000000
"""

import subprocess
from pathlib import Path

# -----------------------------------------------------------------------------
# Experiment grid (feel free to tweak)
# -----------------------------------------------------------------------------

STIFFNESS_LEVELS = [k * 1000 for k in range(5, 51, 5)]  # 5k → 50k, step 5k
LR = 5e-4  # constant learning rate for all runs
SEED_START = 100
SEED_END = 124
TOTAL_TIMESTEPS = 1_000_000  # keep small for quick tests

# -----------------------------------------------------------------------------
# Submission helper
# -----------------------------------------------------------------------------

def submit_all_jobs() -> None:
    root = Path(__file__).resolve().parent
    script = root / "train.sh"

    if not script.exists():
        raise FileNotFoundError(f"train.sh not found at {script}")

    for stiffness in STIFFNESS_LEVELS:
        cmd = [
            "sbatch",
            str(script),
            str(LR),
            str(stiffness),  # stiffness_start
            str(stiffness),  # stiffness_end (same → constant)
            str(SEED_START),
            str(SEED_END),
            str(TOTAL_TIMESTEPS),
        ]

        print(f"📤 Submitting: {' '.join(cmd)}")
        try:
            res = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✅ Submitted job @ {stiffness:5d}: {res.stdout.strip()}")
        except subprocess.CalledProcessError as exc:
            print(f"❌ Submission failed @ {stiffness:5d}: {exc.stderr.strip()}")


if __name__ == "__main__":
    submit_all_jobs()
