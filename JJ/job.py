"""
Batch-submit SLURM jobs for PPO experiments with different stiffness schedules.

Assumes `train.sh` accepts positional args in this order:
    1) learning_rate (float)
    2) stiffness_start (int or 'Nk' like '5k')
    3) stiffness_end   (int or 'Nk')
    4) seed_start (int)
    5) seed_end   (int)
    6) total_timesteps (int)
    7) growth_type (optional: constant|constant_<val>k|linear|expo|log)  [default: linear]
    8) curve_param (optional: float; only used when growth_type='log')   [default: 9.0]

Examples this script can construct:
    # Constant schedules over a grid of values (5k, 10k, ..., 50k)
    sbatch train.sh 5e-4 5000 5000 100 124 1000000 constant

    # Dynamic schedules:
    sbatch train.sh 5e-4 5000 50000 100 124 1000000 linear
    sbatch train.sh 5e-4 5000 50000 100 124 1000000 expo
    sbatch train.sh 5e-4 5000 50000 100 124 1000000 log 9.0
"""

from __future__ import annotations
import subprocess
from pathlib import Path

# -----------------------------------------------------------------------------
# Global knobs (tweak as needed)
# -----------------------------------------------------------------------------
LR = 5e-4
SEED_START = 100
SEED_END = 124
TOTAL_TIMESTEPS = 1_000_000

# Choose one of: "constant", "linear", "expo", "log"
GROWTH_TYPE = "constant"

# Used only when GROWTH_TYPE == "log"
CURVE_PARAM = 9.0

# For constant schedules: try multiple constant stiffness values (5k → 50k)
STIFFNESS_LEVELS = [k * 1000 for k in range(5, 51, 5)]

# For dynamic schedules: define (start, end) pairs you want to sweep
DYNAMIC_PAIRS = [
    (5_000, 50_000),
    # add more pairs if desired, e.g. (10_000, 40_000),
]

# -----------------------------------------------------------------------------
# Submission helper
# -----------------------------------------------------------------------------
def submit_all_jobs() -> None:
    root = Path(__file__).resolve().parent
    script = root / "train.sh"
    if not script.exists():
        raise FileNotFoundError(f"train.sh not found at {script}")

    if GROWTH_TYPE == "constant":
        # Submit a job per constant stiffness level
        for stiffness in STIFFNESS_LEVELS:
            cmd = [
                "sbatch",
                str(script),
                str(LR),
                str(stiffness),    # stiffness_start
                str(stiffness),    # stiffness_end (same → constant)
                str(SEED_START),
                str(SEED_END),
                str(TOTAL_TIMESTEPS),
                "constant",        # 7) growth_type
                # no 8th arg for constant
            ]
            _run(cmd, tag=f"const@{stiffness}")
    else:
        # Submit for each (start, end) pair using the chosen dynamic schedule
        for start, end in DYNAMIC_PAIRS:
            cmd = [
                "sbatch",
                str(script),
                str(LR),
                str(start),
                str(end),
                str(SEED_START),
                str(SEED_END),
                str(TOTAL_TIMESTEPS),
                GROWTH_TYPE,       # 7) growth_type: linear|expo|log
            ]
            # Only pass curve_param for 'log'
            if GROWTH_TYPE == "log":
                cmd.append(str(CURVE_PARAM))  # 8) curve_param
            _run(cmd, tag=f"{GROWTH_TYPE}@{start}-{end}")

def _run(cmd: list[str], tag: str) -> None:
    pretty = " ".join(cmd)
    print(f"📤 Submitting: {pretty}")
    try:
        res = subprocess.run(cmd, check=True, capture_output=True, text=True)
        out = res.stdout.strip() or "(no stdout)"
        print(f"✅ Submitted ({tag}): {out}")
    except subprocess.CalledProcessError as exc:
        err = (exc.stderr or "").strip()
        print(f"❌ Submission failed ({tag}): {err}")

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    submit_all_jobs()
