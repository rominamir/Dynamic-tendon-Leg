"""Batch-submit SLURM jobs for PPO experiments with different stiffness schedules.

Assumes `train.sh` accepts positional args in this order:
    1) learning_rate (float)
    2) stiffness_start (int or 'Nk' like '5k')
    3) stiffness_end   (int or 'Nk')
    4) seed_start (int)
    5) seed_end   (int)
    6) total_timesteps (int)
    7) growth_type (optional: constant|constant_<val>k|linear|expo|log|sigmoid)  [default: linear]
    8) curve_param (optional: float; used when growth_type in {'log','sigmoid'}) [default: 9.0]

Examples this script can construct:
    # Constant schedules over a grid of values (5k, 10k, ..., 50k)
    sbatch train.sh 5e-4 5000 5000 100 124 1000000 constant

    # Dynamic schedules:
    sbatch train.sh 5e-4 5000 50000 100 124 1000000 linear
    sbatch train.sh 5e-4 5000 50000 100 124 1000000 expo
    sbatch train.sh 5e-4 5000 50000 100 124 1000000 log 9.0
    sbatch train.sh 5e-4 5000 50000 100 124 1000000 sigmoid 10.0
"""

from __future__ import annotations
import subprocess
from pathlib import Path
from typing import Iterable, Tuple, Union

# -----------------------------------------------------------------------------
# Global knobs (tweak as needed)
# -----------------------------------------------------------------------------
LR: float = 5e-4
SEED_START: int = 100
SEED_END: int = 124
TOTAL_TIMESTEPS: int = 1_000_000

# Choose one of: "constant", "linear", "expo", "log", "sigmoid"
GROWTH_TYPE: str = "constant"

# For 'log' and 'sigmoid': you can set a single value OR a list for sweeping.
CURVE_PARAM: float = 9.0
CURVE_PARAMS: list[float] | None = None  # e.g., [5.0, 9.0, 12.0]; None -> use CURVE_PARAM

# For constant schedules: try multiple constant stiffness values (5k → 50k)
# Accepts ints (e.g., 5000) or strings (e.g., "5k")
STIFFNESS_LEVELS: list[Union[int, str]] = [k * 1000 for k in range(5, 51, 5)]

# For dynamic schedules: define (start, end) pairs you want to sweep
# Accepts ints or 'Nk' strings; pairs must be 2-tuples
DYNAMIC_PAIRS: list[Tuple[Union[int, str], Union[int, str]]] = [
    (5_000, 50_000),
    # Add more pairs if desired, e.g. ("10k", "40k"),
]

# Optional: sbatch extra flags (partition, qos, etc.)
SBATCH_FLAGS: list[str] = []  # e.g., ["-p", "gpu", "--qos", "normal"]

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _ensure_script() -> Path:
    root = Path(__file__).resolve().parent
    script = root / "train.sh"
    if not script.exists():
        raise FileNotFoundError(f"train.sh not found at {script}")
    return script

def _to_str(x: Union[int, float, str]) -> str:
    """Convert int/float/str to str without altering content (keeps '5k' as-is)."""
    return str(x)

def _curve_iter() -> Iterable[float]:
    """Yield curve_param values to try (for log/sigmoid)."""
    if CURVE_PARAMS is not None and len(CURVE_PARAMS) > 0:
        return CURVE_PARAMS
    return [CURVE_PARAM]

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
# Main submission logic
# -----------------------------------------------------------------------------
def submit_all_jobs() -> None:
    script = _ensure_script()

    # Assemble base sbatch invocation (allow optional flags)
    def base_cmd() -> list[str]:
        return ["sbatch", *SBATCH_FLAGS, str(script)]

    if GROWTH_TYPE == "constant":
        # Submit a job per constant stiffness level
        for stiffness in STIFFNESS_LEVELS:
            cmd = base_cmd() + [
                _to_str(LR),
                _to_str(stiffness),  # stiffness_start
                _to_str(stiffness),  # stiffness_end (same → constant)
                _to_str(SEED_START),
                _to_str(SEED_END),
                _to_str(TOTAL_TIMESTEPS),
                "constant",          # 7) growth_type
                # no 8th arg for constant
            ]
            _run(cmd, tag=f"const@{stiffness}")
        return

    # Dynamic schedules: linear | expo | log | sigmoid
    if GROWTH_TYPE not in {"linear", "expo", "log", "sigmoid"}:
        raise ValueError(f"Unsupported GROWTH_TYPE: {GROWTH_TYPE}")

    for start, end in DYNAMIC_PAIRS:
        if GROWTH_TYPE in {"log", "sigmoid"}:
            # Sweep curve_param(s)
            for cp in _curve_iter():
                cmd = base_cmd() + [
                    _to_str(LR),
                    _to_str(start),
                    _to_str(end),
                    _to_str(SEED_START),
                    _to_str(SEED_END),
                    _to_str(TOTAL_TIMESTEPS),
                    GROWTH_TYPE,       # 7) growth_type
                    _to_str(cp),       # 8) curve_param
                ]
                _run(cmd, tag=f"{GROWTH_TYPE}@{start}-{end}|cp={cp}")
        else:
            # linear / expo (no curve_param)
            cmd = base_cmd() + [
                _to_str(LR),
                _to_str(start),
                _to_str(end),
                _to_str(SEED_START),
                _to_str(SEED_END),
                _to_str(TOTAL_TIMESTEPS),
                GROWTH_TYPE,         # 7) growth_type
                # no 8th arg
            ]
            _run(cmd, tag=f"{GROWTH_TYPE}@{start}-{end}")

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    submit_all_jobs()
