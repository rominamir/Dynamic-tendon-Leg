"""Entry-point script aligned with the streamlined PPO setup.

Examples:
    # Constant stiffness via suffix (supports k/K):
    python train.py --growth_type constant_20k --lr 5e-4 --seed_start 100 --seed_end 109 --total_timesteps 100000

    # Constant stiffness via flags (no suffix in growth_type):
    python train.py --growth_type constant --stiffness_start 20000 --seed_start 100 --seed_end 100 --total_timesteps 100000

    # Linear growth:
    python train.py --growth_type linear --stiffness_start 5k --stiffness_end 50k --seed_start 100 --seed_end 100 --total_timesteps 100000

    # Exponential (geometric) growth:
    python train.py --growth_type expo --stiffness_start 5k --stiffness_end 50k --seed_start 100 --seed_end 100

    # Logarithmic growth (front-loaded; curve_param controls early acceleration):
    python train.py --growth_type log --stiffness_start 5k --stiffness_end 50k --curve_param 9.0 --seed_start 100 --seed_end 100
"""

import argparse
import sys
from datetime import datetime
import traceback
import re
import os

from env import TrainingConfig, train, aggregate_and_save_results

# Ensure UTF-8 printing on Windows terminals
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _parse_int_with_k(v: str) -> int:
    """
    Parse integers that may have a 'k' suffix (case-insensitive), e.g., '20k' -> 20000.
    Accepts plain ints as well (e.g., '20000').
    """
    if isinstance(v, int):
        return v
    s = str(v).strip().lower()
    m = re.fullmatch(r"(\d+(?:\.\d+)?)(k)?", s)
    if not m:
        raise argparse.ArgumentTypeError(f"Invalid numeric value with optional 'k' suffix: {v}")
    base = float(m.group(1))
    is_k = bool(m.group(2))
    return int(base * 1000 if is_k else base)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--lr", type=float, default=5e-4, help="Constant learning rate for PPO")
    p.add_argument("--seed_start", type=int, default=100)
    p.add_argument("--seed_end", type=int, default=109)
    p.add_argument("--total_timesteps", type=int, default=1_000_000)
    # Allow k-suffix for stiffness values passed via flags too
    p.add_argument("--stiffness_start", type=_parse_int_with_k, default=_parse_int_with_k("5k"),
                   help="Start stiffness. Supports 'k' suffix, e.g., 5k = 5000.")
    p.add_argument("--stiffness_end", type=_parse_int_with_k, default=_parse_int_with_k("50k"),
                   help="End stiffness. Supports 'k' suffix, e.g., 50k = 50000.")
    p.add_argument("--max_episode_steps", type=int, default=1_000)
    p.add_argument(
        "--growth_type",
        type=str,
        default="constant_20k",
        help=("Stiffness schedule type. "
              "Supports: 'constant', 'constant_<val>[k]', 'linear', 'expo', 'log'. "
              "Examples: constant_20k | constant | linear | expo | log")
    )
    p.add_argument("--lr_schedule_type", type=str, default="constant", help="Learning rate schedule type (reserved)")
    p.add_argument("--curve_param", type=float, default=9.0,
                   help="Shape parameter for 'log' schedule (larger -> more front-loaded).")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Basic sanity checks
    if args.seed_end < args.seed_start:
        raise ValueError(f"seed_end ({args.seed_end}) must be >= seed_start ({args.seed_start})")

    run_date = datetime.now().strftime("%b%d")

    # ----------------------------------------------
    # Parse growth_type
    #   - constant_<val>[k]  -> constant schedule with parsed value
    #   - constant           -> constant schedule using stiffness_start
    #   - linear/expo/log    -> dynamic schedules, require distinct start/end
    # ----------------------------------------------
    schedule_type = args.growth_type.lower().strip()

    if schedule_type.startswith("constant_"):
        match = re.search(r"constant_(\d+(?:\.\d+)?)(k?)", schedule_type)
        if match:
            base_val = float(match.group(1))
            is_k = match.group(2) == "k"
            stiffness_val = int(base_val * 1000 if is_k else base_val)
            args.stiffness_start = stiffness_val
            args.stiffness_end = stiffness_val
            schedule_type = "constant"
            print(f"[INFO] Parsed constant stiffness: {stiffness_val} from growth_type '{args.growth_type}'")
        else:
            print(f"[WARN] Could not parse stiffness from growth_type: '{args.growth_type}'. Falling back to 'constant'.")
            schedule_type = "constant"

    elif schedule_type == "constant":
        # Use provided stiffness_start as the fixed value (end matched)
        args.stiffness_end = args.stiffness_start
        print(f"[INFO] Using constant stiffness: {args.stiffness_start}")

    elif schedule_type in {"linear", "expo", "log"}:
        print(f"[INFO] Using dynamic stiffness schedule: {schedule_type}")
        if args.stiffness_end == args.stiffness_start:
            # Provide a helpful automatic nudge
            suggested = args.stiffness_start * 10 if args.stiffness_start > 0 else _parse_int_with_k("50k")
            print(f"[WARN] stiffness_end == stiffness_start for dynamic '{schedule_type}'. "
                  f"Setting stiffness_end to {suggested}.")
            args.stiffness_end = suggested

        if schedule_type == "expo" and (args.stiffness_start <= 0 or args.stiffness_end <= 0):
            # Expo relies on logs; keep it explicit and helpful.
            print(f"[WARN] 'expo' requires positive start/end; switching to 'linear' with the same range.")
            schedule_type = "linear"

    else:
        raise ValueError(f"Unsupported growth_type: {args.growth_type}. "
                         f"Use 'constant', 'constant_<val>[k]', 'linear', 'expo', or 'log'.")

    # ----------------------------------------------
    # Use global range for folder naming consistency
    # ----------------------------------------------
    folder_seed_start = int(os.environ.get("SEED_RANGE_START", args.seed_start))
    folder_seed_end = int(os.environ.get("SEED_RANGE_END", args.seed_end))

    # ----------------------------------------------
    # Build TrainingConfig
    #   Note: pass schedule_type (normalized) to growth_type,
    #   not the raw 'constant_20k' literal.
    # ----------------------------------------------
    cfg = TrainingConfig(
        stiffness_start=int(args.stiffness_start),
        stiffness_end=int(args.stiffness_end),
        num_seeds=args.seed_end - args.seed_start + 1,
        total_timesteps=args.total_timesteps,
        lr=args.lr,
        seed_start=args.seed_start,
        seed_end=args.seed_end,
        run_date=run_date,
        max_episode_steps=args.max_episode_steps,
        growth_type=schedule_type,  # normalized name: constant | linear | expo | log
        folder_seed_start=folder_seed_start,
        folder_seed_end=folder_seed_end,
    )

    # If using 'log', pass curve_param via environment var (or extend TrainingConfig if you prefer)
    # Option A (minimal change): env reads os.environ.get("LOG_CURVE_PARAM", "9.0")
    # Option B (recommended): extend TrainingConfig/LegEnvBase to accept curve_param explicitly.
    os.environ["LOG_CURVE_PARAM"] = str(args.curve_param)

    # ----------------------------------------------
    # Run training loop for all seeds
    # ----------------------------------------------
    for seed in range(args.seed_start, args.seed_end + 1):
        print(f"🚀 Training | Seed={seed} | schedule={schedule_type} | "
              f"Stiffness={args.stiffness_start}->{args.stiffness_end} | LR={args.lr:.0e}")
        try:
            train(cfg, seed)
            print(f"✅ Finished training for Seed {seed}")
        except Exception as exc:
            print(f"❌ Failed training for Seed {seed}: {exc}")
            traceback.print_exc()

    print("✅ All seeds complete.")
    aggregate_and_save_results(cfg)


if __name__ == "__main__":
    main()
