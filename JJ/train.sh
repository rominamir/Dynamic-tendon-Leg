#!/bin/bash
###############################################################################
# PPO (stiffness schedule = constant / linear / expo / log) — SLURM batch script
#
# Example submissions:
#   # Old usage (6 args): defaults to linear
#   sbatch train.sh 5e-4 5000 50000 100 109 1000000
#
#   # Explicit growth type (7th arg) and curve_param (8th arg, log only)
#   sbatch train.sh 5e-4 5k 50k 100 109 1_000_000 linear
#   sbatch train.sh 5e-4 5k 50k 100 100 1_000_000 expo
#   sbatch train.sh 5e-4 5k 50k 100 100 1_000_000 log 9.0
#   sbatch train.sh 5e-4 constant_20k constant_20k 100 109 1_000_000 constant
#
# Positional args:
#   1) learning rate            → $LR
#   2) stiffness_start value    → $STIFF_START   (supports k suffix; or constant_*)
#   3) stiffness_end value      → $STIFF_END     (same as start for constant)
#   4) first seed               → $SEED_START
#   5) last seed (inclusive)    → $SEED_END
#   6) total timesteps          → $TOTAL_TS
#   7) growth type (optional)   → $GROWTH_TYPE   [constant|constant_<val>k|linear|expo|log] (default: linear)
#   8) curve_param (optional)   → $CURVE_PARAM   (only for log; default: 9.0)
###############################################################################

######################## SLURM DIRECTIVES #####################################
#SBATCH --job-name=ppo_stiffness           # Job name
#SBATCH --partition=gpu                    # GPU partition
#SBATCH --gpus-per-task=1                   # Request 1 GPU
#SBATCH --nodes=1                           # Single node
#SBATCH --ntasks=1                          # Single task
#SBATCH --cpus-per-task=4                   # CPU cores per task
#SBATCH --mem=32G                           # Memory per node
#SBATCH --time=24:00:00                     # Max walltime
#SBATCH --output=logs/%x_%j.out             # STDOUT log file
#SBATCH --error=logs/%x_%j.err              # STDERR log file
###############################################################################

set -euo pipefail  # Exit on error, undefined variable, or pipe failure

######################## ARGUMENTS ############################################
LR="${1:?LR required}"                      # Learning rate
STIFF_START="${2:?stiffness_start required}" # Start stiffness
STIFF_END="${3:?stiffness_end required}"    # End stiffness
SEED_START="${4:?seed_start required}"      # First seed
SEED_END="${5:?seed_end required}"          # Last seed (inclusive)
TOTAL_TS="${6:?total_timesteps required}"   # Total timesteps
GROWTH_TYPE="${7:-linear}"                  # Growth type (default: linear)
CURVE_PARAM="${8:-9.0}"                     # Curve parameter for log schedule

######################## PREP: FOLDERS & MODULES ##############################
mkdir -p logs                               # Ensure log directory exists

module purge                                # Clear any loaded modules
module load gcc/13.3.0                      # GCC for compiling extensions
module load cuda/12.6.3                     # CUDA version for GPU support

# Activate Conda environment with MuJoCo, PyTorch, etc.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lab_render

######################## MUJOCO / RENDER ######################################
export MUJOCO_GL=egl                        # Use EGL for headless rendering
export XDG_RUNTIME_DIR="/tmp/xdg-runtime-$UID" # Temporary runtime dir
mkdir -p "$XDG_RUNTIME_DIR" && chmod 700 "$XDG_RUNTIME_DIR"

# Limit thread usage for numerical libs
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1                   # Unbuffered stdout/stderr

######################## FOLDER NAMING HINTS ##################################
# Pass seed range to Python for consistent folder naming and aggregation
export SEED_RANGE_START="$SEED_START"
export SEED_RANGE_END="$SEED_END"

# Pass log curve parameter via environment variable (used if growth_type='log')
export LOG_CURVE_PARAM="$CURVE_PARAM"

######################## LAUNCH TRAINING ######################################
cd /project2/valero_995/Dynamic-tendon-Leg/JJ/

# Run training script
# - growth_type supports: constant_20k / constant / linear / expo / log
# - stiffness values can use 'k' suffix (parsed in train.py)
# - For constant_20k, start=end will be normalized to constant mode
python train.py \
  --growth_type "$GROWTH_TYPE" \
  --lr "$LR" \
  --stiffness_start "$STIFF_START" \
  --stiffness_end "$STIFF_END" \
  --seed_start "$SEED_START" \
  --seed_end "$SEED_END" \
  --total_timesteps "$TOTAL_TS"

echo "✅ Training job completed."
