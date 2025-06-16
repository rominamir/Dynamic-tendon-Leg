#!/bin/bash
# ------------------------------------------------------------------
# SLURM submission script for constant-stiffness PPO runs
# Args (expected from submit_jobs.py):
#   1) learning rate          (float)  → $LR
#   2) stiffness_start value  (int)    → $STIFF_START
#   3) stiffness_end value    (int)    → $STIFF_END (same as start for constant)
#   4) seed_start             (int)    → $SEED_START
#   5) seed_end               (int)    → $SEED_END
#   6) total_timesteps        (int)    → $TOTAL_TS
# ------------------------------------------------------------------

#SBATCH --job-name=ppo-${STIFF_START}_${LR}
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out

# --- Modules & env -------------------------------------------------
module purge
module load gcc/13.3.0
module load cuda/12.6.3

# (Assumes mamba/conda is available via module or profile)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lab_render
export MUJOCO_GL=egl


# --- Variables ------------------------------------------------------
LR=$1
STIFF_START=$2
STIFF_END=$3
SEED_START=$4
SEED_END=$5
TOTAL_TS=$6

# --- Launch ---------------------------------------------------------
cd /home1/jiajinzh/Dynamic-tendon-Leg/JJ/

python train.py \
  --lr $LR \
  --stiffness_start $STIFF_START \
  --stiffness_end $STIFF_END \
  --seed_start $SEED_START \
  --seed_end $SEED_END \
  --total_timesteps $TOTAL_TS
