#!/bin/bash
###############################################################################
# Constant-stiffness PPO — SLURM batch script
#
# Example submission:
#   sbatch train.sh 5e-4 5000 5000 100 109 1000000
#
# Positional args:
#   1) learning rate            → $LR
#   2) stiffness_start value    → $STIFF_START
#   3) stiffness_end value      → $STIFF_END   (same as start for constant)
#   4) first seed               → $SEED_START
#   5) last seed (inclusive)    → $SEED_END
#   6) total timesteps          → $TOTAL_TS
###############################################################################

######################## SLURM DIRECTIVES #####################################
#SBATCH --job-name=ppo_constant            # Do NOT use shell vars here
#SBATCH --partition=gpu                    # CARC GPU partition
# #SBATCH --gpus-per-task=a100:1             # Request exactly one A100
#SBATCH --gpus-per-task=1             # Request exactly one A100
# #SBATCH --constraint=a100-40gb           # Uncomment for 40 GB only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out   # → log/ppo_constant_<jobid>.out
#SBATCH --error=logs/%x_%j.err    # → log/ppo_constant_<jobid>.err
###############################################################################

######################## ARGUMENTS ############################################
LR="$1"
STIFF_START="$2"
STIFF_END="$3"
SEED_START="$4"
SEED_END="$5"
TOTAL_TS="$6"

######################## MODULES & ENVIRONMENT ################################
module purge
module load gcc/13.3.0
module load cuda/12.6.3

source "$(conda info --base)/etc/profile.d/conda.sh"
mamba activate lab_render                # Your Conda env with MuJoCo + PyTorch

# Headless OpenGL backend for MuJoCo
export MUJOCO_GL=egl
export XDG_RUNTIME_DIR="/tmp/xdg-runtime-$UID"
mkdir -p "$XDG_RUNTIME_DIR" && chmod 700 "$XDG_RUNTIME_DIR"

######################## LAUNCH TRAINING ######################################
cd /project2/valero_995/Dynamic-tendon-Leg/JJ/

python train.py \
  --growth_type linear \
  --lr "$LR" \
  --stiffness_start "$STIFF_START" \
  --stiffness_end "$STIFF_END" \
  --seed_start "$SEED_START" \
  --seed_end "$SEED_END" \
  --total_timesteps "$TOTAL_TS"
###############################################################################
