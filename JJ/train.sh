#!/bin/bash
#SBATCH --job-name=ppo-${GROWTH}_${LR}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --begin=now+30minutes


module purge
eval "$(conda shell.bash hook)"
conda activate lab

cd /home1/jiajinzh/Dynamic-tendon-Leg/JJ/

# Read variables passed in
GROWTH_TYPE=$1
LR_SCHEDULE_TYPE=$2

python train.py --growth_type ${GROWTH_TYPE} --lr_schedule_type ${LR_SCHEDULE_TYPE} --seed_start 100 --seed_end 109
