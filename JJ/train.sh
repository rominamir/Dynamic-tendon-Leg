#!/bin/bash
#SBATCH --job-name=ppo-${GROWTH}_${LR}
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1  
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out

module purge
module load gcc/13.3.0
module load cuda/12.6.3
eval "$(conda shell.bash hook)"
mamba activate lab

cd /home1/jiajinzh/Dynamic-tendon-Leg/JJ/

GROWTH_TYPE=$1
LR_SCHEDULE_TYPE=$2

python train.py --growth_type ${GROWTH_TYPE} --lr_schedule_type ${LR_SCHEDULE_TYPE} --seed_start 100 --seed_end 124