#!/bin/bash
#SBATCH --job-name=ppo-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out

module purge
eval "$(conda shell.bash hook)"
conda activate lab

cd /home1/jiajinzh/Dynamic-tendon-Leg/JJ/
python 15k.py
