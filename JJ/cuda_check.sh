#!/bin/bash
#SBATCH --job-name=check_cuda
#SBATCH --partition=gpu
#SBATCH --ntasks=1             # 加上这个，才能用 gpus-per-task
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:02:00

module purge
module load cuda/12.6.3        # 加载 CUDA 模块，必要时可改
eval "$(conda shell.bash hook)"
conda activate lab

cd /home1/jiajinzh/Dynamic-tendon-Leg/JJ/

echo "=== nvidia-smi ==="
nvidia-smi || echo "nvidia-smi not found!"

echo "=== Running CUDA check script ==="
python cuda_check.py
