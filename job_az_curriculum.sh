#!/bin/bash
#SBATCH --account=mscbdtsuperpod
#SBATCH --partition=normal
#SBATCH --gpus-per-node=2
#SBATCH --time=24:00:00
#SBATCH --job-name=ttt-az-cl
#SBATCH --output=logs/az_curriculum_%j.out
#SBATCH --error=logs/az_curriculum_%j.err

mkdir -p logs checkpoints_az_cl

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tictactoe

echo "Job started: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -m super_tictactoe.alphazero_train \
    --iterations 200 \
    --games 100 \
    --simulations 50 \
    --epochs 10 \
    --batch-size 512 \
    --device cuda \
    --checkpoint-dir checkpoints_az_cl \
    --save-every 10 \
    --eval-every 10 \
    --curriculum

echo "Job finished: $(date)"
