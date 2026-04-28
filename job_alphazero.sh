#!/bin/bash
#SBATCH --account=mscbdtsuperpod
#SBATCH --partition=normal
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --job-name=ttt-alphazero
#SBATCH --output=logs/alphazero_%j.out
#SBATCH --error=logs/alphazero_%j.err

mkdir -p logs checkpoints_az

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
    --checkpoint-dir checkpoints_az \
    --save-every 10 \
    --eval-every 10

echo "Job finished: $(date)"
