#!/bin/bash
#SBATCH --account=mscbdtsuperpod
#SBATCH --partition=normal
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --job-name=ttt-train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# ── One-time setup (run this interactively BEFORE submitting, not in the job) ──
# srun --account=mscbdtsuperpod --partition=normal --gpus-per-node=1 --time=00:30:00 --pty bash
# conda create -n tictactoe python=3.11 -y
# conda activate tictactoe
# pip install torch torchvision numpy pytest
# pip install pygame  # optional, GUI not used on cluster
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p logs checkpoints

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tictactoe

# Confirm GPU is visible
echo "Job started: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Run training
python -m super_tictactoe.train \
    --updates 3000 \
    --episodes 512 \
    --device cuda \
    --save-every 100 \
    --eval-every 100 \
    --checkpoint-dir checkpoints

echo "Job finished: $(date)"
