#!/bin/bash
#SBATCH --account=mscbdtsuperpod
#SBATCH --partition=normal
#SBATCH --gpus-per-node=2
#SBATCH --time=18:00:00
#SBATCH --job-name=ttt-ppo-phase2
#SBATCH --output=logs/ppo_phase2_%j.out
#SBATCH --error=logs/ppo_phase2_%j.err

mkdir -p logs checkpoints_ppo_phase2

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tictactoe

echo "Job started: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Phase 2: fine-tune from Phase 1 checkpoint with mostly self-play.
# 20% heuristic keeps defensive knowledge alive, 60% pool generalises strategy.
# Lower learning rate (1e-4) to avoid catastrophic forgetting.
python -m super_tictactoe.train \
    --updates 2000 \
    --episodes 512 \
    --device cuda \
    --save-every 100 \
    --checkpoint-dir checkpoints_ppo_phase2 \
    --resume checkpoints_ppo_phase1/model_final.pt \
    --pool-size 10 \
    --pool-prob 0.6 \
    --heuristic-prob 0.2 \
    --curriculum \
    --lr 1e-4

echo "Phase 2 finished: $(date)"
