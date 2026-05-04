#!/bin/bash
#SBATCH --account=mscbdtsuperpod
#SBATCH --partition=normal
#SBATCH --gpus-per-node=2
#SBATCH --time=12:00:00
#SBATCH --job-name=ttt-ppo-phase1
#SBATCH --output=logs/ppo_phase1_%j.out
#SBATCH --error=logs/ppo_phase1_%j.err

mkdir -p logs checkpoints_ppo_phase1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tictactoe

echo "Job started: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Phase 1: train heavily against blocking heuristic to force learning of defensive play.
# 80% heuristic opponent, 10% pool, 10% self-play.
# Curriculum: deterministic → mild → stochastic.
python -m super_tictactoe.train \
    --updates 1000 \
    --episodes 512 \
    --device cuda \
    --save-every 100 \
    --checkpoint-dir checkpoints_ppo_phase1 \
    --pool-size 10 \
    --pool-prob 0.1 \
    --heuristic-prob 0.8 \
    --curriculum

echo "Phase 1 finished: $(date)"
