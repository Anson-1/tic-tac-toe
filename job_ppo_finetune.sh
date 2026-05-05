#!/bin/bash
#SBATCH --account=mscbdtsuperpod
#SBATCH --partition=normal
#SBATCH --gpus-per-node=2
#SBATCH --time=12:00:00
#SBATCH --job-name=ttt-ppo-finetune
#SBATCH --output=logs/ppo_finetune_%j.out
#SBATCH --error=logs/ppo_finetune_%j.err

mkdir -p logs checkpoints_ppo_finetune

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tictactoe

echo "Job started: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Fine-tune PPO-curriculum against heuristic opponents.
# Starts from the already-trained 57%-vs-blocking model.
# Low LR (1e-4) to preserve existing knowledge while learning to block better.
# 80% heuristic opponent forces explicit blocking practice.
python -m super_tictactoe.train \
    --updates 1000 \
    --episodes 512 \
    --device cuda \
    --save-every 100 \
    --checkpoint-dir checkpoints_ppo_finetune \
    --resume checkpoints_ppo_cl/model_final.pt \
    --pool-size 10 \
    --pool-prob 0.1 \
    --heuristic-prob 0.8 \
    --lr 1e-4

echo "Fine-tune finished: $(date)"
