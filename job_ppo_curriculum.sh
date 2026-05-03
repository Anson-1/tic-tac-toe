#!/bin/bash
#SBATCH --account=mscbdtsuperpod
#SBATCH --partition=normal
#SBATCH --gpus-per-node=2
#SBATCH --time=24:00:00
#SBATCH --job-name=ttt-ppo-cl
#SBATCH --output=logs/ppo_curriculum_%j.out
#SBATCH --error=logs/ppo_curriculum_%j.err

mkdir -p logs checkpoints_ppo_cl

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tictactoe

echo "Job started: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -m super_tictactoe.train \
    --updates 3000 \
    --episodes 512 \
    --device cuda \
    --save-every 100 \
    --checkpoint-dir checkpoints_ppo_cl \
    --pool-size 10 \
    --pool-prob 0.5 \
    --curriculum

echo "Job finished: $(date)"
