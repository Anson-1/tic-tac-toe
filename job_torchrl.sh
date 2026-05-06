#!/bin/bash
#SBATCH --account=mscbdtsuperpod
#SBATCH --partition=normal
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --job-name=ttt-torchrl
#SBATCH --output=logs/torchrl_%j.out
#SBATCH --error=logs/torchrl_%j.err

mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tictactoe

echo "Job started: $(date)"
echo "Node: $(hostname)"

# Train DQN (1000 updates)
echo "=== Starting DQN training ==="
python torchrl_dqn/torchrl_dqn_train.py \
    --num-updates 1000 \
    --episodes 128 \
    --batch-size 512 \
    --lr 1e-3 \
    --eval-every 50 \
    --checkpoint-dir torchrl_dqn/checkpoints \
    --device cuda

echo "DQN finished: $(date)"

# Train A2C (1000 updates)
echo "=== Starting A2C training ==="
python torchrl_a2c/torchrl_a2c_train.py \
    --num-updates 1000 \
    --episodes 128 \
    --lr 3e-4 \
    --eval-every 50 \
    --checkpoint-dir torchrl_a2c/checkpoints \
    --device cuda

echo "A2C finished: $(date)"
echo "All done: $(date)"
