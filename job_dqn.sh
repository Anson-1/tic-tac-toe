#!/bin/bash
#SBATCH --account=mscbdtsuperpod
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --job-name=ttt-dqn
#SBATCH --output=logs/dqn_%j.out
#SBATCH --error=logs/dqn_%j.err

cd $SLURM_SUBMIT_DIR
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tictactoe

echo "Job started: $(date)"
echo "Node: $(hostname)"

python torchrl_dqn/torchrl_dqn_train.py \
    --num-updates 1000 \
    --episodes 128 \
    --batch-size 512 \
    --lr 1e-3 \
    --eval-every 50 \
    --checkpoint-dir torchrl_dqn/checkpoints \
    --device cpu

echo "DQN finished: $(date)"
