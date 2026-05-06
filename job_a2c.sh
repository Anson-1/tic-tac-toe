#!/bin/bash
#SBATCH --account=mscbdtsuperpod
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --job-name=ttt-a2c
#SBATCH --output=logs/a2c_%j.out
#SBATCH --error=logs/a2c_%j.err

cd $SLURM_SUBMIT_DIR
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tictactoe

echo "Job started: $(date)"
echo "Node: $(hostname)"

python torchrl_a2c/torchrl_a2c_train.py \
    --num-updates 1000 \
    --episodes 128 \
    --lr 3e-4 \
    --eval-every 50 \
    --checkpoint-dir torchrl_a2c/checkpoints \
    --device cpu

echo "A2C finished: $(date)"
