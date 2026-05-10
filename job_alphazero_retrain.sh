#!/bin/bash

export PYTHONPATH="$(pwd):$PYTHONPATH"

echo "AlphaZero retraining started: $(date)"

python -m super_tictactoe.alphazero_train \
    --iterations 200 \
    --games 100 \
    --simulations 50 \
    --epochs 10 \
    --batch-size 512 \
    --checkpoint-dir alphazero/checkpoints_retrained \
    --save-every 10 \
    --eval-every 10 \
    --init-from ppo_curriculum/checkpoints/model_final.pt \
    --device mps

echo "AlphaZero retraining finished: $(date)"
