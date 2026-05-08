#!/bin/bash

export PYTHONPATH="$(pwd):$PYTHONPATH"

# Phase 0: train vs random
python torchrl_ppo/torchrl_train.py \
    --phase 0 \
    --num-updates 500 \
    --episodes 128 \
    --lr 3e-4 \
    --eval-every 50 \
    --checkpoint-dir torchrl_ppo/checkpoints \
    --device mps

# Phase 1: train vs horizontal + self-play, resume from phase 0 best
python torchrl_ppo/torchrl_train.py \
    --phase 1 \
    --num-updates 500 \
    --episodes 128 \
    --lr 3e-4 \
    --eval-every 50 \
    --selfplay-prob 0.5 \
    --checkpoint-dir torchrl_ppo/checkpoints \
    --device mps \
    --resume torchrl_ppo/checkpoints/phase0_random/model_best.pt

echo "TorchRL PPO training finished: $(date)"
