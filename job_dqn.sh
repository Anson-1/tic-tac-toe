#!/bin/bash

RESUME=""
if [ -f torchrl_dqn/checkpoints/phase1_greedy/checkpoint.pt ]; then
    RESUME="--resume torchrl_dqn/checkpoints/phase1_greedy/checkpoint.pt"
elif [ -f torchrl_dqn/checkpoints/phase0_random/model_final.pt ]; then
    RESUME="--resume torchrl_dqn/checkpoints/phase0_random/model_final.pt"
fi

python torchrl_dqn/torchrl_dqn_train.py \
    --num-updates 500 \
    --phase 1 \
    --epsilon-start 0.3 \
    --episodes 128 \
    --batch-size 512 \
    --lr 1e-3 \
    --eval-every 50 \
    --checkpoint-dir torchrl_dqn/checkpoints \
    --device mps \
    $RESUME

echo "DQN phase 1 finished: $(date)"
