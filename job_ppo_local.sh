#!/bin/bash

export PYTHONPATH="$(pwd):$PYTHONPATH"

RESUME=""
if [ -f torchrl_ppo/checkpoints/phase0_random/checkpoint.pt ]; then
    RESUME="--resume torchrl_ppo/checkpoints/phase0_random/checkpoint.pt"
elif [ -f torchrl_ppo/checkpoints/phase0_random/model_final.pt ]; then
    RESUME="--resume torchrl_ppo/checkpoints/phase0_random/model_final.pt"
fi

python torchrl_ppo/torchrl_train.py \
    --num-updates 500 \
    --phase 0 \
    --episodes 128 \
    --lr 3e-4 \
    --eval-every 50 \
    --checkpoint-dir torchrl_ppo/checkpoints \
    --device mps \
    $RESUME

echo "PPO phase 0 finished: $(date)"
