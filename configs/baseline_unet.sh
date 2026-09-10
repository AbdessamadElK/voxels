#!/usr/bin/env bash
# Baseline: the existing UNetTransformer with CombinedLoss, the arm the factorized
# space-time runs are measured against.
set -euo pipefail

python train.py \
  --model_config    configs/models/unet_transformer.json \
  --loss_config     configs/losses/default.json \
  --training_config configs/training/default.json \
  --checkpoint_dir  ckpts/baseline_unet \
  --wandb_name      baseline_unet \
  "$@"
