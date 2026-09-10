#!/usr/bin/env bash
# Same model as factorized_st_divided.sh, axial attention instead: time, then H, then W.
# A third faster per step for 11% more parameters.
set -euo pipefail

python train.py \
  --model_config    configs/models/factorized_st_axial.json \
  --loss_config     configs/losses/default.json \
  --training_config configs/training/default.json \
  --checkpoint_dir  ckpts/factorized_st_axial \
  --wandb_name      factorized_st_axial \
  "$@"
