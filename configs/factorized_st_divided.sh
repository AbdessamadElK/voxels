#!/usr/bin/env bash
# Factorized space-time UNet, divided space-time attention at the bottleneck.
# Batch stays at the training config default so the arm compares against the baseline
# step for step: 17.1 GB at batch 6, under the 19.5 GB the baseline already takes.
# Pass --batch_size 4 for 11.4 GB on a smaller card.
set -euo pipefail

python train.py \
  --model_config    configs/models/factorized_st.json \
  --loss_config     configs/losses/default.json \
  --training_config configs/training/default.json \
  --checkpoint_dir  ckpts/factorized_st_divided \
  --wandb_name      factorized_st_divided \
  "$@"
