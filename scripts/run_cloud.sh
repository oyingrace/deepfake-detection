#!/usr/bin/env bash
set -euo pipefail

export WANDB_PROJECT=enhanced-deepfake-detector
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python3 -m src.train.train --config configs/train/aat_pgd.yaml
python3 -m src.eval.evaluate --checkpoint outputs/best.pt --config configs/train/aat_pgd.yaml
python3 -m src.eval.ablation
