#!/usr/bin/env bash
set -euo pipefail
cd /root/Fine-Tuning4Material
export PATH=/root/miniconda3/envs/llama/bin:"$PATH"
export PYTHONPATH=/root/Fine-Tuning4Material/src
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TMPDIR=/root/autodl-tmp/tmp
export HF_HOME=/root/autodl-tmp/huggingface
export XDG_CACHE_HOME=/root/autodl-tmp/cache
export MPLCONFIGDIR=/root/autodl-tmp/matplotlib
export WANDB_PROJECT=Fine-Tuning4Material
export WANDB_MODE=online
export WANDB_DIR=/root/autodl-tmp/wandb
export WANDB_LOG_MODEL=false
export WANDB_WATCH=false
export MATERIAL_MONITOR_CASES=/root/Fine-Tuning4Material/data/material_diagnostic_20261014/comparison_cases.json
export MATERIAL_MAX_WALL_SECONDS=13500
unset USE_MODELSCOPE_HUB
mkdir -p /root/autodl-tmp/logs "$WANDB_DIR" "$HF_HOME" "$TMPDIR" "$MPLCONFIGDIR"
exec python scripts/material_diagnostic_webui.py
