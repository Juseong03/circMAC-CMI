#!/bin/bash
# Localization focus score — Encoder baselines (LSTM, Transformer, Mamba, Hymba)
# Usage: bash scripts/localization_focus/run_encoders.sh <GPU>
GPU=${1:-0}
python scripts/compute_localization_focus.py \
    --split all \
    --device $GPU \
    --seeds 1 2 3 \
    --radii 0 5 10 20 \
    --model_filter lstm transformer mamba hymba
echo "=== encoders done ==="
