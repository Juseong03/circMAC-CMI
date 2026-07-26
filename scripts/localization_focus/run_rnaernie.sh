#!/bin/bash
# Localization focus score — RNAErnie (ft)
# Usage: bash scripts/localization_focus/run_rnaernie.sh <GPU>
GPU=${1:-0}
python scripts/compute_localization_focus.py \
    --split all \
    --device $GPU \
    --seeds 1 2 3 \
    --radii 0 5 10 20 \
    --model_filter rnaernie
echo "=== rnaernie done ==="
