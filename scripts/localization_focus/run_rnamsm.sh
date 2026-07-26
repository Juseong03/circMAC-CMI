#!/bin/bash
# Localization focus score — RNAMSM (ft)
# Usage: bash scripts/localization_focus/run_rnamsm.sh <GPU>
GPU=${1:-0}
python scripts/compute_localization_focus.py \
    --split all \
    --device $GPU \
    --seeds 1 2 3 \
    --radii 0 5 10 20 \
    --model_filter rnamsm
echo "=== rnamsm done ==="
