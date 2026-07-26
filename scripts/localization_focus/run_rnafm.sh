#!/bin/bash
# Localization focus score — RNA-FM (ft)
# Usage: bash scripts/localization_focus/run_rnafm.sh <GPU>
GPU=${1:-0}
python scripts/compute_localization_focus.py \
    --split all \
    --device $GPU \
    --seeds 1 2 3 \
    --radii 0 5 10 20 \
    --model_filter rnafm
echo "=== rnafm done ==="
