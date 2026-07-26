#!/bin/bash
# Localization focus score — CircMAC (NoPT)
# Usage: bash scripts/localization_focus/run_circmac_nopt.sh <GPU>
GPU=${1:-0}
python scripts/compute_localization_focus.py \
    --split all \
    --device $GPU \
    --seeds 1 2 3 \
    --radii 0 5 10 20 \
    --model_filter circmac_nopt
echo "=== circmac_nopt done ==="
