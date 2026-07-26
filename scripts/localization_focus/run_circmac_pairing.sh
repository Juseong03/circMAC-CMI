#!/bin/bash
# Localization focus score — CircMAC (Pairing)
# Usage: bash scripts/localization_focus/run_circmac_pairing.sh <GPU>
GPU=${1:-0}
python scripts/compute_localization_focus.py \
    --split all \
    --device $GPU \
    --seeds 1 2 3 \
    --radii 0 5 10 20 \
    --model_filter circmac_pairing
echo "=== circmac_pairing done ==="
