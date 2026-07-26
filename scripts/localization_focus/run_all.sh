#!/bin/bash
# Run all models sequentially on a single GPU
# Usage: bash scripts/localization_focus/run_all.sh <GPU>
GPU=${1:-0}

echo "=== Localization Focus Score — All Models ==="
python scripts/compute_localization_focus.py \
    --split all \
    --device $GPU \
    --seeds 1 2 3 \
    --radii 0 5 10 20
echo "=== All done ==="
