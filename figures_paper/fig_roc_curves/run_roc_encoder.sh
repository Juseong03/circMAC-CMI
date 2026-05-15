#!/bin/bash
# run_roc_encoder.sh
# Computes ROC data for encoder models and generates the figure.
#
# Usage:
#   ./figures_paper/fig_roc_curves/run_roc_encoder.sh [DEVICE]
#   DEVICE: GPU index (default: 0)
#
# Output:
#   figures_paper/fig_roc_curves/roc_cache_encoder.pkl
#   figures_paper/fig_roc_curves/fig2b_roc_encoder.{pdf,png}

set -e

DEVICE=${1:-0}
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo "========================================="
echo " ROC — Encoder models  (device=$DEVICE)"
echo "========================================="

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python figures_paper/fig_roc_curves/compute_roc_data.py \
    --device "$DEVICE" \
    --group encoder

echo ""
echo "=== Generating figures ==="
python figures_paper/fig_roc_curves/fig_roc_curves.py

echo ""
echo "Done! Figures saved in figures_paper/fig_roc_curves/"
