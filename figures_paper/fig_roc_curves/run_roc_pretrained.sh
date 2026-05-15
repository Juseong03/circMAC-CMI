#!/bin/bash
# run_roc_pretrained.sh
# Computes ROC data for pretrained models and generates the figure.
#
# Models:
#   RNABERT, RNAErnie, RNAMSM, RNA-FM  (all fine-tuned)
#   CircMAC (Pairing pretraining)
#   CircMAC (NoPT — no pretraining)
#
# Usage:
#   ./figures_paper/fig_roc_curves/run_roc_pretrained.sh [DEVICE]
#   DEVICE: GPU index (default: 0)
#
# Output:
#   figures_paper/fig_roc_curves/roc_cache_pretrained.pkl
#   figures_paper/fig_roc_curves/fig_roc_pretrained.{pdf,png}

set -e

DEVICE=${1:-0}
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo "=========================================="
echo " ROC — Pretrained models  (device=$DEVICE)"
echo "=========================================="

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python figures_paper/fig_roc_curves/compute_roc_data.py \
    --device "$DEVICE" \
    --group pretrained

echo ""
echo "=== Generating figures ==="
python figures_paper/fig_roc_curves/fig_roc_curves.py

echo ""
echo "Done! Figures saved in figures_paper/fig_roc_curves/"
