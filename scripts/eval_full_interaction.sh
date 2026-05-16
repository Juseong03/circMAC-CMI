#!/bin/bash
# eval_full_interaction.sh — interaction + site_head 그룹 평가
# Usage: ./scripts/eval_full_interaction.sh [GPU]
GPU=${1:-0}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
echo "=== eval_full: interaction (GPU $GPU) ==="
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/eval_full.py --device "$GPU" --group interaction
echo "=== eval_full: site_head (GPU $GPU) ==="
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/eval_full.py --device "$GPU" --group site_head
