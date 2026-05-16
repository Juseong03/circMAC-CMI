#!/bin/bash
# eval_full_pretrained.sh — pretrained 그룹 평가
# Usage: ./scripts/eval_full_pretrained.sh [GPU]
GPU=${1:-0}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
echo "=== eval_full: pretrained (GPU $GPU) ==="
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/eval_full.py --device "$GPU" --group pretrained
