#!/bin/bash
# eval_full_pretraining.sh — pretraining 전략 비교 평가
# Usage: ./scripts/eval_full_pretraining.sh [GPU]
GPU=${1:-0}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
echo "=== eval_full: pretraining (GPU $GPU) ==="
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/eval_full.py --device "$GPU" --group pretraining
