#!/bin/bash
# eval_full_encoder.sh — encoder 그룹 평가 (GPU 지정 가능)
# Usage: ./scripts/eval_full_encoder.sh [GPU]
GPU=${1:-0}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
echo "=== eval_full: encoder (GPU $GPU) ==="
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/eval_full.py --device "$GPU" --group encoder
