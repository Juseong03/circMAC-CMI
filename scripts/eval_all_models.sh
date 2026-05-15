#!/bin/bash
# eval_all_models.sh
# saved_models/ 의 모든 실험 결과를 올바른 방법으로 재평가하여 CSV로 저장
#
# Usage:
#   ./scripts/eval_all_models.sh [GPU]
#   GPU: GPU index (default: 0)
#
# Output:
#   eval_results/eval_summary.csv

set -e

GPU=${1:-0}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p eval_results

echo "=========================================="
echo " Eval all saved models  (GPU $GPU)"
echo "=========================================="

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/eval_all_models.py --device "$GPU"

echo ""
echo "Done! Results → eval_results/eval_summary.csv"
