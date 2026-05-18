#!/bin/bash
# run_seed4_verify.sh
# Seed 4 verification run for CircMAC v2_abl_full
# Purpose: compare logged metrics (fixed code) vs eval_full.py to verify label alignment
#
# Usage: ./scripts/run_seed4_verify.sh [GPU]
# Default GPU: 0

GPU=${1:-0}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

EXP="v2_abl_full"
SEED=4
MODEL="circmac"

echo "=== Seed 4 verification: $MODEL / $EXP / seed $SEED (GPU $GPU) ==="

LOG_DIR="logs_0512/$MODEL/${EXP}_s${SEED}/$SEED"
if [ -f "$LOG_DIR/training.json" ]; then
    echo "Already done: $LOG_DIR/training.json — skipping."
    exit 0
fi

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python training.py \
    --model_name $MODEL \
    --device $GPU \
    --task sites \
    --seed $SEED \
    --exp "${EXP}_s${SEED}" \
    --d_model 128 \
    --n_layer 6 \
    --batch_size 128 \
    --interaction cross_attention \
    --epochs 300 \
    --earlystop 20 \
    --verbose

echo "Done. Check logs_0512/$MODEL/${EXP}_s${SEED}/$SEED/training.log"
echo ""
echo "To eval with eval_full.py, add seed 4 entry to EXPS in scripts/eval_full.py and run:"
echo "  python3.10 scripts/eval_full.py --device $GPU --group encoder"
