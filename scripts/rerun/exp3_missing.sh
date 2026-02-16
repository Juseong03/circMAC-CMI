#!/bin/bash
#===============================================================================
# RERUN: Exp3 Missing Runs
# - Transformer: all 3 seeds (batch_size=64 to avoid OOM)
# - CircMAC: seed 3 only
#
# Runs: 4
# Usage: ./scripts/rerun/exp3_missing.sh [GPU_ID]
#===============================================================================

set -e

GPU=${1:-0}
TASK="sites"

# Hyperparameters
D_MODEL=128
N_LAYER=6
EPOCHS=150
EARLYSTOP=20
LR=1e-4
NUM_WORKERS=4

mkdir -p logs/exp3
mkdir -p saved_models

TOTAL=0
SKIPPED=0
RAN=0

echo "=============================================="
echo "  RERUN: Exp3 Missing Runs"
echo "  GPU: $GPU | Expected: 4 runs"
echo "=============================================="

# Helper function to check if experiment already completed
check_and_run() {
    local MODEL=$1
    local SEED=$2
    local BS=$3
    local EXP_NAME="exp3_${MODEL}_${TASK}_s${SEED}"

    TOTAL=$((TOTAL + 1))

    RESULT_DIR="saved_models/${MODEL}/${EXP_NAME}"
    if find "$RESULT_DIR" -name "training.json" 2>/dev/null | grep -q .; then
        echo "[DONE] $EXP_NAME (already completed, skipping)"
        SKIPPED=$((SKIPPED + 1))
        return
    fi

    RAN=$((RAN + 1))
    echo "[RUN]  $EXP_NAME (batch_size=$BS)"

    python training.py \
        --model_name "$MODEL" \
        --task "$TASK" \
        --seed "$SEED" \
        --d_model "$D_MODEL" \
        --n_layer "$N_LAYER" \
        --batch_size "$BS" \
        --num_workers "$NUM_WORKERS" \
        --epochs "$EPOCHS" \
        --earlystop "$EARLYSTOP" \
        --lr "$LR" \
        --device "$GPU" \
        --exp "$EXP_NAME" \
        --interaction cross_attention \
        --verbose \
        2>&1 | tee "logs/exp3/${EXP_NAME}.log"
}

# 1. Transformer: all 3 seeds with batch_size=64 (OOM fix)
echo ""
echo "--- Transformer (batch_size=64) ---"
for SEED in 1 2 3; do
    check_and_run "transformer" "$SEED" 64
done

# 2. CircMAC: seed 3 only
echo ""
echo "--- CircMAC (seed 3) ---"
check_and_run "circmac" 3 128

echo ""
echo "=============================================="
echo "  RERUN Complete: Exp3 Missing"
echo "  Total: $TOTAL | Ran: $RAN | Skipped: $SKIPPED"
echo "=============================================="
