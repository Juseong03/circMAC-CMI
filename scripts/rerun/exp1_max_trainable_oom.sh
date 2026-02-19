#!/bin/bash
#===============================================================================
# RERUN: Exp1 max_trainable OOM fix
# rnafm, rnamsm trainable mode with batch_size=16 (was 32, OOM on 40GB GPU)
# rnaernie already completed successfully
#
# Runs: 6 (2 models × 3 seeds)
# Usage: ./scripts/rerun/exp1_max_trainable_oom.sh [GPU_ID]
#===============================================================================

set -e

GPU=${1:-0}
SEEDS=(1 2 3)
TASK="sites"

# Hyperparameters
D_MODEL=128
BATCH_SIZE=16    # Reduced from 32 to avoid OOM
EPOCHS=150
EARLYSTOP=20
LR=1e-4
NUM_WORKERS=4

OOM_MODELS=("rnafm" "rnamsm")

mkdir -p logs/exp1 saved_models

TOTAL=0; SKIPPED=0; RAN=0

echo "=============================================="
echo "  RERUN: Exp1 max_trainable OOM fix"
echo "  Models: ${OOM_MODELS[*]}"
echo "  batch_size: $BATCH_SIZE (reduced for OOM)"
echo "  GPU: $GPU | Runs: 6"
echo "=============================================="

for MODEL in "${OOM_MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        TOTAL=$((TOTAL + 1))
        EXP_NAME="exp1_max_trainable_${MODEL}_s${SEED}"

        # Check if already completed
        RESULT_DIR="saved_models/${MODEL}/${EXP_NAME}"
        if find "$RESULT_DIR" -name "training.json" 2>/dev/null | grep -q .; then
            echo "[DONE] $EXP_NAME"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        RAN=$((RAN + 1))
        echo "[RUN]  $EXP_NAME (batch_size=$BATCH_SIZE)"

        python training.py \
            --model_name "$MODEL" \
            --task "$TASK" \
            --seed "$SEED" \
            --d_model "$D_MODEL" \
            --batch_size "$BATCH_SIZE" \
            --num_workers "$NUM_WORKERS" \
            --lr "$LR" \
            --epochs "$EPOCHS" \
            --earlystop "$EARLYSTOP" \
            --device "$GPU" \
            --exp "$EXP_NAME" \
            --interaction cross_attention \
            --trainable_pretrained \
            --verbose \
            2>&1 | tee "logs/exp1/${EXP_NAME}.log"
    done
    echo "Completed: $MODEL (trainable, max, bs=$BATCH_SIZE)"
done

echo ""
echo "=============================================="
echo "  RERUN Complete: Exp1 max_trainable OOM"
echo "  Total: $TOTAL | Ran: $RAN | Skipped: $SKIPPED"
echo "=============================================="
