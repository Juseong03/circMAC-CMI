#!/bin/bash
#===============================================================================
# Exp1: Encoder Architecture Comparison
# RQ: Which encoder is best for circRNA binding site prediction?
#
# Models: LSTM, Transformer, Mamba, Hymba, CircMAC (all from scratch)
# Runs:   5 models × 3 seeds = 15 runs
# Usage:  ./scripts/final/exp1_encoders.sh [GPU_ID]
#===============================================================================
set -e

GPU=${1:-0}
SEEDS=(1 2 3)
TASK="sites"

# Common hyperparameters
D_MODEL=128; N_LAYER=6; LR=1e-4; EPOCHS=150; EARLYSTOP=20; NUM_WORKERS=4
BS=128  # default batch size

mkdir -p logs/exp1 saved_models

TOTAL=0; SKIPPED=0; RAN=0

echo "=============================================="
echo "  Exp1: Encoder Architecture Comparison"
echo "  GPU: $GPU | Runs: 15"
echo "=============================================="

for MODEL in lstm transformer mamba hymba circmac; do
    # Transformer OOM with bs=128 on some GPUs
    if [ "$MODEL" = "transformer" ]; then BS_RUN=64; else BS_RUN=$BS; fi

    for SEED in "${SEEDS[@]}"; do
        TOTAL=$((TOTAL + 1))
        EXP_NAME="exp1_${MODEL}_s${SEED}"

        if find "saved_models/${MODEL}/${EXP_NAME}" -name "training.json" 2>/dev/null | grep -q .; then
            echo "[DONE] $EXP_NAME"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        RAN=$((RAN + 1))
        echo "[RUN]  $EXP_NAME (bs=$BS_RUN)"

        python training.py \
            --model_name "$MODEL" \
            --task "$TASK" \
            --seed "$SEED" \
            --d_model "$D_MODEL" \
            --n_layer "$N_LAYER" \
            --batch_size "$BS_RUN" \
            --num_workers "$NUM_WORKERS" \
            --lr "$LR" \
            --epochs "$EPOCHS" \
            --earlystop "$EARLYSTOP" \
            --device "$GPU" \
            --exp "$EXP_NAME" \
            --interaction cross_attention \
            --verbose \
            2>&1 | tee "logs/exp1/${EXP_NAME}.log"
    done
    echo "  Done: $MODEL"
done

echo ""
echo "=============================================="
echo "  Exp1 Complete: $RAN ran, $SKIPPED skipped / $TOTAL total"
echo "=============================================="
