#!/bin/bash
#===============================================================================
# Repair Script S3: EXP5 cross_attention + EXP3 rnafm  → Server3 GPU0
# Usage: ./scripts/final/exp_repair_s3.sh [GPU_ID]
#===============================================================================
set -e

GPU=${1:-0}
SEEDS=(1 2 3)
TASK="sites"

D_MODEL=128; N_LAYER=6; LR=1e-4; EPOCHS=150; EARLYSTOP=20; NUM_WORKERS=4
BS=128; BS_FM=8

mkdir -p logs/exp5 logs/exp3 saved_models
TOTAL=0; SKIPPED=0; RAN=0

echo "=== Repair S3: EXP5 cross_attention + EXP3 rnafm (GPU $GPU) ==="

echo ""
echo "--- EXP5: cross_attention s1,s2,s3 ---"
for SEED in "${SEEDS[@]}"; do
    TOTAL=$((TOTAL + 1))
    EXP_NAME="exp5_cross_attention_s${SEED}"
    if find "saved_models/circmac/${EXP_NAME}" -name "training.json" 2>/dev/null | grep -q .; then
        echo "[DONE] $EXP_NAME"; SKIPPED=$((SKIPPED + 1)); continue; fi
    RAN=$((RAN + 1)); echo "[RUN]  $EXP_NAME"
    python training.py \
        --model_name circmac --task "$TASK" --seed "$SEED" \
        --d_model "$D_MODEL" --n_layer "$N_LAYER" \
        --batch_size "$BS" --num_workers "$NUM_WORKERS" \
        --lr "$LR" --epochs "$EPOCHS" --earlystop "$EARLYSTOP" \
        --device "$GPU" --exp "$EXP_NAME" \
        --interaction cross_attention --verbose \
        2>&1 | tee "logs/exp5/${EXP_NAME}.log"
done

echo ""
echo "--- EXP3: RNA-FM trainable bs=8 ---"
for SEED in "${SEEDS[@]}"; do
    TOTAL=$((TOTAL + 1))
    EXP_NAME="exp3_rnafm_trainable_s${SEED}"
    if find "saved_models/rnafm/${EXP_NAME}" -name "training.json" 2>/dev/null | grep -q .; then
        echo "[DONE] $EXP_NAME"; SKIPPED=$((SKIPPED + 1)); continue; fi
    RAN=$((RAN + 1)); echo "[RUN]  $EXP_NAME (bs=$BS_FM)"
    python training.py \
        --model_name rnafm --task "$TASK" --seed "$SEED" \
        --batch_size "$BS_FM" --num_workers "$NUM_WORKERS" \
        --lr "$LR" --epochs "$EPOCHS" --earlystop "$EARLYSTOP" \
        --device "$GPU" --exp "$EXP_NAME" \
        --interaction cross_attention \
        --trainable_pretrained --verbose \
        2>&1 | tee "logs/exp3/${EXP_NAME}.log"
done

echo "=== Repair S3 Complete: $RAN ran, $SKIPPED skipped / $TOTAL total ==="
