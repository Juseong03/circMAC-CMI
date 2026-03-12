#!/bin/bash
# Exp1: CircMAC from scratch only (+ Hymba s3 missing)
# Usage: ./scripts/final/exp1_circmac.sh [GPU_ID]
set -e

GPU=${1:-0}
SEEDS=(1 2 3)
TASK="sites"
D_MODEL=128; N_LAYER=6; LR=1e-4; EPOCHS=150; EARLYSTOP=20; BS=128; NUM_WORKERS=4

mkdir -p logs/exp1 saved_models
TOTAL=0; SKIPPED=0; RAN=0

echo "=== Exp1: CircMAC + Hymba(s3) ==="

for MODEL in circmac hymba; do
    for SEED in "${SEEDS[@]}"; do
        TOTAL=$((TOTAL + 1))
        EXP_NAME="exp1_${MODEL}_s${SEED}"
        if find "saved_models/${MODEL}/${EXP_NAME}" -name "training.json" 2>/dev/null | grep -q .; then
            echo "[DONE] $EXP_NAME"; SKIPPED=$((SKIPPED + 1)); continue; fi
        RAN=$((RAN + 1))
        echo "[RUN]  $EXP_NAME"
        python training.py --model_name "$MODEL" --task "$TASK" --seed "$SEED" \
            --d_model $D_MODEL --n_layer $N_LAYER --batch_size $BS --num_workers $NUM_WORKERS \
            --lr $LR --epochs $EPOCHS --earlystop $EARLYSTOP --device $GPU \
            --exp "$EXP_NAME" --interaction cross_attention --verbose \
            2>&1 | tee "logs/exp1/${EXP_NAME}.log"
    done
done

echo "=== Done: $RAN ran, $SKIPPED skipped / $TOTAL total ==="
