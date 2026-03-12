#!/bin/bash
# Exp3: Trainable RNA-FM + RNA-MSM (bs=16, large 95-96M param models)
# Usage: ./scripts/final/exp3_trainable_large.sh [GPU_ID]
set -e

GPU=${1:-0}
SEEDS=(1 2 3)
TASK="sites"
D_MODEL=128; LR=1e-4; EPOCHS=150; EARLYSTOP=20; BS=16; NUM_WORKERS=4
declare -A MAX_LENS=( ["rnafm"]=1024 ["rnamsm"]=1024 )

mkdir -p logs/exp3 saved_models
TOTAL=0; SKIPPED=0; RAN=0

echo "=== Exp3 Trainable: RNA-FM + RNA-MSM (bs=16) ==="

# Also run missing rnamsm frozen s3
EXP_NAME="exp3_rnamsm_frozen_s3"
if ! find "saved_models/rnamsm/${EXP_NAME}" -name "training.json" 2>/dev/null | grep -q .; then
    RAN=$((RAN + 1)); TOTAL=$((TOTAL + 1))
    echo "[RUN]  $EXP_NAME (frozen, missing seed)"
    python training.py --model_name rnamsm --task "$TASK" --seed 3 \
        --d_model $D_MODEL --max_len 1024 \
        --batch_size 32 --num_workers $NUM_WORKERS \
        --lr $LR --epochs $EPOCHS --earlystop $EARLYSTOP --device $GPU \
        --exp "$EXP_NAME" --interaction cross_attention --verbose \
        2>&1 | tee "logs/exp3/${EXP_NAME}.log"
else
    echo "[DONE] $EXP_NAME"; SKIPPED=$((SKIPPED + 1)); TOTAL=$((TOTAL + 1))
fi

for MODEL in rnafm rnamsm; do
    ML=${MAX_LENS[$MODEL]}
    for SEED in "${SEEDS[@]}"; do
        TOTAL=$((TOTAL + 1))
        EXP_NAME="exp3_${MODEL}_trainable_s${SEED}"
        if find "saved_models/${MODEL}/${EXP_NAME}" -name "training.json" 2>/dev/null | grep -q .; then
            echo "[DONE] $EXP_NAME"; SKIPPED=$((SKIPPED + 1)); continue; fi
        RAN=$((RAN + 1))
        echo "[RUN]  $EXP_NAME (max_len=$ML, bs=$BS)"
        python training.py --model_name "$MODEL" --task "$TASK" --seed "$SEED" \
            --d_model $D_MODEL --max_len $ML \
            --batch_size $BS --num_workers $NUM_WORKERS \
            --lr $LR --epochs $EPOCHS --earlystop $EARLYSTOP --device $GPU \
            --exp "$EXP_NAME" --interaction cross_attention \
            --trainable_pretrained --verbose \
            2>&1 | tee "logs/exp3/${EXP_NAME}.log"
    done
done

echo "=== Done: $RAN ran, $SKIPPED skipped / $TOTAL total ==="
