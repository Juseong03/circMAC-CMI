#!/bin/bash
# MAX — RNA-FM frozen + fine-tuned, seeds 1 2 3
# Usage: ./scripts/max/rnafm.sh [GPU_ID]
set -e
GPU=${1:-0}; MODEL="rnafm"; MAX_LEN=1022; PREFIX="max"
LR=1e-4; EPOCHS=150; EARLYSTOP=20; NUM_WORKERS=4; D_MODEL=128; N_LAYER=6

mkdir -p "logs/${PREFIX}" saved_models
TOTAL=0; SKIPPED=0; RAN=0
echo "=== MAX ${MODEL} (GPU $GPU) ==="

for SEED in 1 2 3; do
    for MODE in "frozen:128:0" "ft:32:1"; do
        TAG="${MODE%%:*}"; BS="${MODE#*:}"; BS="${BS%:*}"; TRAINABLE="${MODE##*:}"
        EXP="${PREFIX}_${MODEL}_${TAG}_s${SEED}"
        TOTAL=$((TOTAL+1))
        if find "saved_models/${MODEL}/${EXP}" -name "training.json" 2>/dev/null | grep -q .; then
            echo "[SKIP] $EXP"; SKIPPED=$((SKIPPED+1)); continue
        fi
        RAN=$((RAN+1)); echo "[RUN]  $EXP (bs=$BS)"
        EXTRA=""
        [ "$TRAINABLE" = "1" ] && EXTRA="--trainable_pretrained"
        python training.py \
            --model_name "$MODEL" --task sites --seed $SEED \
            --d_model $D_MODEL --n_layer $N_LAYER --max_len $MAX_LEN \
            --batch_size $BS --num_workers $NUM_WORKERS \
            --lr $LR --epochs $EPOCHS --earlystop $EARLYSTOP \
            --device $GPU --exp "$EXP" \
            --interaction cross_attention --verbose $EXTRA \
            2>&1 | tee "logs/${PREFIX}/${EXP}.log"
    done
done
echo "=== Done: $RAN ran, $SKIPPED skipped / $TOTAL total ==="
