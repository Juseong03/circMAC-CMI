#!/bin/bash
# SUB511 — RNAMSM fine-tuned BS=32, seeds 1 2 3
# Usage: ./scripts/sub511/gpu1_rnamsm_ft.sh [GPU_ID]
set -e
GPU=${1:-1}; MODEL="rnamsm"; MAX_LEN=511; BS=32; PREFIX="sub511"
LR=1e-4; EPOCHS=150; EARLYSTOP=20; NUM_WORKERS=4; D_MODEL=128; N_LAYER=6

mkdir -p "logs/${PREFIX}" saved_models
TOTAL=0; SKIPPED=0; RAN=0
echo "=== SUB511 ${MODEL} fine-tuned (GPU $GPU) ==="

for SEED in 1 2 3; do
    EXP="${PREFIX}_${MODEL}_ft_s${SEED}"
    TOTAL=$((TOTAL+1))
    if find "saved_models/${MODEL}/${EXP}" -name "training.json" 2>/dev/null | grep -q .; then
        echo "[SKIP] $EXP"; SKIPPED=$((SKIPPED+1)); continue
    fi
    RAN=$((RAN+1)); echo "[RUN]  $EXP"
    python training.py \
        --model_name "$MODEL" --task sites --seed $SEED \
        --d_model $D_MODEL --n_layer $N_LAYER --max_len $MAX_LEN \
        --batch_size $BS --num_workers $NUM_WORKERS \
        --lr $LR --epochs $EPOCHS --earlystop $EARLYSTOP \
        --device $GPU --exp "$EXP" \
        --trainable_pretrained \
        --interaction cross_attention --verbose \
        2>&1 | tee "logs/${PREFIX}/${EXP}.log"
done
echo "=== Done: $RAN ran, $SKIPPED skipped / $TOTAL total ==="
