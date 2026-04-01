#!/bin/bash
#===============================================================================
# Server3 GPU1: exp3 Trainable RNA Models
# RNABERT / RNAErnie / RNA-FM / RNA-MSM (trainable, cross_attention)
# A100 40GB batch sizes: rnabert/rnaernie=32, rnafm=8, rnamsm=16
#
# Usage: ./scripts/final/run_s3_gpu1.sh [GPU_ID]
#===============================================================================
set -e

GPU=${1:-1}
SEEDS=(1 2 3)
TASK="sites"
D_MODEL=128; LR=1e-4; EPOCHS=150; EARLYSTOP=20; NUM_WORKERS=4

declare -A MAX_LENS=(  ["rnabert"]=440 ["rnaernie"]=510 ["rnafm"]=1024 ["rnamsm"]=1024 )
declare -A BATCH_SIZES=( ["rnabert"]=32  ["rnaernie"]=32  ["rnafm"]=8    ["rnamsm"]=16  )
MODELS=(rnabert rnaernie rnafm rnamsm)

mkdir -p logs/exp3 saved_models
TOTAL=0; SKIPPED=0; RAN=0

echo "=== Server3 GPU1: Trainable RNA Models (GPU $GPU) ==="

for MODEL in "${MODELS[@]}"; do
    ML=${MAX_LENS[$MODEL]}
    BS=${BATCH_SIZES[$MODEL]}
    for SEED in "${SEEDS[@]}"; do
        TOTAL=$((TOTAL + 1))
        EXP_NAME="exp3_${MODEL}_trainable_s${SEED}"
        if find "saved_models/${MODEL}/${EXP_NAME}" -name "training.json" 2>/dev/null | grep -q .; then
            echo "[DONE] $EXP_NAME"; SKIPPED=$((SKIPPED + 1)); continue; fi
        RAN=$((RAN + 1)); echo "[RUN]  $EXP_NAME (max_len=$ML, bs=$BS)"
        python training.py \
            --model_name "$MODEL" --task "$TASK" --seed "$SEED" \
            --d_model "$D_MODEL" --max_len "$ML" \
            --batch_size "$BS" --num_workers "$NUM_WORKERS" \
            --lr "$LR" --epochs "$EPOCHS" --earlystop "$EARLYSTOP" \
            --device "$GPU" --exp "$EXP_NAME" \
            --interaction cross_attention --trainable_pretrained --verbose \
            2>&1 | tee "logs/exp3/${EXP_NAME}.log"
    done
done

echo "=== Server3 GPU1 Complete: $RAN ran, $SKIPPED skipped / $TOTAL total ==="
