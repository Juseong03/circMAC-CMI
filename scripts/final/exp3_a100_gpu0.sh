#!/bin/bash
#===============================================================================
# Exp3 A100 GPU0: Frozen models (RNABERT, RNAErnie, RNA-FM, RNA-MSM)
# A100 40GB — mamba_ssm 호환 (CircMAC 사용 가능)
#
# Batch sizes (frozen: backbone gradient 없음, activation only):
#   RNABERT  (125M, len= 440): bs=128
#   RNAErnie (125M, len= 510): bs=128
#   RNA-FM   (640M, len=1024): bs=32
#   RNA-MSM  (100M, len=1024): bs=64
#
# Usage: ./scripts/final/exp3_a100_gpu0.sh [GPU_ID]
#===============================================================================
set -e

GPU=${1:-0}
SEEDS=(1 2 3)
TASK="sites"
D_MODEL=128; N_LAYER=6; LR=1e-4; EPOCHS=150; EARLYSTOP=20; NUM_WORKERS=4

declare -A MAX_LENS=(  ["rnabert"]=440 ["rnaernie"]=510 ["rnafm"]=1024 ["rnamsm"]=1024 )
declare -A BATCH_SIZES=( ["rnabert"]=128 ["rnaernie"]=128 ["rnafm"]=32   ["rnamsm"]=64  )
MODELS=(rnabert rnaernie rnafm rnamsm)

mkdir -p logs/exp3 saved_models
TOTAL=0; SKIPPED=0; RAN=0

echo "=== Exp3 A100 GPU0: Frozen (GPU $GPU) ==="

for MODEL in "${MODELS[@]}"; do
    ML=${MAX_LENS[$MODEL]}
    BS=${BATCH_SIZES[$MODEL]}
    for SEED in "${SEEDS[@]}"; do
        TOTAL=$((TOTAL + 1))
        EXP_NAME="exp3_${MODEL}_frozen_s${SEED}"
        if find "saved_models/${MODEL}/${EXP_NAME}" -name "training.json" 2>/dev/null | grep -q .; then
            echo "[DONE] $EXP_NAME"; SKIPPED=$((SKIPPED + 1)); continue; fi
        RAN=$((RAN + 1)); echo "[RUN]  $EXP_NAME (max_len=$ML, bs=$BS)"
        python training.py \
            --model_name "$MODEL" --task "$TASK" --seed "$SEED" \
            --d_model "$D_MODEL" --max_len "$ML" \
            --batch_size "$BS" --num_workers "$NUM_WORKERS" \
            --lr "$LR" --epochs "$EPOCHS" --earlystop "$EARLYSTOP" \
            --device "$GPU" --exp "$EXP_NAME" \
            --interaction cross_attention --verbose \
            2>&1 | tee "logs/exp3/${EXP_NAME}.log"
    done
done

echo "=== GPU0 Complete: $RAN ran, $SKIPPED skipped / $TOTAL total ==="
