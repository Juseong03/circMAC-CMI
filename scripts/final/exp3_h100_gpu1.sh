#!/bin/bash
#===============================================================================
# Exp3 H100 GPU1: Trainable models (RNABERT, RNAErnie, RNA-FM, RNA-MSM)
# H100 (80GB) — Mamba/CircMAC NOT supported (mamba_ssm 비호환)
# CircMAC-PT → 별도 서버에서 exp3_circmac_pt_v3.sh 사용
#
# Batch sizes (trainable: full gradient 필요):
#   RNABERT  (125M, len=440): bs=64
#   RNAErnie (125M, len=510): bs=64
#   RNA-FM   (640M, len=1024): bs=16  (대형 모델)
#   RNA-MSM  (100M, len=1024): bs=32
#
# Usage: ./scripts/final/exp3_h100_gpu1.sh [GPU_ID]
#===============================================================================
set -e

GPU=${1:-0}
SEEDS=(1 2 3)
TASK="sites"
D_MODEL=128; N_LAYER=6; LR=1e-4; EPOCHS=150; EARLYSTOP=20; NUM_WORKERS=4

declare -A MAX_LENS=(  ["rnabert"]=440 ["rnaernie"]=510 ["rnafm"]=1024 ["rnamsm"]=1024 )
declare -A BATCH_SIZES=( ["rnabert"]=64  ["rnaernie"]=64  ["rnafm"]=16   ["rnamsm"]=32  )
MODELS=(rnabert rnaernie rnafm rnamsm)

mkdir -p logs/exp3 saved_models
TOTAL=0; SKIPPED=0; RAN=0

echo "=== Exp3 H100 GPU1: Trainable (GPU $GPU) ==="

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

echo "=== GPU1 Complete: $RAN ran, $SKIPPED skipped / $TOTAL total ==="
echo ""
echo "NOTE: CircMAC-PT는 mamba_ssm 호환 서버에서 실행:"
echo "  ./scripts/final/exp3_circmac_pt_v3.sh [GPU_ID] [BEST_PT_EXP]"
