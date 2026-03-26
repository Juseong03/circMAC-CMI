#!/bin/bash
#===============================================================================
# Exp3: Pretrained RNA Model Comparison — H100 optimized
# RQ: Does CircMAC-PT outperform general-purpose pretrained RNA models?
#
# Model    | max_len | data coverage
# ─────────────────────────────────
# RNABERT  |   440   |   37.6%  †
# RNAErnie |   510   |   48.2%  ††
# RNA-FM   |  1024   |  100.0%
# RNA-MSM  |  1024   |  100.0%
# CircMAC  |  1022   |  100.0%
#
# H100 (80GB): trainable 모델 bs=64 가능
#
# Usage: ./scripts/final/exp3_h100.sh [GPU_ID] [BEST_PT_EXP]
#   BEST_PT_EXP: exp2v3_pt_mlm_ntp (default, after exp2v3 completes)
#===============================================================================
set -e

GPU=${1:-0}
BEST_PT_EXP=${2:-"exp2v3_pt_mlm_ntp"}
SEEDS=(1 2 3)
TASK="sites"

D_MODEL=128; N_LAYER=6; LR=1e-4; EPOCHS=150; EARLYSTOP=20; NUM_WORKERS=4

declare -A MAX_LENS=( ["rnabert"]=440 ["rnaernie"]=510 ["rnafm"]=1024 ["rnamsm"]=1024 )
MODELS=(rnabert rnaernie rnafm rnamsm)

mkdir -p logs/exp3 saved_models
TOTAL=0; SKIPPED=0; RAN=0

echo "=============================================="
echo "  Exp3: Pretrained RNA Model Comparison (H100)"
echo "  GPU: $GPU"
echo "=============================================="

run_ft() {
    local EXP_NAME=$1; local MODEL=$2; local ML=$3; local BS=$4; shift 4
    TOTAL=$((TOTAL + 1))
    if find "saved_models/${MODEL}/${EXP_NAME}" -name "training.json" 2>/dev/null | grep -q .; then
        echo "[DONE] $EXP_NAME"; SKIPPED=$((SKIPPED + 1)); return 0; fi
    RAN=$((RAN + 1)); echo "[RUN]  $EXP_NAME (max_len=$ML, bs=$BS)"
    python training.py \
        --model_name "$MODEL" --task "$TASK" --seed "$SEED" \
        --d_model "$D_MODEL" --max_len "$ML" \
        --batch_size "$BS" --num_workers "$NUM_WORKERS" \
        --lr "$LR" --epochs "$EPOCHS" --earlystop "$EARLYSTOP" \
        --device "$GPU" --exp "$EXP_NAME" \
        --interaction cross_attention --verbose "$@" \
        2>&1 | tee "logs/exp3/${EXP_NAME}.log"
}

# ── 3A: Frozen ────────────────────────────────────────────────────────────────
echo ""
echo "--- 3A: Frozen (bs=64) ---"
for MODEL in "${MODELS[@]}"; do
    ML=${MAX_LENS[$MODEL]}
    for SEED in "${SEEDS[@]}"; do
        run_ft "exp3_${MODEL}_frozen_s${SEED}" "$MODEL" "$ML" 64
    done
done

# ── 3B: Trainable ─────────────────────────────────────────────────────────────
echo ""
echo "--- 3B: Trainable (H100: bs=64) ---"
for MODEL in "${MODELS[@]}"; do
    ML=${MAX_LENS[$MODEL]}
    for SEED in "${SEEDS[@]}"; do
        run_ft "exp3_${MODEL}_trainable_s${SEED}" "$MODEL" "$ML" 64 --trainable_pretrained
    done
done

# ── 3C: CircMAC-PT ────────────────────────────────────────────────────────────
echo ""
echo "--- 3C: CircMAC-PT (exp2v3 best) ---"
PT_PATH="saved_models/circmac/${BEST_PT_EXP}/42/pretrain/model.pth"
if [ ! -f "$PT_PATH" ]; then
    echo "[WARN] Pretrain model not found: $PT_PATH"
    echo "  → exp2v3 완료 후 재실행: ./scripts/final/exp3_h100.sh $GPU $BEST_PT_EXP"
else
    for SEED in "${SEEDS[@]}"; do
        TOTAL=$((TOTAL + 1))
        EXP_NAME="exp3_circmac_pt_s${SEED}"
        if find "saved_models/circmac/${EXP_NAME}" -name "training.json" 2>/dev/null | grep -q .; then
            echo "[DONE] $EXP_NAME"; SKIPPED=$((SKIPPED + 1)); continue; fi
        RAN=$((RAN + 1)); echo "[RUN]  $EXP_NAME"
        python training.py \
            --model_name circmac --task "$TASK" --seed "$SEED" \
            --d_model "$D_MODEL" --n_layer "$N_LAYER" --max_len 1022 \
            --batch_size 64 --num_workers "$NUM_WORKERS" \
            --lr "$LR" --epochs "$EPOCHS" --earlystop "$EARLYSTOP" \
            --device "$GPU" --exp "$EXP_NAME" \
            --load_pretrained "$PT_PATH" \
            --interaction cross_attention --verbose \
            2>&1 | tee "logs/exp3/${EXP_NAME}.log"
    done
fi

echo ""
echo "=============================================="
echo "  Exp3 H100 Complete: $RAN ran, $SKIPPED skipped / $TOTAL total"
echo "=============================================="
