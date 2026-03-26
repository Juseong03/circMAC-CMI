#!/bin/bash
#===============================================================================
# Exp3 H100 GPU1: Trainable models + CircMAC-PT
# Usage: ./scripts/final/exp3_h100_gpu1.sh [GPU_ID] [BEST_PT_EXP]
#   BEST_PT_EXP: exp2v3 pretrain exp name (default: exp2v3_pt_mlm_ntp)
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

echo "=== Exp3 H100 GPU1: Trainable + CircMAC-PT (GPU $GPU) ==="

# ── Trainable ─────────────────────────────────────────────────────────────────
echo "--- Trainable (bs=64) ---"
for MODEL in "${MODELS[@]}"; do
    ML=${MAX_LENS[$MODEL]}
    for SEED in "${SEEDS[@]}"; do
        TOTAL=$((TOTAL + 1))
        EXP_NAME="exp3_${MODEL}_trainable_s${SEED}"
        if find "saved_models/${MODEL}/${EXP_NAME}" -name "training.json" 2>/dev/null | grep -q .; then
            echo "[DONE] $EXP_NAME"; SKIPPED=$((SKIPPED + 1)); continue; fi
        RAN=$((RAN + 1)); echo "[RUN]  $EXP_NAME (max_len=$ML, bs=64)"
        python training.py \
            --model_name "$MODEL" --task "$TASK" --seed "$SEED" \
            --d_model "$D_MODEL" --max_len "$ML" \
            --batch_size 64 --num_workers "$NUM_WORKERS" \
            --lr "$LR" --epochs "$EPOCHS" --earlystop "$EARLYSTOP" \
            --device "$GPU" --exp "$EXP_NAME" \
            --interaction cross_attention --trainable_pretrained --verbose \
            2>&1 | tee "logs/exp3/${EXP_NAME}.log"
    done
done

# ── CircMAC-PT (exp2v3 완료 후) ───────────────────────────────────────────────
echo ""
echo "--- CircMAC-PT ($BEST_PT_EXP) ---"
PT_PATH="saved_models/circmac/${BEST_PT_EXP}/42/pretrain/model.pth"
if [ ! -f "$PT_PATH" ]; then
    echo "[WARN] Pretrain model not found: $PT_PATH"
    echo "  → exp2v3 완료 후 재실행:"
    echo "    ./scripts/final/exp3_h100_gpu1.sh $GPU <best_pt_exp>"
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

echo "=== GPU1 Complete: $RAN ran, $SKIPPED skipped / $TOTAL total ==="
