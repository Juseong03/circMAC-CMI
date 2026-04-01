#!/bin/bash
#===============================================================================
# exp_repair_0401: 보완 실험
#
# 대상:
#   1. exp4 no_circ_bias (3 seeds) — 이전에 CUDA 오류로 실패
#      → exp4_no_circular_bias 네이밍 사용 (이미 성공한 naming과 동일)
#   2. CircMAC-PT (exp2v4 best 모델 사용) — exp2v4 완료 후 실행
#
# Usage: ./scripts/final/exp_repair_0401.sh [GPU_ID] [BEST_PT_EXP]
#   BEST_PT_EXP: exp2v4에서 best pretrain exp 이름
#                default: exp2v4_pt_ntp
#===============================================================================
set -e

GPU=${1:-0}
BEST_PT_EXP=${2:-"exp2v4_pt_ntp"}
SEEDS=(1 2 3)
TASK="sites"
D_MODEL=128; N_LAYER=6; LR=1e-4; EPOCHS=150; EARLYSTOP=20; BS=32; NUM_WORKERS=4

mkdir -p logs/exp4 logs/exp3 saved_models
TOTAL=0; SKIPPED=0; RAN=0

echo "=== exp_repair_0401 (GPU $GPU) ==="

# ── 1. EXP4: no_circular_bias ────────────────────────────────────────────────
echo ""
echo "--- EXP4: no_circular_bias (no_circular_rel_bias) ---"
for SEED in "${SEEDS[@]}"; do
    TOTAL=$((TOTAL + 1))
    EXP_NAME="exp4_no_circular_bias_s${SEED}"
    if find "saved_models/circmac/${EXP_NAME}" -name "training.json" 2>/dev/null | grep -q .; then
        echo "[DONE] $EXP_NAME"; SKIPPED=$((SKIPPED + 1)); continue; fi
    RAN=$((RAN + 1)); echo "[RUN]  $EXP_NAME"
    python training.py \
        --model_name circmac --task "$TASK" --seed "$SEED" \
        --d_model "$D_MODEL" --n_layer "$N_LAYER" --max_len 1022 \
        --batch_size "$BS" --num_workers "$NUM_WORKERS" \
        --lr "$LR" --epochs "$EPOCHS" --earlystop "$EARLYSTOP" \
        --device "$GPU" --exp "$EXP_NAME" \
        --interaction cross_attention --no_circular_rel_bias --verbose \
        2>&1 | tee "logs/exp4/${EXP_NAME}.log"
done

# ── 2. CircMAC-PT ─────────────────────────────────────────────────────────────
echo ""
echo "--- CircMAC-PT ($BEST_PT_EXP) ---"
PT_PATH="saved_models/circmac/${BEST_PT_EXP}/42/pretrain/model.pth"
if [ ! -f "$PT_PATH" ]; then
    echo "[WARN] Pretrain model not found: $PT_PATH"
    echo "  → exp2_v4 완료 후 재실행:"
    echo "    ./scripts/final/exp_repair_0401.sh $GPU <best_pt_exp>"
else
    for SEED in "${SEEDS[@]}"; do
        TOTAL=$((TOTAL + 1))
        EXP_NAME="exp3_circmac_pt_s${SEED}"
        if find "saved_models/circmac/${EXP_NAME}" -name "training.json" 2>/dev/null | grep -q .; then
            echo "[DONE] $EXP_NAME"; SKIPPED=$((SKIPPED + 1)); continue; fi
        RAN=$((RAN + 1)); echo "[RUN]  $EXP_NAME (bs=$BS)"
        python training.py \
            --model_name circmac --task "$TASK" --seed "$SEED" \
            --d_model "$D_MODEL" --n_layer "$N_LAYER" --max_len 1022 \
            --batch_size "$BS" --num_workers "$NUM_WORKERS" \
            --lr "$LR" --epochs "$EPOCHS" --earlystop "$EARLYSTOP" \
            --device "$GPU" --exp "$EXP_NAME" \
            --load_pretrained "$PT_PATH" \
            --interaction cross_attention --verbose \
            2>&1 | tee "logs/exp3/${EXP_NAME}.log"
    done
fi

echo ""
echo "=== repair_0401 Complete: $RAN ran, $SKIPPED skipped / $TOTAL total ==="
