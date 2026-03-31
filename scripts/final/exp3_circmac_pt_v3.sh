#!/bin/bash
#===============================================================================
# Exp3: CircMAC-PT finetune (exp2_v3 기반)
# mamba_ssm 호환 서버에서 실행 (H100 불가)
#
# Usage: ./scripts/final/exp3_circmac_pt_v3.sh [GPU_ID] [BEST_PT_EXP]
#   BEST_PT_EXP: exp2_v3에서 best pretraining exp 이름
#                default: exp2v3_pt_ntp  (NTP가 현재 best 단독 전략)
#
# Example:
#   ./scripts/final/exp3_circmac_pt_v3.sh 0 exp2v3_pt_ntp
#   ./scripts/final/exp3_circmac_pt_v3.sh 0 exp2v3_pt_mlm_ntp
#===============================================================================
set -e

GPU=${1:-0}
BEST_PT_EXP=${2:-"exp2v3_pt_ntp"}
SEEDS=(1 2 3)
TASK="sites"
D_MODEL=128; N_LAYER=6; LR=1e-4; EPOCHS=150; EARLYSTOP=20; BS=32; NUM_WORKERS=4

mkdir -p logs/exp3 saved_models
TOTAL=0; SKIPPED=0; RAN=0

echo "=== Exp3: CircMAC-PT (exp=$BEST_PT_EXP, GPU=$GPU) ==="

PT_PATH="saved_models/circmac/${BEST_PT_EXP}/42/pretrain/model.pth"
if [ ! -f "$PT_PATH" ]; then
    echo "[ERROR] Pretrain model not found: $PT_PATH"
    echo "  → exp2_v3 완료 후 실행하세요"
    echo "  → 경로 확인: ls saved_models/circmac/ | grep exp2v3"
    exit 1
fi
echo "  Pretrained model: $PT_PATH"

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

echo "=== Done: $RAN ran, $SKIPPED skipped / $TOTAL total ==="
