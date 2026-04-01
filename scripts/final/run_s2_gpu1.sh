#!/bin/bash
#===============================================================================
# Server2 GPU1: CircMAC-PT — exp2v4 완료 후 best PT 모델로 finetune
#
# exp2v4 A/B/C 전부 완료된 후 실행
# best PT exp 결정 방법:
#   logs/exp2v4/finetune/ 의 F1 결과 비교 → 가장 높은 전략 선택
#
# Usage: ./scripts/final/run_s2_gpu1.sh [GPU_ID] [BEST_PT_EXP]
#   BEST_PT_EXP: default exp2v4_pt_ntp
#                ex) exp2v4_pt_ssp, exp2v4_pt_pair, exp2v4_pt_mlm_ntp, exp2v4_pt_all
#===============================================================================
set -e

GPU=${1:-1}
BEST_PT_EXP=${2:-"exp2v4_pt_ntp"}
SEEDS=(1 2 3)
TASK="sites"
D_MODEL=128; N_LAYER=6; LR=1e-4; EPOCHS=150; EARLYSTOP=20; BS=32; NUM_WORKERS=4

mkdir -p logs/exp3 saved_models
TOTAL=0; SKIPPED=0; RAN=0

echo "=== Server2 GPU1: CircMAC-PT ($BEST_PT_EXP, GPU $GPU) ==="

PT_PATH="saved_models/circmac/${BEST_PT_EXP}/42/pretrain/model.pth"
if [ ! -f "$PT_PATH" ]; then
    echo "[ERROR] Pretrain model not found: $PT_PATH"
    echo "  사용 가능한 exp2v4 pretrain 목록:"
    ls saved_models/circmac/ 2>/dev/null | grep exp2v4_pt || echo "  (없음)"
    echo ""
    echo "  → exp2v4 A/B/C 완료 후 재실행:"
    echo "    ./scripts/final/run_s2_gpu1.sh $GPU <best_pt_exp>"
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

echo "=== Server2 GPU1 Complete: $RAN ran, $SKIPPED skipped / $TOTAL total ==="
