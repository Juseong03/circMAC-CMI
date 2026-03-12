#!/bin/bash
# Exp3: CircMAC-PT finetune (run AFTER exp2_add completes)
# Usage: ./scripts/final/exp3_circmac_pt.sh [GPU_ID] [BEST_PT_CONFIG]
# Example: ./scripts/final/exp3_circmac_pt.sh 0 mlm_ntp_bsj
set -e

GPU=${1:-0}
BEST_PT=${2:-"mlm_ntp"}   # default: current best
SEEDS=(1 2 3)
TASK="sites"
D_MODEL=128; N_LAYER=6; LR=1e-4; EPOCHS=150; EARLYSTOP=20; BS=32; NUM_WORKERS=4

mkdir -p logs/exp3 saved_models
TOTAL=0; SKIPPED=0; RAN=0

echo "=== Exp3: CircMAC-PT (best_pt=$BEST_PT) ==="

PT_PATH="saved_models/circmac/exp2_pt_${BEST_PT}/42/pretrain/model.pth"
if [ ! -f "$PT_PATH" ]; then
    echo "[ERROR] Pretrain model not found: $PT_PATH"
    echo "  → Run exp2_add.sh first, then check best config"
    exit 1
fi
echo "  Using pretrained model: $PT_PATH"

for SEED in "${SEEDS[@]}"; do
    TOTAL=$((TOTAL + 1))
    EXP_NAME="exp3_circmac_pt_s${SEED}"
    if find "saved_models/circmac/${EXP_NAME}" -name "training.json" 2>/dev/null | grep -q .; then
        echo "[DONE] $EXP_NAME"; SKIPPED=$((SKIPPED + 1)); continue; fi
    RAN=$((RAN + 1))
    echo "[RUN]  $EXP_NAME (max_len=1022, bs=$BS)"
    python training.py --model_name circmac --task "$TASK" --seed "$SEED" \
        --d_model $D_MODEL --n_layer $N_LAYER --max_len 1022 \
        --batch_size $BS --num_workers $NUM_WORKERS \
        --lr $LR --epochs $EPOCHS --earlystop $EARLYSTOP --device $GPU \
        --exp "$EXP_NAME" --load_pretrained "$PT_PATH" \
        --interaction cross_attention --verbose \
        2>&1 | tee "logs/exp3/${EXP_NAME}.log"
done

echo "=== Done: $RAN ran, $SKIPPED skipped / $TOTAL total ==="
