#!/bin/bash
#===============================================================================
# FT: v2_pt_mlm_pairing_s2  ← v2_ptm_mlm_pairing  (seed 2)
#
# Usage: ./scripts/v2_new_pt/ft_mlm_pairing_s2.sh [GPU_ID]
#===============================================================================
set -e

GPU=${1:-0}
SEED=2
EXP="v2_pt_mlm_pairing_s${SEED}"
PT_EXP="v2_ptm_mlm_pairing"
PT_SEED=42
PT_PATH="saved_models/circmac/${PT_EXP}/${PT_SEED}/pretrain/model.pth"

TASK="sites"
D_MODEL=128; N_LAYER=6; MAX_LEN=1022; NUM_WORKERS=4
FT_BS=64; FT_LR=1e-4; FT_EP=150; FT_ES=20

mkdir -p logs/v2/pt saved_models

echo "========================================"
echo "  FT: $EXP  (GPU $GPU)"
echo "  PT: $PT_PATH"
echo "  BS=$FT_BS LR=$FT_LR EP=$FT_EP ES=$FT_ES"
echo "========================================"

if [ ! -f "$PT_PATH" ]; then
    echo "[ERROR] PT model not found: $PT_PATH"; exit 1
fi

if find "saved_models/circmac/${EXP}" -name "training.json" 2>/dev/null | grep -q .; then
    echo "[SKIP] Already done: $EXP"; exit 0
fi

echo "[RUN]  $EXP"
python training.py \
    --model_name circmac --task $TASK --seed $SEED \
    --d_model $D_MODEL --n_layer $N_LAYER --max_len $MAX_LEN \
    --batch_size $FT_BS --num_workers $NUM_WORKERS \
    --lr $FT_LR --epochs $FT_EP --earlystop $FT_ES \
    --device $GPU --exp $EXP \
    --interaction cross_attention \
    --load_pretrained "$PT_PATH" \
    --verbose \
    2>&1 | tee "logs/v2/pt/${EXP}.log"

echo ""
echo "Done: $EXP"
