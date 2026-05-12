#!/bin/bash
#===============================================================================
# PT: v2_ptm_mlm_cpcl_ssp  (--mlm --cpcl --ssp)
#
# Usage: ./scripts/v2_new_pt/pt_mlm_cpcl_ssp.sh [GPU_ID]
#===============================================================================
set -e

GPU=${1:-0}
PT_EXP="v2_ptm_mlm_cpcl_ssp"
PT_SEED=42
PT_PATH="saved_models/circmac/${PT_EXP}/${PT_SEED}/pretrain/model.pth"

D_MODEL=128; N_LAYER=6; MAX_LEN=1022; NUM_WORKERS=4
PT_DATA="df_pretrain"
PT_BS=64; PT_LR=1e-3; PT_WD=0.01; PT_EP=300; PT_ES=30

mkdir -p logs/v2/ptm saved_models

echo "========================================"
echo "  PT: $PT_EXP  (GPU $GPU)"
echo "  Flags: --mlm --cpcl --ssp"
echo "  BS=$PT_BS LR=$PT_LR EP=$PT_EP ES=$PT_ES"
echo "========================================"

if [ -f "$PT_PATH" ]; then
    echo "[SKIP] Already done: $PT_PATH"
    exit 0
fi

echo "[RUN]  $PT_EXP"
python pretraining.py \
    --model_name circmac --data_file "$PT_DATA" \
    --max_len $MAX_LEN --d_model $D_MODEL --n_layer $N_LAYER \
    --batch_size $PT_BS --num_workers $NUM_WORKERS \
    --optimizer adamw --lr $PT_LR --w_decay $PT_WD \
    --epochs $PT_EP --earlystop $PT_ES \
    --device $GPU --exp "$PT_EXP" --seed $PT_SEED \
    --verbose --mlm --cpcl --ssp \
    2>&1 | tee "logs/v2/ptm/${PT_EXP}.log"

if [ ! -f "$PT_PATH" ]; then
    echo "[ERROR] PT model not saved: $PT_PATH"; exit 1
fi

echo ""
echo "✓ Done: $PT_EXP"
echo "  Saved → $PT_PATH"
