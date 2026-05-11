#!/bin/bash
#===============================================================================
# Phase 1 — GPU 0: Pretrain  v2_ptm_mlm_cpcl_pairing  (--mlm --cpcl --pairing)
#
# Run in PARALLEL with phase1_g1_pt.sh (mlm_cpcl_ssp_pairing on GPU 1).
# After both finish, run phase2_g0_ft.sh and phase2_g1_ft.sh.
#
# Usage: ./scripts/v2_pairing/phase1_g0_pt.sh [GPU_ID]
#===============================================================================
set -e

GPU=${1:-0}
PT_SEED=42
D_MODEL=128; N_LAYER=6; MAX_LEN=1022; NUM_WORKERS=4
PT_DATA="df_pretrain"
PT_BS=64; PT_LR=1e-3; PT_WD=0.01; PT_EP=300; PT_ES=30

mkdir -p logs/v2/ptm saved_models

PT_EXP="v2_ptm_mlm_cpcl_pairing"
PT_PATH="saved_models/circmac/${PT_EXP}/${PT_SEED}/pretrain/model.pth"

echo "========================================"
echo "  Phase 1 GPU $GPU — PT: $PT_EXP"
echo "  Flags: --mlm --cpcl --pairing"
echo "  Data: $PT_DATA"
echo "  BS=$PT_BS LR=$PT_LR epochs=$PT_EP es=$PT_ES"
echo "========================================"

if [ -f "$PT_PATH" ]; then
    echo "[SKIP] pretrain: $PT_EXP (already done)"
else
    echo "[RUN]  pretrain: $PT_EXP"
    python pretraining.py \
        --model_name circmac --data_file "$PT_DATA" \
        --max_len $MAX_LEN --d_model $D_MODEL --n_layer $N_LAYER \
        --batch_size $PT_BS --num_workers $NUM_WORKERS \
        --optimizer adamw --lr $PT_LR --w_decay $PT_WD \
        --epochs $PT_EP --earlystop $PT_ES \
        --device $GPU --exp "$PT_EXP" --seed $PT_SEED \
        --verbose --mlm --cpcl --pairing \
        2>&1 | tee "logs/v2/ptm/${PT_EXP}.log"
fi

if [ ! -f "$PT_PATH" ]; then
    echo "ERROR: Pretrained model not found: $PT_PATH"; exit 1
fi

echo ""
echo "✓ Phase 1 GPU $GPU done — $PT_EXP pretrained."
echo "  Now run: ./scripts/v2_pairing/phase2_g0_ft.sh $GPU"
