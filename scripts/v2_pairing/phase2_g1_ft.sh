#!/bin/bash
#===============================================================================
# Phase 2 — GPU 1: Finetune seeds 1,2,3 for  v2_ptm_mlm_cpcl_ssp_pairing
#
# Requires Phase 1 GPU 1 to be complete.
# Run in PARALLEL with phase2_g0_ft.sh.
#
# Experiments:
#   v2_pt_mlm_cpcl_ssp_pairing_s1
#   v2_pt_mlm_cpcl_ssp_pairing_s2
#   v2_pt_mlm_cpcl_ssp_pairing_s3
#
# Usage: ./scripts/v2_pairing/phase2_g1_ft.sh [GPU_ID]
#===============================================================================
set -e

GPU=${1:-1}
SEEDS=(1 2 3)
TASK="sites"
PT_SEED=42
D_MODEL=128; N_LAYER=6; MAX_LEN=1022; NUM_WORKERS=4
FT_BS=64; FT_LR=1e-4; FT_EP=150; FT_ES=20

mkdir -p logs/v2/pt saved_models
TOTAL=0; SKIPPED=0; RAN=0

PT_EXP="v2_ptm_mlm_cpcl_ssp_pairing"
PT_PATH="saved_models/circmac/${PT_EXP}/${PT_SEED}/pretrain/model.pth"
EXP_KEY="mlm_cpcl_ssp_pairing"

echo "========================================"
echo "  Phase 2 GPU $GPU — Finetune: $EXP_KEY  seeds=(${SEEDS[*]})"
echo "  PT: $PT_EXP"
echo "  FT: BS=$FT_BS LR=$FT_LR epochs=$FT_EP es=$FT_ES"
echo "========================================"

if [ ! -f "$PT_PATH" ]; then
    echo "ERROR: Pretrained model not found: $PT_PATH"
    echo "       Run phase1_g1_pt.sh first."
    exit 1
fi

for SEED in "${SEEDS[@]}"; do
    EXP="v2_pt_${EXP_KEY}_s${SEED}"
    TOTAL=$((TOTAL+1))

    if find "saved_models/circmac/${EXP}" -name "training.json" 2>/dev/null | grep -q .; then
        echo "[SKIP] $EXP"; SKIPPED=$((SKIPPED+1)); continue
    fi

    RAN=$((RAN+1))
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
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Done: $RAN ran, $SKIPPED skipped / $TOTAL total"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
