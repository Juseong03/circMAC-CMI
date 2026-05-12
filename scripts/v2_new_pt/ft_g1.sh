#!/bin/bash
#===============================================================================
# FT GPU 1 — seed 2
#   1. v2_pt_mlm_cpcl_pairing_s2     ← v2_ptm_mlm_cpcl_pairing
#   2. v2_pt_mlm_cpcl_pairing_ssp_s2 ← v2_ptm_mlm_cpcl_pairing_ssp
#
# Usage: ./scripts/v2_new_pt/ft_g1.sh [GPU_ID]
#===============================================================================
set -e

GPU=${1:-1}
SEED=2
TASK="sites"
PT_SEED=42
D_MODEL=128; N_LAYER=6; MAX_LEN=1022; NUM_WORKERS=4
FT_BS=64; FT_LR=1e-4; FT_EP=150; FT_ES=20

mkdir -p logs/v2/pt saved_models
TOTAL=0; SKIPPED=0; RAN=0

echo "========================================"
echo "  FT GPU $GPU — seed=$SEED"
echo "  Jobs: mlm_cpcl_pairing, mlm_cpcl_pairing_ssp"
echo "  BS=$FT_BS LR=$FT_LR EP=$FT_EP ES=$FT_ES"
echo "========================================"

run_ft() {
    local EXP_KEY=$1
    local PT_EXP=$2
    local EXP="v2_pt_${EXP_KEY}_s${SEED}"
    local PT_PATH="saved_models/circmac/${PT_EXP}/${PT_SEED}/pretrain/model.pth"
    TOTAL=$((TOTAL+1))

    if [ ! -f "$PT_PATH" ]; then
        echo "[SKIP] $EXP — PT model missing: $PT_PATH"; TOTAL=$((TOTAL-1)); return
    fi
    if find "saved_models/circmac/${EXP}" -name "training.json" 2>/dev/null | grep -q .; then
        echo "[SKIP] $EXP"; SKIPPED=$((SKIPPED+1)); return
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
}

run_ft "mlm_cpcl_pairing"     "v2_ptm_mlm_cpcl_pairing"
run_ft "mlm_cpcl_pairing_ssp" "v2_ptm_mlm_cpcl_pairing_ssp"

echo ""
echo "━━━ Done: $RAN ran, $SKIPPED skipped / $TOTAL total ━━━"
