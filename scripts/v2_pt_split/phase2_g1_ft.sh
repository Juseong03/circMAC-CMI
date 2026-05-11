#!/bin/bash
#===============================================================================
# Phase 2 — GPU 1: Finetune seed=2 for all 3 new PT experiments
#
# Requires Phase 1 to be complete (all pretrained models must exist).
#
# Usage: ./scripts/v2_pt_split/phase2_g1_ft.sh [GPU_ID]
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
echo "  Phase 2 GPU $GPU — Finetune seed=$SEED"
echo "  Experiments: mlm_cpcl, pairing, mlm_cpcl_ssp"
echo "  FT: BS=$FT_BS LR=$FT_LR epochs=$FT_EP es=$FT_ES"
echo "========================================"

run_ft() {
    local EXP_KEY=$1
    local PT_EXP=$2
    local EXP="v2_pt_${EXP_KEY}_s${SEED}"
    local PT_PATH="saved_models/circmac/${PT_EXP}/${PT_SEED}/pretrain/model.pth"
    TOTAL=$((TOTAL+1))

    if [ ! -f "$PT_PATH" ]; then
        echo "[ERROR] PT model not found: $PT_PATH  (run phase1 first)"; exit 1
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

echo ""
echo "━━━ seed=$SEED finetunes ━━━"
run_ft "mlm_cpcl"     "v2_ptm_mlm_cpcl"
run_ft "pairing"      "v2_ptm_pairing"
run_ft "mlm_cpcl_ssp" "v2_ptm_mlm_cpcl_ssp"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Done: $RAN ran, $SKIPPED skipped / $TOTAL total"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
