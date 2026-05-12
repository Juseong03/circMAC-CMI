#!/bin/bash
#===============================================================================
# Phase 2 — GPU 0: Finetune for all resumed PT experiments (seeds 1,2,3)
#
# Requires phase1_g0_pt.sh to complete first.
#
# FT experiments:
#   v2_pt_bsj_s{1,2,3}              ← v2_ptm_bsj/42/pretrain/model.pth
#   v2_pt_mlm_cpcl_pairing_s{1,2,3} ← v2_ptm_mlm_cpcl_pairing/42/pretrain/model.pth
#   v2_pt_mlm_cpcl_ssp_s{1,2,3}    ← v2_ptm_mlm_cpcl_ssp/42/pretrain/model.pth
#
# Usage: ./scripts/v2_resume/phase2_g0_ft.sh [GPU_ID]
#===============================================================================
set -e

GPU=${1:-0}
TASK="sites"
PT_SEED=42
D_MODEL=128; N_LAYER=6; MAX_LEN=1022; NUM_WORKERS=4
FT_BS=64; FT_LR=1e-4; FT_EP=150; FT_ES=20

mkdir -p logs/v2/pt saved_models
TOTAL=0; SKIPPED=0; RAN=0

echo "========================================"
echo "  Phase 2 GPU $GPU — FT for resumed PT"
echo "  FT: BS=$FT_BS LR=$FT_LR epochs=$FT_EP es=$FT_ES"
echo "========================================"

run_ft() {
    local EXP_KEY=$1   # e.g. bsj
    local PT_EXP=$2    # e.g. v2_ptm_bsj
    local SEED=$3
    local EXP="v2_pt_${EXP_KEY}_s${SEED}"
    local PT_PATH="saved_models/circmac/${PT_EXP}/${PT_SEED}/pretrain/model.pth"
    TOTAL=$((TOTAL+1))

    if [ ! -f "$PT_PATH" ]; then
        echo "[SKIP] $EXP — PT model missing: $PT_PATH (run phase1 first)"
        TOTAL=$((TOTAL-1))
        return
    fi

    if find "saved_models/circmac/${EXP}" -name "training.json" 2>/dev/null | grep -q .; then
        echo "[SKIP] $EXP"
        SKIPPED=$((SKIPPED+1))
        return
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

for SEED in 1 2 3; do
    echo ""
    echo "━━━ seed=$SEED ━━━"
    run_ft "bsj"               "v2_ptm_bsj"               $SEED
    run_ft "mlm_cpcl_pairing"  "v2_ptm_mlm_cpcl_pairing"  $SEED
    run_ft "mlm_cpcl_ssp"      "v2_ptm_mlm_cpcl_ssp"      $SEED
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  FT done: $RAN ran, $SKIPPED skipped / $TOTAL total"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
