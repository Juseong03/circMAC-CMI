#!/bin/bash
#===============================================================================
# Phase 1 — GPU 0: Re-run stopped PT experiments (sequential)
#
# Order (priority):
#   1. v2_ptm_bsj                 (261/300 stopped → BSJ pretraining)
#   2. v2_ptm_mlm_cpcl_pairing    (22/300 stopped)
#   3. v2_ptm_mlm_cpcl_ssp        (87/300 stopped)
#
# Usage: ./scripts/v2_resume/phase1_g0_pt.sh [GPU_ID]
#===============================================================================
set -e

GPU=${1:-0}
PT_SEED=42
D_MODEL=128; N_LAYER=6; MAX_LEN=1022; NUM_WORKERS=4
PT_DATA="df_pretrain"
PT_BS=64; PT_LR=1e-3; PT_WD=0.01; PT_EP=300; PT_ES=30

mkdir -p logs/v2/ptm saved_models
TOTAL=0; SKIPPED=0; RAN=0

echo "========================================"
echo "  Phase 1 GPU $GPU — Resume stopped PT"
echo "  BS=$PT_BS LR=$PT_LR epochs=$PT_EP es=$PT_ES"
echo "========================================"

run_pt() {
    local PT_EXP=$1; shift
    local FLAGS="$@"
    local PT_PATH="saved_models/circmac/${PT_EXP}/${PT_SEED}/pretrain/model.pth"
    TOTAL=$((TOTAL+1))

    if [ -f "$PT_PATH" ]; then
        echo "[SKIP] $PT_EXP (already done)"
        SKIPPED=$((SKIPPED+1))
        return
    fi

    RAN=$((RAN+1))
    echo "[RUN]  $PT_EXP  flags: $FLAGS"
    python pretraining.py \
        --model_name circmac --data_file "$PT_DATA" \
        --max_len $MAX_LEN --d_model $D_MODEL --n_layer $N_LAYER \
        --batch_size $PT_BS --num_workers $NUM_WORKERS \
        --optimizer adamw --lr $PT_LR --w_decay $PT_WD \
        --epochs $PT_EP --earlystop $PT_ES \
        --device $GPU --exp "$PT_EXP" --seed $PT_SEED \
        --verbose $FLAGS \
        2>&1 | tee "logs/v2/ptm/${PT_EXP}.log"

    if [ ! -f "$PT_PATH" ]; then
        echo "[ERROR] PT model not saved: $PT_PATH"; exit 1
    fi
    echo "[DONE] $PT_EXP"
}

echo ""
echo "━━━ PT experiments (GPU $GPU) ━━━"

# Priority 1: BSJ (was at 261/300 — nearly done)
run_pt "v2_ptm_bsj"               "--bsj_mlm"

# Priority 2: MLM+CPCL+Pairing
run_pt "v2_ptm_mlm_cpcl_pairing"  "--mlm --cpcl --pairing"

# Priority 4: MLM+CPCL+SSP
run_pt "v2_ptm_mlm_cpcl_ssp"      "--mlm --cpcl --ssp"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  PT done: $RAN ran, $SKIPPED skipped / $TOTAL total"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Next: ./scripts/v2_resume/phase2_g0_ft.sh $GPU"
