#!/bin/bash
#===============================================================================
# Final Re-run — Script 2 (10-seed / 6-GPU friendly)
#
# EXP2 Pretraining Strategy Comparison
#   Pretrain phase : 6 strategies × 1 seed (42) = 6 PT runs
#   Finetune phase : 7 strategies × 10 seeds   = 70 FT runs
#
# Modes:
#   PRETRAIN_ONLY=1   -> run only the pretraining phase and write a marker
#   SKIP_PRETRAIN=1   -> wait for the marker and run finetuning only
#
# Usage:
#   ./scripts/final_v2/run_final_s2_10seed.sh [GPU_ID]
#===============================================================================
set -e

GPU=${1:-0}
if [ -n "${SEEDS_OVERRIDE:-}" ]; then
    read -r -a SEEDS <<< "$SEEDS_OVERRIDE"
else
    SEEDS=(1 2 3 4 5 6 7 8 9 10)
fi

PRETRAIN_ONLY=${PRETRAIN_ONLY:-0}
SKIP_PRETRAIN=${SKIP_PRETRAIN:-0}

TASK="sites"
PT_SEED=42

D_MODEL=128; N_LAYER=6; MAX_LEN=1022; NUM_WORKERS=4

# Pretraining hyperparams
PT_DATA="df_pretrain"
PT_BS=64; PT_LR=1e-3; PT_WD=0.01; PT_EP=300; PT_ES=30

# Finetuning hyperparams
FT_BS=64; FT_LR=1e-4; FT_EP=150; FT_ES=20

mkdir -p logs/v2/ptm logs/v2/pt saved_models
TOTAL=0; SKIPPED=0; RAN=0
PRETRAIN_DONE_MARKER="logs/v2/ptm/.exp2_pretrain_done"

echo "========================================"
echo "  Final Re-run Script 2 (10-seed)  (GPU $GPU)"
echo "  EXP2 Pretraining Strategy Comparison"
echo "  PRETRAIN_ONLY=$PRETRAIN_ONLY SKIP_PRETRAIN=$SKIP_PRETRAIN"
echo "========================================"

if [ "$SKIP_PRETRAIN" = "1" ] && [ ! -f "$PRETRAIN_DONE_MARKER" ]; then
    echo "[WAIT] pretraining marker not found: $PRETRAIN_DONE_MARKER"
    echo "       waiting for pretraining-only worker to finish..."
    while [ ! -f "$PRETRAIN_DONE_MARKER" ]; do
        sleep 60
    done
    echo "[READY] pretraining marker found."
fi

run_pretrain() {
    local STRATEGY=$1; shift
    local PT_EXP="v2_ptm_${STRATEGY}"
    TOTAL=$((TOTAL+1))
    local PT_MODEL="saved_models/circmac/${PT_EXP}/${PT_SEED}/pretrain/model.pth"
    if [ -f "$PT_MODEL" ]; then
        echo "[SKIP] pretrain: $PT_EXP"
        SKIPPED=$((SKIPPED+1))
        return 0
    fi
    RAN=$((RAN+1)); echo "[RUN]  pretrain: $PT_EXP"
    python pretraining.py \
        --model_name circmac \
        --data_file "$PT_DATA" \
        --max_len $MAX_LEN \
        --d_model $D_MODEL --n_layer $N_LAYER \
        --batch_size $PT_BS --num_workers $NUM_WORKERS \
        --optimizer adamw --lr $PT_LR --w_decay $PT_WD \
        --epochs $PT_EP --earlystop $PT_ES \
        --device $GPU --exp "$PT_EXP" --seed $PT_SEED \
        --verbose "$@" \
        2>&1 | tee "logs/v2/ptm/${PT_EXP}.log"
}

run_finetune() {
    local STRATEGY=$1
    local PT_PATH=$2
    for SEED in "${SEEDS[@]}"; do
        local EXP="v2_pt_${STRATEGY}_s${SEED}"
        TOTAL=$((TOTAL+1))
        if find "saved_models/circmac/${EXP}" -name "model.pth" 2>/dev/null | grep -q .; then
            echo "[SKIP] finetune: $EXP"
            SKIPPED=$((SKIPPED+1))
            continue
        fi
        RAN=$((RAN+1)); echo "[RUN]  finetune: $EXP"
        FT_CMD="python training.py \
            --model_name circmac --task $TASK --seed $SEED \
            --d_model $D_MODEL --n_layer $N_LAYER --max_len $MAX_LEN \
            --batch_size $FT_BS --num_workers $NUM_WORKERS \
            --lr $FT_LR --epochs $FT_EP --earlystop $FT_ES \
            --device $GPU --exp $EXP \
            --interaction cross_attention --verbose"
        [ "$PT_PATH" != "none" ] && FT_CMD="$FT_CMD --load_pretrained $PT_PATH"
        eval "$FT_CMD" 2>&1 | tee "logs/v2/pt/${EXP}.log"
    done
}

echo ""
echo "━━━ [1/7] No PT (baseline) ━━━"
if [ "$SKIP_PRETRAIN" != "1" ] && [ "$PRETRAIN_ONLY" != "1" ]; then
    run_finetune "nopt" "none"
fi

echo ""
echo "━━━ [2/7] MLM ━━━"
if [ "$SKIP_PRETRAIN" != "1" ]; then
    run_pretrain "mlm" --mlm
fi
if [ "$PRETRAIN_ONLY" != "1" ]; then
    PT_PATH="saved_models/circmac/v2_ptm_mlm/${PT_SEED}/pretrain/model.pth"
    [ -f "$PT_PATH" ] && run_finetune "mlm" "$PT_PATH"
fi

echo ""
echo "━━━ [3/7] NTP ━━━"
if [ "$SKIP_PRETRAIN" != "1" ]; then
    run_pretrain "ntp" --ntp
fi
if [ "$PRETRAIN_ONLY" != "1" ]; then
    PT_PATH="saved_models/circmac/v2_ptm_ntp/${PT_SEED}/pretrain/model.pth"
    [ -f "$PT_PATH" ] && run_finetune "ntp" "$PT_PATH"
fi

echo ""
echo "━━━ [4/7] MLM + SSP ━━━"
if [ "$SKIP_PRETRAIN" != "1" ]; then
    run_pretrain "mlm_ssp" --mlm --ssp
fi
if [ "$PRETRAIN_ONLY" != "1" ]; then
    PT_PATH="saved_models/circmac/v2_ptm_mlm_ssp/${PT_SEED}/pretrain/model.pth"
    [ -f "$PT_PATH" ] && run_finetune "mlm_ssp" "$PT_PATH"
fi

echo ""
echo "━━━ [5/7] MLM + CPCL ━━━"
if [ "$SKIP_PRETRAIN" != "1" ]; then
    run_pretrain "mlm_cpcl" --mlm --cpcl
fi
if [ "$PRETRAIN_ONLY" != "1" ]; then
    PT_PATH="saved_models/circmac/v2_ptm_mlm_cpcl/${PT_SEED}/pretrain/model.pth"
    [ -f "$PT_PATH" ] && run_finetune "mlm_cpcl" "$PT_PATH"
fi

echo ""
echo "━━━ [6/7] MLM + NTP ━━━"
if [ "$SKIP_PRETRAIN" != "1" ]; then
    run_pretrain "mlm_ntp" --mlm --ntp
fi
if [ "$PRETRAIN_ONLY" != "1" ]; then
    PT_PATH="saved_models/circmac/v2_ptm_mlm_ntp/${PT_SEED}/pretrain/model.pth"
    [ -f "$PT_PATH" ] && run_finetune "mlm_ntp" "$PT_PATH"
fi

echo ""
echo "━━━ [7/7] All (MLM+NTP+SSP+Pairing+CPCL+BSJ_MLM) ━━━"
if [ "$SKIP_PRETRAIN" != "1" ]; then
    run_pretrain "all" --mlm --ntp --ssp --pairing --cpcl --bsj_mlm
fi
if [ "$PRETRAIN_ONLY" != "1" ]; then
    PT_PATH="saved_models/circmac/v2_ptm_all/${PT_SEED}/pretrain/model.pth"
    [ -f "$PT_PATH" ] && run_finetune "all" "$PT_PATH"
fi

if [ "$PRETRAIN_ONLY" = "1" ]; then
    touch "$PRETRAIN_DONE_MARKER"
    echo "[MARK] wrote $PRETRAIN_DONE_MARKER"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Done: $RAN ran, $SKIPPED skipped / $TOTAL total"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
