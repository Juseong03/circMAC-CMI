#!/bin/bash
#===============================================================================
# Missing Experiments — Server1 GPU 0
#
# EXP2: v2_pt_ssp  (MLM + SSP × 3 seeds)
#   Pretrain: v2_ptm_ssp  (--mlm --ssp, seed=42)
#   Finetune: v2_pt_ssp_s1, v2_pt_ssp_s2, v2_pt_ssp_s3
#
# Usage: ./scripts/v2_missing/run_missing_s1g0.sh [GPU_ID]
#===============================================================================
set -e

GPU=${1:-0}
SEEDS=(1 2 3)
TASK="sites"
PT_SEED=42

D_MODEL=128; N_LAYER=6; MAX_LEN=1022; NUM_WORKERS=4
PT_DATA="df_pretrain"
PT_BS=64; PT_LR=1e-3; PT_WD=0.01; PT_EP=300; PT_ES=30
FT_BS=64; FT_LR=1e-4; FT_EP=150; FT_ES=20

mkdir -p logs/v2/ptm logs/v2/pt saved_models
TOTAL=0; SKIPPED=0; RAN=0

echo "========================================"
echo "  Missing Experiments — Server1 GPU $GPU"
echo "  EXP2: v2_pt_ssp  (MLM + Secondary Structure)"
echo "  PT:  BS=$PT_BS LR=$PT_LR epochs=$PT_EP es=$PT_ES"
echo "  FT:  BS=$FT_BS LR=$FT_LR epochs=$FT_EP es=$FT_ES"
echo "========================================"

# ── Helper: pretraining ─────────────────────────────────────────────────────────
run_pretrain() {
    local STRATEGY=$1; shift
    local PT_EXP="v2_ptm_${STRATEGY}"
    TOTAL=$((TOTAL+1))
    local PT_MODEL="saved_models/circmac/${PT_EXP}/${PT_SEED}/pretrain/model.pth"
    if [ -f "$PT_MODEL" ]; then
        echo "[SKIP] pretrain: $PT_EXP"; SKIPPED=$((SKIPPED+1)); return 0
    fi
    RAN=$((RAN+1)); echo "[RUN]  pretrain: $PT_EXP"
    python pretraining.py \
        --model_name circmac --data_file "$PT_DATA" \
        --max_len $MAX_LEN --d_model $D_MODEL --n_layer $N_LAYER \
        --batch_size $PT_BS --num_workers $NUM_WORKERS \
        --optimizer adamw --lr $PT_LR --w_decay $PT_WD \
        --epochs $PT_EP --earlystop $PT_ES \
        --device $GPU --exp "$PT_EXP" --seed $PT_SEED \
        --verbose "$@" \
        2>&1 | tee "logs/v2/ptm/${PT_EXP}.log"
}

# ── Helper: finetuning ──────────────────────────────────────────────────────────
run_finetune() {
    local STRATEGY=$1 PT_PATH=$2
    for SEED in "${SEEDS[@]}"; do
        local EXP="v2_pt_${STRATEGY}_s${SEED}"
        TOTAL=$((TOTAL+1))
        if find "saved_models/circmac/${EXP}" -name "training.json" 2>/dev/null | grep -q .; then
            echo "[SKIP] $EXP"; SKIPPED=$((SKIPPED+1)); continue
        fi
        RAN=$((RAN+1)); echo "[RUN]  $EXP"
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

# ══ v2_pt_ssp: MLM + SSP ═════════════════════════════════════════════════════
echo ""
echo "━━━ v2_pt_ssp: MLM + Secondary Structure Prediction ━━━"
run_pretrain "ssp" --mlm --ssp
PT_PATH="saved_models/circmac/v2_ptm_ssp/${PT_SEED}/pretrain/model.pth"
if [ -f "$PT_PATH" ]; then
    run_finetune "ssp" "$PT_PATH"
else
    echo "[WARN] Pretrain model not found after PT run: $PT_PATH"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Done: $RAN ran, $SKIPPED skipped / $TOTAL total"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
