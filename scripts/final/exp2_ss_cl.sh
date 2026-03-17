#!/bin/bash
#===============================================================================
# Exp2-SS_CL: SS-pair Contrastive Learning experiments
#
# Configs (all with df_circ_ss_5, pair_mode=True):
#   ss_cl          — SS-CL only
#   mlm_ss_cl      — MLM + SS-CL
#   mlm_ntp_ss_cl  — MLM + NTP + SS-CL  (expected best)
#
# Each: 1 pretrain (seed=42) + 3 finetune (seed=1,2,3) = 4 runs × 3 = 12 runs
# Usage: ./scripts/final/exp2_ss_cl.sh [GPU_ID]
#===============================================================================
set -e

GPU=${1:-0}
SEEDS=(1 2 3)
TASK="sites"; PT_SEED=42; PREFIX="exp2"

DATA_FILE="df_circ_ss_5"; D_MODEL=128; N_LAYER=6; MAX_LEN=1022; NUM_WORKERS=4
PT_BS=64; PT_LR=5e-4; PT_WD=0.01; PT_EP=500; PT_ES=50
FT_BS=32; FT_LR=1e-4; FT_EP=150; FT_ES=20

mkdir -p logs/exp2/pretrain logs/exp2/finetune saved_models

TOTAL=0; SKIPPED=0; RAN=0

echo "=============================================="
echo "  Exp2 SS-CL: SS-pair Contrastive Learning"
echo "  GPU: $GPU | Data: $DATA_FILE (pair_mode)"
echo "=============================================="

run_pretrain_ss_cl() {
    local PT_EXP=$1; shift
    local PT_FLAGS="$@"

    TOTAL=$((TOTAL + 1))
    if find "saved_models/circmac/${PT_EXP}/${PT_SEED}/pretrain" -name "model.pth" 2>/dev/null | grep -q .; then
        echo "[DONE-PT] $PT_EXP"; SKIPPED=$((SKIPPED + 1)); return; fi

    RAN=$((RAN + 1))
    echo "[PT]  $PT_EXP  ($PT_FLAGS)"
    python pretraining.py \
        --model_name circmac --data_file $DATA_FILE --max_len $MAX_LEN \
        --d_model $D_MODEL --n_layer $N_LAYER --batch_size $PT_BS --num_workers $NUM_WORKERS \
        --optimizer adamw --lr $PT_LR --w_decay $PT_WD \
        --epochs $PT_EP --earlystop $PT_ES --device $GPU \
        --exp "$PT_EXP" --seed $PT_SEED --verbose \
        $PT_FLAGS \
        2>&1 | tee "logs/exp2/pretrain/${PT_EXP}.log"
}

run_finetune() {
    local PT_EXP=$1; local FT_SUFFIX=$2

    PT_PATH="saved_models/circmac/${PT_EXP}/${PT_SEED}/pretrain/model.pth"
    if [ ! -f "$PT_PATH" ]; then
        echo "[SKIP] pretrain model not found: $PT_PATH"; return; fi

    for SEED in "${SEEDS[@]}"; do
        TOTAL=$((TOTAL + 1))
        EXP_NAME="${PREFIX}_${FT_SUFFIX}_${TASK}_s${SEED}"
        if find "saved_models/circmac/${EXP_NAME}" -name "training.json" 2>/dev/null | grep -q .; then
            echo "[DONE] $EXP_NAME"; SKIPPED=$((SKIPPED + 1)); continue; fi
        RAN=$((RAN + 1))
        echo "[FT]  $EXP_NAME"
        python training.py \
            --model_name circmac --task "$TASK" --seed "$SEED" \
            --d_model $D_MODEL --n_layer $N_LAYER --batch_size $FT_BS --num_workers $NUM_WORKERS \
            --lr $FT_LR --epochs $FT_EP --earlystop $FT_ES --device $GPU \
            --exp "$EXP_NAME" --interaction cross_attention --verbose \
            --load_pretrained "$PT_PATH" \
            2>&1 | tee "logs/exp2/finetune/${EXP_NAME}.log"
    done
}

# ── Config 1: SS-CL only ──────────────────────────────────────────────────────
echo ""
echo "--- SS-CL only ---"
PT_EXP="${PREFIX}_pt_ss_cl"
run_pretrain_ss_cl "$PT_EXP" --ss_cl
run_finetune "$PT_EXP" "ss_cl"

# ── Config 2: MLM + SS-CL ────────────────────────────────────────────────────
echo ""
echo "--- MLM + SS-CL ---"
PT_EXP="${PREFIX}_pt_mlm_ss_cl"
run_pretrain_ss_cl "$PT_EXP" --mlm --ss_cl
run_finetune "$PT_EXP" "mlm_ss_cl"

# ── Config 3: MLM + NTP + SS-CL ──────────────────────────────────────────────
echo ""
echo "--- MLM + NTP + SS-CL (expected best) ---"
PT_EXP="${PREFIX}_pt_mlm_ntp_ss_cl"
run_pretrain_ss_cl "$PT_EXP" --mlm --ntp --ss_cl
run_finetune "$PT_EXP" "mlm_ntp_ss_cl"

echo ""
echo "=============================================="
echo "  SS-CL Complete: $RAN ran, $SKIPPED skipped / $TOTAL total"
echo "=============================================="
