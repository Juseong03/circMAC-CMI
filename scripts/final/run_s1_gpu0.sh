#!/bin/bash
#===============================================================================
# Server1 GPU0: No PT baseline + MLM + NTP
# PT_BS=192 (3× from 64, GPU utilization 10GB→30GB / 40GB)
# PT_LR=1e-3  (2× LR scaling for BS increase)
# Usage: ./scripts/final/run_s1_gpu0.sh [GPU_ID]
#===============================================================================
set -e

GPU=${1:-0}
SEEDS=(1 2 3)
TASK="sites"; PT_SEED=42; PREFIX="exp2v4"

DATA_FILE="df_pretrain"; D_MODEL=128; N_LAYER=6; MAX_LEN=1022; NUM_WORKERS=4
PT_BS=192; PT_LR=1e-3; PT_WD=0.01; PT_EP=1000; PT_ES=100
FT_BS=32;  FT_LR=1e-4; FT_EP=150;  FT_ES=20

mkdir -p logs/exp2v4/pretrain logs/exp2v4/finetune saved_models
TOTAL=0; SKIPPED=0; RAN=0

echo "=== S1 GPU${GPU}: No PT + MLM + NTP  (PT_BS=${PT_BS}, LR=${PT_LR}) ==="

run_pretrain() {
    local PT_EXP=$1; shift
    TOTAL=$((TOTAL + 1))
    PT_MODEL="saved_models/circmac/${PT_EXP}/${PT_SEED}/pretrain/model.pth"
    if [ -f "$PT_MODEL" ]; then echo "[SKIP] pretrain: $PT_EXP"; SKIPPED=$((SKIPPED+1)); return 0; fi
    RAN=$((RAN+1)); echo "[RUN]  pretrain: $PT_EXP  (bs=$PT_BS)"
    python pretraining.py \
        --model_name circmac --data_file $DATA_FILE --max_len $MAX_LEN \
        --d_model $D_MODEL --n_layer $N_LAYER \
        --batch_size $PT_BS --num_workers $NUM_WORKERS \
        --optimizer adamw --lr $PT_LR --w_decay $PT_WD \
        --epochs $PT_EP --earlystop $PT_ES \
        --device $GPU --exp "$PT_EXP" --seed $PT_SEED --verbose "$@" \
        2>&1 | tee "logs/exp2v4/pretrain/${PT_EXP}.log"
}

run_finetune() {
    local PT_NAME=$1; local PT_PATH=$2
    for SEED in "${SEEDS[@]}"; do
        TOTAL=$((TOTAL+1))
        EXP="${PREFIX}_${PT_NAME}_${TASK}_s${SEED}"
        if find "saved_models/circmac/${EXP}" -name "training.json" 2>/dev/null | grep -q .; then
            echo "[SKIP] finetune: $EXP"; SKIPPED=$((SKIPPED+1)); continue; fi
        RAN=$((RAN+1)); echo "[RUN]  finetune: $EXP"
        FT_CMD="python training.py \
            --model_name circmac --task $TASK --seed $SEED \
            --d_model $D_MODEL --n_layer $N_LAYER --max_len $MAX_LEN \
            --batch_size $FT_BS --num_workers $NUM_WORKERS \
            --lr $FT_LR --epochs $FT_EP --earlystop $FT_ES \
            --device $GPU --exp $EXP --interaction cross_attention --verbose"
        [ "$PT_PATH" != "none" ] && FT_CMD="$FT_CMD --load_pretrained $PT_PATH"
        eval "$FT_CMD" 2>&1 | tee "logs/exp2v4/finetune/${EXP}.log"
    done
}

# ── [1/3] No Pretraining baseline ─────────────────────────────────────────────
echo "--- [1/3] No PT baseline ---"
run_finetune "nopt" "none"

# ── [2/3] MLM ─────────────────────────────────────────────────────────────────
echo "--- [2/3] MLM ---"
run_pretrain "${PREFIX}_pt_mlm" --mlm
PT_PATH="saved_models/circmac/${PREFIX}_pt_mlm/${PT_SEED}/pretrain/model.pth"
[ -f "$PT_PATH" ] && run_finetune "mlm" "$PT_PATH"

# ── [3/3] NTP ─────────────────────────────────────────────────────────────────
echo "--- [3/3] NTP ---"
run_pretrain "${PREFIX}_pt_ntp" --ntp
PT_PATH="saved_models/circmac/${PREFIX}_pt_ntp/${PT_SEED}/pretrain/model.pth"
[ -f "$PT_PATH" ] && run_finetune "ntp" "$PT_PATH"

echo "=== S1 GPU${GPU} Done: $RAN ran, $SKIPPED skipped / $TOTAL total ==="
