#!/bin/bash
# Template — used by gen_exp2_scripts.py to generate remaining exp2 scripts
# Each script: 1 pretrain + 3 finetune = 4 runs
# Usage: ./scripts/final/exp2_s{N}_gpu{N}.sh [GPU_ID]

set -e
GPU=${1:-1}
SEEDS=(1 2 3)
TASK="sites"; PT_SEED=42; PREFIX="exp2"

DATA_FILE="df_circ_ss_5"; D_MODEL=128; N_LAYER=6; MAX_LEN=1022; NUM_WORKERS=4
PT_BS=64; PT_LR=5e-4; PT_WD=0.01; PT_EP=500; PT_ES=50
FT_BS=32; FT_LR=1e-4; FT_EP=150; FT_ES=20

mkdir -p logs/exp2/pretrain logs/exp2/finetune saved_models

TOTAL=0; SKIPPED=0; RAN=0
CONFIG_NAME="mlm_ntp"
CONFIG_FLAGS="--mlm --ntp"

echo "=== Exp2 Server1-GPU1: ${CONFIG_NAME^^} ==="

# ── Pretrain ─────────────────────────────────────────────────────────────────
echo "--- Pretrain: $CONFIG_NAME ---"
PT_EXP="${PREFIX}_pt_${CONFIG_NAME}"
PT_MODEL="saved_models/circmac/${PT_EXP}/${PT_SEED}/pretrain/model.pth"
TOTAL=$((TOTAL + 1))
if [ -f "$PT_MODEL" ]; then echo "[DONE] $PT_EXP"; SKIPPED=$((SKIPPED + 1))
else
    RAN=$((RAN + 1))
    echo "[RUN]  $PT_EXP ($CONFIG_FLAGS)"
    python pretraining.py --model_name circmac --data_file $DATA_FILE --max_len $MAX_LEN \
        --d_model $D_MODEL --n_layer $N_LAYER --batch_size $PT_BS --num_workers $NUM_WORKERS \
        --optimizer adamw --lr $PT_LR --w_decay $PT_WD \
        --epochs $PT_EP --earlystop $PT_ES --device $GPU \
        --exp "$PT_EXP" --seed $PT_SEED --verbose \
        $CONFIG_FLAGS \
        2>&1 | tee "logs/exp2/pretrain/${PT_EXP}.log"
fi

# ── Finetune ─────────────────────────────────────────────────────────────────
echo "--- Finetune: $CONFIG_NAME ---"
PT_PATH="saved_models/circmac/${PREFIX}_pt_${CONFIG_NAME}/${PT_SEED}/pretrain/model.pth"
if [ ! -f "$PT_PATH" ]; then echo "[SKIP] pretrain model not found: $PT_PATH"; exit 1; fi

for SEED in "${SEEDS[@]}"; do
    TOTAL=$((TOTAL + 1))
    EXP_NAME="${PREFIX}_${CONFIG_NAME}_${TASK}_s${SEED}"
    if find "saved_models/circmac/${EXP_NAME}" -name "training.json" 2>/dev/null | grep -q .; then
        echo "[DONE] $EXP_NAME"; SKIPPED=$((SKIPPED + 1)); continue; fi
    RAN=$((RAN + 1))
    echo "[RUN]  $EXP_NAME"
    python training.py --model_name circmac --task "$TASK" --seed "$SEED" \
        --d_model $D_MODEL --n_layer $N_LAYER --batch_size $FT_BS --num_workers $NUM_WORKERS \
        --lr $FT_LR --epochs $FT_EP --earlystop $FT_ES --device $GPU \
        --exp "$EXP_NAME" --load_pretrained "$PT_PATH" \
        --interaction cross_attention --verbose \
        2>&1 | tee "logs/exp2/finetune/${EXP_NAME}.log"
done

echo ""
echo "=== Server1-GPU1 Complete: $RAN ran, $SKIPPED skipped / $TOTAL total ==="
