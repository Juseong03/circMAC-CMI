#!/bin/bash
#===============================================================================
# NEW Exp2v2 - Server 1, GPU 0
# Config: MLM (+ no-pretrain baseline)
# Runs: 1 pretrain + 3 finetune + 3 baseline = 7
#===============================================================================
set -e
GPU=${1:-0}
SEEDS=(1 2 3)
TASK="sites"; PT_SEED=42; PREFIX="exp2v2"

# HP
DATA_FILE="df_circ_ss_5"; D_MODEL=128; N_LAYER=6; MAX_LEN=1022; NUM_WORKERS=4
PT_BS=128; PT_LR=5e-4; PT_EP=200; PT_ES=30
FT_BS=128; FT_LR=1e-4; FT_EP=150; FT_ES=20

mkdir -p logs/exp2v2/pretrain logs/exp2v2/finetune saved_models

CONFIG_NAME="mlm"
CONFIG_FLAGS="--mlm"

echo "=== Exp2v2 Server1-GPU0: Baseline + MLM ==="

# Baseline (no pretrain)
echo "--- Baseline ---"
for SEED in "${SEEDS[@]}"; do
    EXP_NAME="${PREFIX}_nopt_${TASK}_s${SEED}"
    if find "saved_models/circmac/${EXP_NAME}" -name "training.json" 2>/dev/null | grep -q .; then
        echo "[DONE] $EXP_NAME"; continue; fi
    echo "[RUN]  $EXP_NAME"
    python training.py --model_name circmac --task "$TASK" --seed "$SEED" \
        --d_model $D_MODEL --n_layer $N_LAYER --batch_size $FT_BS --num_workers $NUM_WORKERS \
        --lr $FT_LR --epochs $FT_EP --earlystop $FT_ES --device $GPU \
        --exp "$EXP_NAME" --interaction cross_attention --verbose \
        2>&1 | tee "logs/exp2v2/finetune/${EXP_NAME}.log"
done

# Pretrain
echo "--- Pretrain: $CONFIG_NAME ---"
EXP_NAME="${PREFIX}_pt_${CONFIG_NAME}"
PT_MODEL="saved_models/circmac/${EXP_NAME}/${PT_SEED}/pretrain/model.pth"
if [ -f "$PT_MODEL" ]; then echo "[DONE] $EXP_NAME"
else
    echo "[RUN]  $EXP_NAME ($CONFIG_FLAGS)"
    python pretraining.py --model_name circmac --data_file $DATA_FILE --max_len $MAX_LEN \
        --d_model $D_MODEL --n_layer $N_LAYER --batch_size $PT_BS --num_workers $NUM_WORKERS \
        --optimizer adamw --lr $PT_LR --w_decay 0.01 \
        --epochs $PT_EP --earlystop $PT_ES --device $GPU \
        --exp "$EXP_NAME" --seed $PT_SEED --verbose $CONFIG_FLAGS \
        2>&1 | tee "logs/exp2v2/pretrain/${EXP_NAME}.log"
fi

# Finetune
echo "--- Finetune: $CONFIG_NAME ---"
PT_PATH="saved_models/circmac/${PREFIX}_pt_${CONFIG_NAME}/${PT_SEED}/pretrain/model.pth"
if [ ! -f "$PT_PATH" ]; then echo "[SKIP] model not found"; exit 1; fi
for SEED in "${SEEDS[@]}"; do
    EXP_NAME="${PREFIX}_${CONFIG_NAME}_${TASK}_s${SEED}"
    if find "saved_models/circmac/${EXP_NAME}" -name "training.json" 2>/dev/null | grep -q .; then
        echo "[DONE] $EXP_NAME"; continue; fi
    echo "[RUN]  $EXP_NAME"
    python training.py --model_name circmac --task "$TASK" --seed "$SEED" \
        --d_model $D_MODEL --n_layer $N_LAYER --batch_size $FT_BS --num_workers $NUM_WORKERS \
        --lr $FT_LR --epochs $FT_EP --earlystop $FT_ES --device $GPU \
        --exp "$EXP_NAME" --load_pretrained "$PT_PATH" \
        --interaction cross_attention --verbose \
        2>&1 | tee "logs/exp2v2/finetune/${EXP_NAME}.log"
done

echo "=== Server1-GPU0 Complete! ==="
