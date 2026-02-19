#!/bin/bash
#===============================================================================
# NEW Exp2 - GPU 1: MLM+CPCL, MLM+Pairing, MLM+NTP+CPCL+Pairing
# Run this on GPU 1, and exp2_new_gpu0.sh on GPU 0
#
# Runs: 3 pretrain + 9 finetune = 12
#===============================================================================

set -e

GPU=${1:-1}
SEEDS=(1 2 3)
TASK="sites"
PT_SEED=42
PREFIX="exp2v2"

# Pretraining HP
DATA_FILE="df_circ_ss_5"
PT_BATCH_SIZE=128
PT_LR=5e-4
PT_EPOCHS=200
PT_EARLYSTOP=30
D_MODEL=128
N_LAYER=6
MAX_LEN=1022
NUM_WORKERS=4

# Fine-tuning HP
FT_BATCH_SIZE=128
FT_LR=1e-4
FT_EPOCHS=150
FT_EARLYSTOP=20

mkdir -p logs/exp2v2/pretrain logs/exp2v2/finetune saved_models

# This GPU's configs
declare -A PT_CONFIGS
PT_CONFIGS["mlm_cpcl"]="--mlm --cpcl"
PT_CONFIGS["mlm_pairing"]="--mlm --pairing"
PT_CONFIGS["mlm_ntp_cpcl_pair"]="--mlm --ntp --cpcl --pairing"

echo "=============================================="
echo "  NEW Exp2 (GPU 1): MLM+CPCL, MLM+Pair, Full"
echo "  GPU: $GPU | Runs: 12"
echo "=============================================="

# Phase 1: Pretrain
echo ""
echo "=== Phase 1: Pretraining ==="
for CONFIG_NAME in "${!PT_CONFIGS[@]}"; do
    CONFIG_FLAGS="${PT_CONFIGS[$CONFIG_NAME]}"
    EXP_NAME="${PREFIX}_pt_${CONFIG_NAME}"
    PT_MODEL="saved_models/circmac/${EXP_NAME}/${PT_SEED}/pretrain/model.pth"
    if [ -f "$PT_MODEL" ]; then
        echo "[DONE] $EXP_NAME"; continue
    fi
    echo "[RUN]  $EXP_NAME ($CONFIG_FLAGS)"
    python pretraining.py \
        --model_name circmac --data_file "$DATA_FILE" --max_len "$MAX_LEN" \
        --d_model "$D_MODEL" --n_layer "$N_LAYER" \
        --batch_size "$PT_BATCH_SIZE" --num_workers "$NUM_WORKERS" \
        --optimizer adamw --lr "$PT_LR" --w_decay 0.01 \
        --epochs "$PT_EPOCHS" --earlystop "$PT_EARLYSTOP" \
        --device "$GPU" --exp "$EXP_NAME" --seed "$PT_SEED" --verbose \
        $CONFIG_FLAGS \
        2>&1 | tee "logs/exp2v2/pretrain/${EXP_NAME}.log"
done

# Phase 2: Finetune
echo ""
echo "=== Phase 2: Fine-tuning ==="
for CONFIG_NAME in "${!PT_CONFIGS[@]}"; do
    PT_PATH="saved_models/circmac/${PREFIX}_pt_${CONFIG_NAME}/${PT_SEED}/pretrain/model.pth"
    if [ ! -f "$PT_PATH" ]; then
        echo "[SKIP] ${CONFIG_NAME}: pretrained model not found"; continue
    fi
    for SEED in "${SEEDS[@]}"; do
        EXP_NAME="${PREFIX}_${CONFIG_NAME}_${TASK}_s${SEED}"
        RESULT_DIR="saved_models/circmac/${EXP_NAME}"
        if find "$RESULT_DIR" -name "training.json" 2>/dev/null | grep -q .; then
            echo "[DONE] $EXP_NAME"; continue
        fi
        echo "[RUN]  $EXP_NAME"
        python training.py \
            --model_name circmac --task "$TASK" --seed "$SEED" \
            --d_model "$D_MODEL" --n_layer "$N_LAYER" \
            --batch_size "$FT_BATCH_SIZE" --num_workers "$NUM_WORKERS" \
            --lr "$FT_LR" --epochs "$FT_EPOCHS" --earlystop "$FT_EARLYSTOP" \
            --device "$GPU" --exp "$EXP_NAME" \
            --load_pretrained "$PT_PATH" \
            --interaction cross_attention --verbose \
            2>&1 | tee "logs/exp2v2/finetune/${EXP_NAME}.log"
    done
done

echo ""
echo "=== GPU 1 Complete! ==="
