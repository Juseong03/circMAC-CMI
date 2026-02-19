#!/bin/bash
#===============================================================================
# NEW Experiment 2: Pretraining Strategy (Bug-fixed version)
#
# Previous pretraining had a critical gradient bug (.item() detaching graph).
# This is the corrected version - ALL pretraining must be redone from scratch.
#
# Design:
#   Phase 0: No-pretrain baseline (3 runs)
#   Phase 1: Pretrain with 6 task configs (6 runs)
#   Phase 2: Finetune each pretrained model (18 runs)
#
# Total: 27 runs (3 baseline + 6 pretrain + 18 finetune)
#
# Pretraining HP (tuned for proper learning):
#   - Data: df_circ_ss_5 (5x augmented, 149K samples)
#   - batch_size: 128 (increased from 64)
#   - lr: 5e-4 (increased from 1e-4 for pretrain)
#   - epochs: 200, earlystop: 30
#   - optimizer: adamw, w_decay: 0.01
#
# Finetuning HP (same as other experiments):
#   - batch_size: 128, lr: 1e-4
#   - epochs: 150, earlystop: 20
#
# Usage: ./scripts/rerun/exp2_new.sh [GPU_ID]
#===============================================================================

set -e

GPU=${1:-0}
SEEDS=(1 2 3)
TASK="sites"
PT_SEED=42

# Pretraining hyperparameters
DATA_FILE="df_circ_ss_5"
PT_BATCH_SIZE=128
PT_LR=5e-4
PT_EPOCHS=200
PT_EARLYSTOP=30
PT_W_DECAY=0.01
D_MODEL=128
N_LAYER=6
MAX_LEN=1022
NUM_WORKERS=4

# Fine-tuning hyperparameters
FT_BATCH_SIZE=128
FT_LR=1e-4
FT_EPOCHS=150
FT_EARLYSTOP=20

# Experiment prefix (distinguish from old exp2)
PREFIX="exp2v2"

mkdir -p logs/exp2v2/pretrain
mkdir -p logs/exp2v2/finetune
mkdir -p saved_models

# Task configurations (6 configs)
# MLM is always the base task
declare -A PT_CONFIGS
PT_CONFIGS["mlm"]="--mlm"
PT_CONFIGS["mlm_ntp"]="--mlm --ntp"
PT_CONFIGS["mlm_ssp"]="--mlm --ssp"
PT_CONFIGS["mlm_cpcl"]="--mlm --cpcl"
PT_CONFIGS["mlm_pairing"]="--mlm --pairing"
PT_CONFIGS["mlm_ntp_cpcl_pair"]="--mlm --ntp --cpcl --pairing"

echo "=============================================="
echo "  NEW Exp2: Pretraining Strategy (bug-fixed)"
echo "  GPU: $GPU"
echo "  Total: 27 runs (3 baseline + 6 pretrain + 18 finetune)"
echo "=============================================="

#-----------------------------------------------
# Phase 0: No-pretrain baseline (3 runs)
#-----------------------------------------------
echo ""
echo "=== Phase 0: No-pretrain Baseline ==="

for SEED in "${SEEDS[@]}"; do
    EXP_NAME="${PREFIX}_nopt_${TASK}_s${SEED}"

    RESULT_DIR="saved_models/circmac/${EXP_NAME}"
    if find "$RESULT_DIR" -name "training.json" 2>/dev/null | grep -q .; then
        echo "[DONE] $EXP_NAME"
        continue
    fi

    echo "[RUN]  $EXP_NAME"
    python training.py \
        --model_name circmac \
        --task "$TASK" \
        --seed "$SEED" \
        --d_model "$D_MODEL" \
        --n_layer "$N_LAYER" \
        --batch_size "$FT_BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" \
        --lr "$FT_LR" \
        --epochs "$FT_EPOCHS" \
        --earlystop "$FT_EARLYSTOP" \
        --device "$GPU" \
        --exp "$EXP_NAME" \
        --interaction cross_attention \
        --verbose \
        2>&1 | tee "logs/exp2v2/finetune/${EXP_NAME}.log"
done
echo "Completed: No-pretrain baseline"

#-----------------------------------------------
# Phase 1: Pretraining (6 runs)
#-----------------------------------------------
echo ""
echo "=== Phase 1: Pretraining ==="

for CONFIG_NAME in "${!PT_CONFIGS[@]}"; do
    CONFIG_FLAGS="${PT_CONFIGS[$CONFIG_NAME]}"
    EXP_NAME="${PREFIX}_pt_${CONFIG_NAME}"

    # Check if pretrain model already exists
    PT_MODEL="saved_models/circmac/${EXP_NAME}/${PT_SEED}/pretrain/model.pth"
    if [ -f "$PT_MODEL" ]; then
        echo "[DONE] $EXP_NAME (model exists)"
        continue
    fi

    echo "[RUN]  $EXP_NAME ($CONFIG_FLAGS)"
    python pretraining.py \
        --model_name circmac \
        --data_file "$DATA_FILE" \
        --max_len "$MAX_LEN" \
        --d_model "$D_MODEL" \
        --n_layer "$N_LAYER" \
        --batch_size "$PT_BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" \
        --optimizer adamw \
        --lr "$PT_LR" \
        --epochs "$PT_EPOCHS" \
        --earlystop "$PT_EARLYSTOP" \
        --device "$GPU" \
        --exp "$EXP_NAME" \
        --seed "$PT_SEED" \
        --verbose \
        $CONFIG_FLAGS \
        2>&1 | tee "logs/exp2v2/pretrain/${EXP_NAME}.log"

    echo "Completed pretrain: $CONFIG_NAME"
done

#-----------------------------------------------
# Phase 2: Fine-tuning (18 runs)
#-----------------------------------------------
echo ""
echo "=== Phase 2: Fine-tuning ==="

for CONFIG_NAME in "${!PT_CONFIGS[@]}"; do
    PT_PATH="saved_models/circmac/${PREFIX}_pt_${CONFIG_NAME}/${PT_SEED}/pretrain/model.pth"

    if [ ! -f "$PT_PATH" ]; then
        echo "[SKIP] ${CONFIG_NAME}: pretrained model not found at $PT_PATH"
        continue
    fi

    for SEED in "${SEEDS[@]}"; do
        EXP_NAME="${PREFIX}_${CONFIG_NAME}_${TASK}_s${SEED}"

        RESULT_DIR="saved_models/circmac/${EXP_NAME}"
        if find "$RESULT_DIR" -name "training.json" 2>/dev/null | grep -q .; then
            echo "[DONE] $EXP_NAME"
            continue
        fi

        echo "[RUN]  $EXP_NAME"
        python training.py \
            --model_name circmac \
            --task "$TASK" \
            --seed "$SEED" \
            --d_model "$D_MODEL" \
            --n_layer "$N_LAYER" \
            --batch_size "$FT_BATCH_SIZE" \
            --num_workers "$NUM_WORKERS" \
            --lr "$FT_LR" \
            --epochs "$FT_EPOCHS" \
            --earlystop "$FT_EARLYSTOP" \
            --device "$GPU" \
            --exp "$EXP_NAME" \
            --load_pretrained "$PT_PATH" \
            --interaction cross_attention \
            --verbose \
            2>&1 | tee "logs/exp2v2/finetune/${EXP_NAME}.log"
    done
done

echo ""
echo "=============================================="
echo "  NEW Exp2 Complete!"
echo "  Results in: saved_models/circmac/${PREFIX}_*"
echo "=============================================="
