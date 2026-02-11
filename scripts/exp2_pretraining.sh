#!/bin/bash
#===============================================================================
# Experiment 2: Pretraining Strategy Comparison (Sites Prediction)
#
# Phase 1: Pretrain CircMAC with different task combinations
# Phase 2: Fine-tune on sites prediction
#
# Usage: ./scripts/exp2_pretraining.sh [GPU_ID]
#===============================================================================

set -e

# Configuration
GPU=${1:-0}
SEEDS=(1 2 3)
TASK="sites"

# Pretraining hyperparameters
PT_EPOCHS=300
PT_EARLYSTOP=30
PT_BATCH_SIZE=64
PT_LR=1e-4
D_MODEL=128
N_LAYER=6
MAX_LEN=1022
NUM_WORKERS=4

# Fine-tuning hyperparameters
FT_EPOCHS=150
FT_EARLYSTOP=20
FT_BATCH_SIZE=128
FT_LR=1e-4

# Create directories
mkdir -p logs/exp2/pretrain
mkdir -p logs/exp2/finetune
mkdir -p saved_models

echo "=============================================="
echo "  Experiment 2: Pretraining Strategy"
echo "=============================================="
echo "GPU: $GPU"
echo "Task: $TASK"
echo "=============================================="

#-----------------------------------------------
# Phase 1: Pretraining
#-----------------------------------------------
echo ""
echo "=== Phase 1: Pretraining ==="
echo ""

# Define pretraining configurations
declare -A PT_CONFIGS
PT_CONFIGS["mlm"]="--mlm"
PT_CONFIGS["mlm_ntp"]="--mlm --ntp"
PT_CONFIGS["mlm_ssp"]="--mlm --ssp"
PT_CONFIGS["mlm_pairing"]="--mlm --pairing"
PT_CONFIGS["mlm_cpcl"]="--mlm --cpcl"
PT_CONFIGS["mlm_bsj"]="--mlm --bsj_mlm"
PT_CONFIGS["mlm_cpcl_bsj"]="--mlm --cpcl --bsj_mlm"
PT_CONFIGS["mlm_cpcl_bsj_pair"]="--mlm --cpcl --bsj_mlm --pairing"
PT_CONFIGS["full"]="--mlm --ntp --ssp --ss_labels_multi --pairing --cpcl --bsj_mlm"

for CONFIG_NAME in "${!PT_CONFIGS[@]}"; do
    CONFIG_FLAGS="${PT_CONFIGS[$CONFIG_NAME]}"
    EXP_NAME="exp2_pt_${CONFIG_NAME}"

    echo "Pretraining: $CONFIG_NAME"
    echo "  Flags: $CONFIG_FLAGS"

    python pretraining.py \
        --model_name circmac \
        --data_file df_circ_ss \
        --max_len "$MAX_LEN" \
        --d_model "$D_MODEL" \
        --n_layer "$N_LAYER" \
        --batch_size "$PT_BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" \
        --lr "$PT_LR" \
        --epochs "$PT_EPOCHS" \
        --earlystop "$PT_EARLYSTOP" \
        --device "$GPU" \
        --exp "$EXP_NAME" \
        --verbose \
        $CONFIG_FLAGS \
        2>&1 | tee "logs/exp2/pretrain/${EXP_NAME}.log"

    echo "Pretrain completed: $CONFIG_NAME"
    echo ""
done

#-----------------------------------------------
# Phase 2: Fine-tuning
#-----------------------------------------------
echo ""
echo "=== Phase 2: Fine-tuning ==="
echo ""

# First: No pretraining baseline
echo "Fine-tuning: No pretraining (baseline)"
for SEED in "${SEEDS[@]}"; do
    EXP_NAME="exp2_nopt_${TASK}_s${SEED}"
    echo "  $EXP_NAME"

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
        2>&1 | tee "logs/exp2/finetune/${EXP_NAME}.log"
done

# Then: With each pretrained model
for CONFIG_NAME in "${!PT_CONFIGS[@]}"; do
    PT_PATH="saved_models/circmac/exp2_pt_${CONFIG_NAME}/best.pt"

    if [ ! -f "$PT_PATH" ]; then
        echo "Skipping $CONFIG_NAME: pretrained model not found"
        continue
    fi

    echo "Fine-tuning with pretrain: $CONFIG_NAME"

    for SEED in "${SEEDS[@]}"; do
        EXP_NAME="exp2_${CONFIG_NAME}_${TASK}_s${SEED}"
        echo "  $EXP_NAME"

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
            2>&1 | tee "logs/exp2/finetune/${EXP_NAME}.log"
    done
done

echo ""
echo "=============================================="
echo "  Experiment 2 Complete!"
echo "=============================================="
