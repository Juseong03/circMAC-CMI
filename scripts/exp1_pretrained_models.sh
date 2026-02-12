#!/bin/bash
#===============================================================================
# Experiment 1: Pretrained RNA Model Comparison (Sites Prediction)
#
# Part 1: Fair Comparison   - all models use max_len=440 (same data)
# Part 2: Max Performance   - each model uses its own max_len
# Each part: frozen + trainable pretrained models
#
# Models: RNABERT, RNAFM, RNAErnie, RNAMSM + CircMAC-PT
#
# Usage: ./scripts/exp1_pretrained_models.sh [GPU_ID] [BEST_PRETRAIN_CONFIG]
#===============================================================================

set -e

# Configuration
GPU=${1:-0}
BEST_PT=${2:-"ss5_mlm_cpcl_bsj_pair"}  # Best config from Exp 2 (ss5 = 5x stochastic SS)
TASK="sites"

SEEDS=(1 2 3)
PRETRAINED_MODELS=("rnabert" "rnafm" "rnaernie" "rnamsm")

# Fair comparison max_len (rnabert limit: 438 + 2 special tokens = 440)
FAIR_MAX_LEN=440

# Hyperparameters
D_MODEL=128
N_LAYER=6
BATCH_SIZE=32  # Smaller for large pretrained models
EPOCHS=150
EARLYSTOP=20
LR=1e-4
NUM_WORKERS=4

# Create directories
mkdir -p logs/exp1
mkdir -p saved_models

echo "=============================================="
echo "  Experiment 1: Pretrained Model Comparison"
echo "=============================================="
echo "GPU: $GPU"
echo "Task: $TASK"
echo "Fair max_len: $FAIR_MAX_LEN"
echo "Models: ${PRETRAINED_MODELS[*]} + CircMAC-PT"
echo "=============================================="

#-----------------------------------------------
# Part 1: Fair Comparison (max_len=440, same data for all)
#-----------------------------------------------
echo ""
echo "======================================="
echo "  Part 1: Fair Comparison (max_len=$FAIR_MAX_LEN)"
echo "======================================="
echo ""

# 1A: Frozen pretrained models
echo "--- 1A: Frozen Pretrained Models ---"
for MODEL in "${PRETRAINED_MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        EXP_NAME="exp1_fair_frozen_${MODEL}_s${SEED}"
        echo "  $EXP_NAME"

        python training.py \
            --model_name "$MODEL" \
            --task "$TASK" \
            --seed "$SEED" \
            --max_len "$FAIR_MAX_LEN" \
            --d_model "$D_MODEL" \
            --batch_size "$BATCH_SIZE" \
            --num_workers "$NUM_WORKERS" \
            --lr "$LR" \
            --epochs "$EPOCHS" \
            --earlystop "$EARLYSTOP" \
            --device "$GPU" \
            --exp "$EXP_NAME" \
            --interaction cross_attention \
            --verbose \
            2>&1 | tee "logs/exp1/${EXP_NAME}.log"
    done
    echo "Completed: $MODEL (frozen, fair)"
done

# 1B: Trainable pretrained models
echo ""
echo "--- 1B: Trainable Pretrained Models ---"
for MODEL in "${PRETRAINED_MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        EXP_NAME="exp1_fair_trainable_${MODEL}_s${SEED}"
        echo "  $EXP_NAME"

        python training.py \
            --model_name "$MODEL" \
            --task "$TASK" \
            --seed "$SEED" \
            --max_len "$FAIR_MAX_LEN" \
            --d_model "$D_MODEL" \
            --batch_size "$BATCH_SIZE" \
            --num_workers "$NUM_WORKERS" \
            --lr "$LR" \
            --epochs "$EPOCHS" \
            --earlystop "$EARLYSTOP" \
            --device "$GPU" \
            --exp "$EXP_NAME" \
            --interaction cross_attention \
            --trainable_pretrained \
            --verbose \
            2>&1 | tee "logs/exp1/${EXP_NAME}.log"
    done
    echo "Completed: $MODEL (trainable, fair)"
done

# 1C: CircMAC-PT (fair)
echo ""
echo "--- 1C: CircMAC-PT (fair) ---"
PT_PATH="saved_models/circmac/exp2_pt_${BEST_PT}/best.pt"
if [ ! -f "$PT_PATH" ]; then
    echo "Warning: Pretrained model not found: $PT_PATH"
    echo "  Please run Experiment 2 first! Skipping CircMAC-PT (fair)..."
else
    for SEED in "${SEEDS[@]}"; do
        EXP_NAME="exp1_fair_circmac_pt_s${SEED}"
        echo "  $EXP_NAME"

        python training.py \
            --model_name circmac \
            --task "$TASK" \
            --seed "$SEED" \
            --max_len "$FAIR_MAX_LEN" \
            --d_model "$D_MODEL" \
            --n_layer "$N_LAYER" \
            --batch_size "$BATCH_SIZE" \
            --num_workers "$NUM_WORKERS" \
            --lr "$LR" \
            --epochs "$EPOCHS" \
            --earlystop "$EARLYSTOP" \
            --device "$GPU" \
            --exp "$EXP_NAME" \
            --load_pretrained "$PT_PATH" \
            --interaction cross_attention \
            --verbose \
            2>&1 | tee "logs/exp1/${EXP_NAME}.log"
    done
    echo "Completed: CircMAC-PT (fair)"
fi

#-----------------------------------------------
# Part 2: Max Performance (model-specific max_len)
# Skip rnabert (same as fair: max_len=438)
#-----------------------------------------------
echo ""
echo "======================================="
echo "  Part 2: Max Performance (model-specific max_len)"
echo "======================================="
echo ""

MAX_MODELS=("rnafm" "rnaernie" "rnamsm")  # Skip rnabert (same as fair)

# 2A: Frozen pretrained models
echo "--- 2A: Frozen Pretrained Models (max performance) ---"
for MODEL in "${MAX_MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        EXP_NAME="exp1_max_frozen_${MODEL}_s${SEED}"
        echo "  $EXP_NAME"

        python training.py \
            --model_name "$MODEL" \
            --task "$TASK" \
            --seed "$SEED" \
            --d_model "$D_MODEL" \
            --batch_size "$BATCH_SIZE" \
            --num_workers "$NUM_WORKERS" \
            --lr "$LR" \
            --epochs "$EPOCHS" \
            --earlystop "$EARLYSTOP" \
            --device "$GPU" \
            --exp "$EXP_NAME" \
            --interaction cross_attention \
            --verbose \
            2>&1 | tee "logs/exp1/${EXP_NAME}.log"
    done
    echo "Completed: $MODEL (frozen, max)"
done

# 2B: Trainable pretrained models
echo ""
echo "--- 2B: Trainable Pretrained Models (max performance) ---"
for MODEL in "${MAX_MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        EXP_NAME="exp1_max_trainable_${MODEL}_s${SEED}"
        echo "  $EXP_NAME"

        python training.py \
            --model_name "$MODEL" \
            --task "$TASK" \
            --seed "$SEED" \
            --d_model "$D_MODEL" \
            --batch_size "$BATCH_SIZE" \
            --num_workers "$NUM_WORKERS" \
            --lr "$LR" \
            --epochs "$EPOCHS" \
            --earlystop "$EARLYSTOP" \
            --device "$GPU" \
            --exp "$EXP_NAME" \
            --interaction cross_attention \
            --trainable_pretrained \
            --verbose \
            2>&1 | tee "logs/exp1/${EXP_NAME}.log"
    done
    echo "Completed: $MODEL (trainable, max)"
done

# 2C: CircMAC-PT (max)
echo ""
echo "--- 2C: CircMAC-PT (max performance) ---"
if [ ! -f "$PT_PATH" ]; then
    echo "Warning: Pretrained model not found: $PT_PATH"
    echo "  Skipping CircMAC-PT (max)..."
else
    for SEED in "${SEEDS[@]}"; do
        EXP_NAME="exp1_max_circmac_pt_s${SEED}"
        echo "  $EXP_NAME"

        python training.py \
            --model_name circmac \
            --task "$TASK" \
            --seed "$SEED" \
            --d_model "$D_MODEL" \
            --n_layer "$N_LAYER" \
            --batch_size "$BATCH_SIZE" \
            --num_workers "$NUM_WORKERS" \
            --lr "$LR" \
            --epochs "$EPOCHS" \
            --earlystop "$EARLYSTOP" \
            --device "$GPU" \
            --exp "$EXP_NAME" \
            --load_pretrained "$PT_PATH" \
            --interaction cross_attention \
            --verbose \
            2>&1 | tee "logs/exp1/${EXP_NAME}.log"
    done
    echo "Completed: CircMAC-PT (max)"
fi

echo ""
echo "=============================================="
echo "  Experiment 1 Complete!"
echo "  Total runs: 48"
echo "=============================================="
