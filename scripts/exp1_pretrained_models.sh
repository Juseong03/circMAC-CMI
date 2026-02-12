#!/bin/bash
#===============================================================================
# Experiment 1: Pretrained RNA Model Comparison (Sites Prediction)
#
# Models: RNABERT, RNAFM, RNAErnie, RNAMSM vs CircMAC-PT
# Task: sites
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

# Hyperparameters (smaller batch for large pretrained models)
D_MODEL=128
N_LAYER=6
BATCH_SIZE=32  # Smaller for RNABERT, RNAFM, etc.
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
echo "Best pretrain config: $BEST_PT"
echo "Models: ${PRETRAINED_MODELS[*]} + CircMAC-PT"
echo "=============================================="

#-----------------------------------------------
# Part 1: Pretrained RNA Models
#-----------------------------------------------
echo ""
echo "=== Part 1: Pretrained RNA Models ==="
echo ""

TOTAL=$((${#PRETRAINED_MODELS[@]} * ${#SEEDS[@]}))
COUNT=0

for MODEL in "${PRETRAINED_MODELS[@]}"; do
    echo "Model: $MODEL"

    for SEED in "${SEEDS[@]}"; do
        COUNT=$((COUNT + 1))
        EXP_NAME="exp1_${MODEL}_${TASK}_s${SEED}"
        echo "  [$COUNT/$TOTAL] $EXP_NAME"

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
    echo "Completed: $MODEL"
    echo ""
done

#-----------------------------------------------
# Part 2: CircMAC with best pretraining
#-----------------------------------------------
echo ""
echo "=== Part 2: CircMAC-PT (${BEST_PT}) ==="
echo ""

PT_PATH="saved_models/circmac/exp2_pt_${BEST_PT}/best.pt"

if [ ! -f "$PT_PATH" ]; then
    echo "Warning: Pretrained model not found: $PT_PATH"
    echo "  Please run Experiment 2 first!"
    echo "  Skipping CircMAC-PT..."
else
    for SEED in "${SEEDS[@]}"; do
        EXP_NAME="exp1_circmac_pt_${TASK}_s${SEED}"
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
    echo "Completed: CircMAC-PT"
fi

echo ""
echo "=============================================="
echo "  Experiment 1 Complete!"
echo "=============================================="
