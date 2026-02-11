#!/bin/bash
#===============================================================================
# Experiment 6: Site Head Structure Comparison (Sites Prediction)
#
# Head types: conv1d (Conv1D classifier), linear (Linear classifier)
# Model: CircMAC
# Task: sites
# Seeds: 1, 2, 3
#
# Usage: ./scripts/exp6_site_head.sh [GPU_ID]
#===============================================================================

set -e

# Configuration
GPU=${1:-0}
SEEDS=(1 2 3)
TASK="sites"
HEAD_TYPES=("conv1d" "linear")

# Hyperparameters
D_MODEL=128
N_LAYER=6
BATCH_SIZE=128
EPOCHS=150
EARLYSTOP=20
LR=1e-4
NUM_WORKERS=4

# Create directories
mkdir -p logs/exp6
mkdir -p saved_models

echo "=============================================="
echo "  Experiment 6: Site Head Structure"
echo "=============================================="
echo "GPU: $GPU"
echo "Task: $TASK"
echo "Head types: ${HEAD_TYPES[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "=============================================="

TOTAL=$((${#HEAD_TYPES[@]} * ${#SEEDS[@]}))
COUNT=0

for HEAD_TYPE in "${HEAD_TYPES[@]}"; do
    echo ""
    echo "Site Head: $HEAD_TYPE"

    for SEED in "${SEEDS[@]}"; do
        COUNT=$((COUNT + 1))
        EXP_NAME="exp6_${HEAD_TYPE}_s${SEED}"

        echo "  [$COUNT/$TOTAL] $EXP_NAME"

        python training.py \
            --model_name circmac \
            --task "$TASK" \
            --seed "$SEED" \
            --d_model "$D_MODEL" \
            --n_layer "$N_LAYER" \
            --batch_size "$BATCH_SIZE" \
            --num_workers "$NUM_WORKERS" \
            --epochs "$EPOCHS" \
            --earlystop "$EARLYSTOP" \
            --lr "$LR" \
            --device "$GPU" \
            --exp "$EXP_NAME" \
            --interaction cross_attention \
            --site_head_type "$HEAD_TYPE" \
            --verbose \
            2>&1 | tee "logs/exp6/${EXP_NAME}.log"
    done
    echo "Completed: $HEAD_TYPE"
done

echo ""
echo "=============================================="
echo "  Experiment 6 Complete!"
echo "  Total runs: $TOTAL"
echo "=============================================="
