#!/bin/bash
#===============================================================================
# Experiment 5: Interaction Mechanism Comparison (Sites Prediction)
#
# Mechanisms: concat, elementwise, cross_attention
# Model: CircMAC
# Task: sites
# Seeds: 1, 2, 3
#
# Usage: ./scripts/exp5_interaction.sh [GPU_ID]
#===============================================================================

set -e

# Configuration
GPU=${1:-0}
SEEDS=(1 2 3)
TASK="sites"
INTERACTIONS=("concat" "elementwise" "cross_attention")

# Hyperparameters
D_MODEL=128
N_LAYER=6
BATCH_SIZE=128
EPOCHS=150
EARLYSTOP=20
LR=1e-4
NUM_WORKERS=4

# Create directories
mkdir -p logs/exp5
mkdir -p saved_models

echo "=============================================="
echo "  Experiment 5: Interaction Mechanism"
echo "=============================================="
echo "GPU: $GPU"
echo "Task: $TASK"
echo "Interactions: ${INTERACTIONS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "=============================================="

TOTAL=$((${#INTERACTIONS[@]} * ${#SEEDS[@]}))
COUNT=0

for INTERACTION in "${INTERACTIONS[@]}"; do
    echo ""
    echo "Interaction: $INTERACTION"

    for SEED in "${SEEDS[@]}"; do
        COUNT=$((COUNT + 1))
        EXP_NAME="exp5_${INTERACTION}_s${SEED}"

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
            --interaction "$INTERACTION" \
            --verbose \
            2>&1 | tee "logs/exp5/${EXP_NAME}.log"
    done
    echo "Completed: $INTERACTION"
done

echo ""
echo "=============================================="
echo "  Experiment 5 Complete!"
echo "  Total runs: $TOTAL"
echo "=============================================="
