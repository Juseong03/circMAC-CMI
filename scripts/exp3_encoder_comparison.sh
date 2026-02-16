#!/bin/bash
#===============================================================================
# Experiment 3: Encoder Architecture Comparison (Sites Prediction)
#
# Models: lstm, transformer, mamba, hymba, circmac
# Task: sites
# Seeds: 1, 2, 3
#
# Usage: ./scripts/exp3_encoder_comparison.sh [GPU_ID]
#===============================================================================

set -e  # Exit on error

# Configuration
GPU=${1:-0}
SEEDS=(1 2 3)
MODELS=("lstm" "transformer" "mamba" "hymba" "circmac")
TASK="sites"

# Hyperparameters
D_MODEL=128
N_LAYER=6
BATCH_SIZE=128
EPOCHS=150
EARLYSTOP=20
LR=1e-4
NUM_WORKERS=4

# Create directories
mkdir -p logs/exp3
mkdir -p saved_models

echo "=============================================="
echo "  Experiment 3: Encoder Architecture Comparison"
echo "=============================================="
echo "GPU: $GPU"
echo "Models: ${MODELS[*]}"
echo "Task: $TASK"
echo "Seeds: ${SEEDS[*]}"
echo "=============================================="

TOTAL=$((${#MODELS[@]} * ${#SEEDS[@]}))
COUNT=0

for MODEL in "${MODELS[@]}"; do
    # Transformer needs smaller batch size to avoid OOM
    if [ "$MODEL" = "transformer" ]; then
        BS=64
    else
        BS=$BATCH_SIZE
    fi

    for SEED in "${SEEDS[@]}"; do
        COUNT=$((COUNT + 1))
        EXP_NAME="exp3_${MODEL}_${TASK}_s${SEED}"

        echo ""
        echo "[$COUNT/$TOTAL] $EXP_NAME (batch_size=$BS)"
        echo "----------------------------------------------"

        python training.py \
            --model_name "$MODEL" \
            --task "$TASK" \
            --seed "$SEED" \
            --d_model "$D_MODEL" \
            --n_layer "$N_LAYER" \
            --batch_size "$BS" \
            --num_workers "$NUM_WORKERS" \
            --epochs "$EPOCHS" \
            --earlystop "$EARLYSTOP" \
            --lr "$LR" \
            --device "$GPU" \
            --exp "$EXP_NAME" \
            --interaction cross_attention \
            --verbose \
            2>&1 | tee "logs/exp3/${EXP_NAME}.log"

        echo "Completed: $EXP_NAME"
    done
done

echo ""
echo "=============================================="
echo "  Experiment 3 Complete!"
echo "  Total runs: $TOTAL"
echo "=============================================="
