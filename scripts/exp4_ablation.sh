#!/bin/bash
#===============================================================================
# Experiment 4: CircMAC Ablation Study (Sites Prediction)
#
# Analyze component contributions of CircMAC:
# - Attention branch
# - Mamba branch
# - CNN branch
# - Circular features (bias, padding)
#
# Usage: ./scripts/exp4_ablation.sh [GPU_ID]
#===============================================================================

set -e

# Configuration
GPU=${1:-0}
SEEDS=(1 2 3)
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
mkdir -p logs/exp4
mkdir -p saved_models

echo "=============================================="
echo "  Experiment 4: CircMAC Ablation Study"
echo "=============================================="
echo "GPU: $GPU"
echo "Task: $TASK"
echo "=============================================="

# Define ablation configurations
# Format: "name:flags"
ABLATIONS=(
    "full:"
    "no_attn:--no_attn"
    "no_mamba:--no_mamba"
    "no_conv:--no_conv"
    "no_circular_bias:--no_circular_rel_bias"
    "no_circular_pad:--no_circular_window"
    "attn_only:--no_mamba --no_conv"
    "mamba_only:--no_attn --no_conv"
    "cnn_only:--no_attn --no_mamba"
)

TOTAL=$((${#ABLATIONS[@]} * ${#SEEDS[@]}))
COUNT=0

for ABLATION in "${ABLATIONS[@]}"; do
    # Parse name and flags
    CONFIG_NAME="${ABLATION%%:*}"
    CONFIG_FLAGS="${ABLATION#*:}"

    echo ""
    echo "Configuration: $CONFIG_NAME"
    [ -n "$CONFIG_FLAGS" ] && echo "  Flags: $CONFIG_FLAGS"

    for SEED in "${SEEDS[@]}"; do
        COUNT=$((COUNT + 1))
        EXP_NAME="exp4_${CONFIG_NAME}_s${SEED}"

        echo "  [$COUNT/$TOTAL] $EXP_NAME"

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
            --interaction cross_attention \
            --verbose \
            $CONFIG_FLAGS \
            2>&1 | tee "logs/exp4/${EXP_NAME}.log"
    done
    echo "Completed: $CONFIG_NAME"
done

echo ""
echo "=============================================="
echo "  Experiment 4 Complete!"
echo "  Total runs: $TOTAL"
echo "=============================================="
