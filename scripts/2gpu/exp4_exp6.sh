#!/bin/bash
#===============================================================================
# Experiment 4 + 6: CircMAC Ablation + Site Head Structure
#
# Exp4: 9 ablation configs × 3 seeds (27 runs)
# Exp6: 2 head types × 3 seeds (6 runs)
#
# Runs: 33
# Usage: ./scripts/2gpu/exp4_exp6.sh [GPU_ID]
#===============================================================================

set -e

GPU=${1:-1}
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

mkdir -p logs/exp4
mkdir -p logs/exp6
mkdir -p saved_models

#-----------------------------------------------
# Experiment 4: CircMAC Ablation Study (27 runs)
#-----------------------------------------------
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

echo "=============================================="
echo "  Experiment 4: CircMAC Ablation"
echo "  GPU: $GPU | Runs: 27"
echo "=============================================="

for ABLATION in "${ABLATIONS[@]}"; do
    CONFIG_NAME="${ABLATION%%:*}"
    CONFIG_FLAGS="${ABLATION#*:}"

    echo ""
    echo "Configuration: $CONFIG_NAME"
    [ -n "$CONFIG_FLAGS" ] && echo "  Flags: $CONFIG_FLAGS"

    for SEED in "${SEEDS[@]}"; do
        EXP_NAME="exp4_${CONFIG_NAME}_s${SEED}"
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
            --interaction cross_attention \
            --verbose \
            $CONFIG_FLAGS \
            2>&1 | tee "logs/exp4/${EXP_NAME}.log"
    done
    echo "Completed: $CONFIG_NAME"
done

echo ""
echo "  Experiment 4 Complete! [27 runs]"
echo ""

#-----------------------------------------------
# Experiment 6: Site Head Structure (6 runs)
#-----------------------------------------------
HEAD_TYPES=("conv1d" "linear")

echo "=============================================="
echo "  Experiment 6: Site Head Structure"
echo "  GPU: $GPU | Runs: 6"
echo "=============================================="

for HEAD_TYPE in "${HEAD_TYPES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        EXP_NAME="exp6_${HEAD_TYPE}_s${SEED}"
        echo "  $EXP_NAME"

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
echo "  Exp4 + Exp6 Complete! [33 runs]"
echo "=============================================="
