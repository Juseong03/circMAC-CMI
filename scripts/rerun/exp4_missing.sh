#!/bin/bash
#===============================================================================
# RERUN: Exp4 Missing Ablation Configs
# Completed: full, no_attn, no_conv, no_mamba (4 configs × 3 seeds)
# Missing:   no_circular_bias, no_circular_pad, attn_only, mamba_only, cnn_only
#
# Runs: 15 (5 configs × 3 seeds)
# Usage: ./scripts/rerun/exp4_missing.sh [GPU_ID]
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
mkdir -p saved_models

# Only the missing ablation configs
ABLATIONS=(
    "no_circular_bias:--no_circular_rel_bias"
    "no_circular_pad:--no_circular_window"
    "attn_only:--no_mamba --no_conv"
    "mamba_only:--no_attn --no_conv"
    "cnn_only:--no_attn --no_mamba"
)

TOTAL=0
SKIPPED=0
RAN=0

echo "=============================================="
echo "  RERUN: Exp4 Missing Ablation Configs"
echo "  GPU: $GPU | Expected: 15 runs"
echo "=============================================="

for ABLATION in "${ABLATIONS[@]}"; do
    CONFIG_NAME="${ABLATION%%:*}"
    CONFIG_FLAGS="${ABLATION#*:}"

    echo ""
    echo "Configuration: $CONFIG_NAME ($CONFIG_FLAGS)"

    for SEED in "${SEEDS[@]}"; do
        TOTAL=$((TOTAL + 1))
        EXP_NAME="exp4_${CONFIG_NAME}_s${SEED}"

        # Check if already completed
        RESULT_DIR="saved_models/circmac/${EXP_NAME}"
        if find "$RESULT_DIR" -name "training.json" 2>/dev/null | grep -q .; then
            echo "[DONE] $EXP_NAME (already completed, skipping)"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        RAN=$((RAN + 1))
        echo "[RUN]  $EXP_NAME"

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
echo "  RERUN Complete: Exp4 Missing Ablation"
echo "  Total: $TOTAL | Ran: $RAN | Skipped: $SKIPPED"
echo "=============================================="
