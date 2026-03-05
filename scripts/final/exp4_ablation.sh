#!/bin/bash
#===============================================================================
# Exp4: CircMAC Ablation Study
# RQ: How does each component contribute to performance?
#
# Base: CircMAC (circular=False default, rel_bias=ON, attn+mamba+conv)
# Configs: 8 × 3 seeds = 24 runs
# Usage:  ./scripts/final/exp4_ablation.sh [GPU_ID]
#===============================================================================
set -e

GPU=${1:-0}
SEEDS=(1 2 3)
TASK="sites"

D_MODEL=128; N_LAYER=6; LR=1e-4; EPOCHS=150; EARLYSTOP=20; BS=128; NUM_WORKERS=4

mkdir -p logs/exp4 saved_models

TOTAL=0; SKIPPED=0; RAN=0

echo "=============================================="
echo "  Exp4: CircMAC Ablation Study"
echo "  GPU: $GPU | Runs: 24"
echo "=============================================="

# Config: (exp_suffix, extra_flags)
declare -a CONFIGS=(
    "full                   "
    "no_attn                --no_attn"
    "no_mamba               --no_mamba"
    "no_conv                --no_conv"
    "no_circ_bias           --no_circular_rel_bias"
    "attn_only              --no_mamba --no_conv"
    "mamba_only             --no_attn --no_conv"
    "cnn_only               --no_attn --no_mamba"
)

for cfg in "${CONFIGS[@]}"; do
    CONFIG_NAME=$(echo "$cfg" | awk '{print $1}')
    CONFIG_FLAGS=$(echo "$cfg" | cut -d' ' -f2-)
    # trim leading/trailing spaces
    CONFIG_FLAGS=$(echo "$CONFIG_FLAGS" | sed 's/^[[:space:]]*//')

    for SEED in "${SEEDS[@]}"; do
        TOTAL=$((TOTAL + 1))
        EXP_NAME="exp4_${CONFIG_NAME}_s${SEED}"

        if find "saved_models/circmac/${EXP_NAME}" -name "training.json" 2>/dev/null | grep -q .; then
            echo "[DONE] $EXP_NAME"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        RAN=$((RAN + 1))
        echo "[RUN]  $EXP_NAME  flags: $CONFIG_FLAGS"

        python training.py \
            --model_name circmac \
            --task "$TASK" \
            --seed "$SEED" \
            --d_model "$D_MODEL" \
            --n_layer "$N_LAYER" \
            --batch_size "$BS" \
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
    echo "  Done: $CONFIG_NAME"
done

echo ""
echo "=============================================="
echo "  Exp4 Complete: $RAN ran, $SKIPPED skipped / $TOTAL total"
echo "=============================================="
