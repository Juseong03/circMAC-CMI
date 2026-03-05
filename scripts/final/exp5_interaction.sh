#!/bin/bash
#===============================================================================
# Exp5: Interaction Mechanism Comparison (Supplementary)
# RQ: Which circRNA-miRNA interaction mechanism works best?
#
# Configs: concat, elementwise, cross_attention
# Runs:    3 × 3 seeds = 9 runs
# Usage:   ./scripts/final/exp5_interaction.sh [GPU_ID]
#===============================================================================
set -e

GPU=${1:-0}
SEEDS=(1 2 3)
TASK="sites"

D_MODEL=128; N_LAYER=6; LR=1e-4; EPOCHS=150; EARLYSTOP=20; BS=128; NUM_WORKERS=4

mkdir -p logs/exp5 saved_models

TOTAL=0; SKIPPED=0; RAN=0

echo "=============================================="
echo "  Exp5: Interaction Mechanism"
echo "  GPU: $GPU | Runs: 9"
echo "=============================================="

for INTERACTION in concat elementwise cross_attention; do
    for SEED in "${SEEDS[@]}"; do
        TOTAL=$((TOTAL + 1))
        EXP_NAME="exp5_${INTERACTION}_s${SEED}"
        if find "saved_models/circmac/${EXP_NAME}" -name "training.json" 2>/dev/null | grep -q .; then
            echo "[DONE] $EXP_NAME"; SKIPPED=$((SKIPPED + 1)); continue; fi
        RAN=$((RAN + 1))
        echo "[RUN]  $EXP_NAME"
        python training.py \
            --model_name circmac --task "$TASK" --seed "$SEED" \
            --d_model "$D_MODEL" --n_layer "$N_LAYER" \
            --batch_size "$BS" --num_workers "$NUM_WORKERS" \
            --lr "$LR" --epochs "$EPOCHS" --earlystop "$EARLYSTOP" \
            --device "$GPU" --exp "$EXP_NAME" \
            --interaction "$INTERACTION" --verbose \
            2>&1 | tee "logs/exp5/${EXP_NAME}.log"
    done
    echo "  Done: $INTERACTION"
done

echo ""
echo "=============================================="
echo "  Exp5 Complete: $RAN ran, $SKIPPED skipped / $TOTAL total"
echo "=============================================="
