#!/bin/bash
#===============================================================================
# Exp6: Site Head Structure Comparison (Supplementary)
# RQ: Conv1D vs Linear head for token-level site prediction
#
# Runs: 2 types × 3 seeds = 6 runs
# Usage: ./scripts/final/exp6_site_head.sh [GPU_ID]
#===============================================================================
set -e

GPU=${1:-0}
SEEDS=(1 2 3)
TASK="sites"

D_MODEL=128; N_LAYER=6; LR=1e-4; EPOCHS=150; EARLYSTOP=20; BS=128; NUM_WORKERS=4

mkdir -p logs/exp6 saved_models

TOTAL=0; SKIPPED=0; RAN=0

echo "=============================================="
echo "  Exp6: Site Head Structure"
echo "  GPU: $GPU | Runs: 6"
echo "=============================================="

for HEAD in conv1d linear; do
    for SEED in "${SEEDS[@]}"; do
        TOTAL=$((TOTAL + 1))
        EXP_NAME="exp6_${HEAD}_s${SEED}"
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
            --interaction cross_attention \
            --site_head_type "$HEAD" --verbose \
            2>&1 | tee "logs/exp6/${EXP_NAME}.log"
    done
    echo "  Done: $HEAD"
done

echo ""
echo "=============================================="
echo "  Exp6 Complete: $RAN ran, $SKIPPED skipped / $TOTAL total"
echo "=============================================="
