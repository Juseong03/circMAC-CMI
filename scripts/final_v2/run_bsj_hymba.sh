#!/bin/bash
# Train hymba on BSJ-DISJOINT x 3 seeds
# Usage: bash scripts/final_v2/run_bsj_hymba.sh <GPU>

GPU=${1:-0}
TRAIN_FILE="./data/df_train_bsj_disjoint.pkl"
TEST_FILE="./data/df_test_bsj_disjoint.pkl"

if [ -n "${SEEDS_OVERRIDE:-}" ]; then
    read -r -a SEEDS <<< "$SEEDS_OVERRIDE"
else
    SEEDS=(1 2 3)
fi

echo "=== BSJ-DISJOINT hymba (GPU=$GPU seeds=${SEEDS[*]}) ==="

for SEED in "${SEEDS[@]}"; do
    EXP="bsj_hymba_s${SEED}"
    CKPT=$(find saved_models/hymba/${EXP} -name "model.pth" 2>/dev/null | head -1)
    if [ -n "$CKPT" ]; then echo "  [SKIP] $EXP"; continue; fi
    echo "  [RUN]  $EXP"
    python training.py \
        --model_name hymba \
        --device $GPU \
        --task sites \
        --seed $SEED \
        --d_model 128 \
        --n_layer 6 \
        --batch_size 128 \
        --interaction cross_attention \
        --max_len 1022 \
        --train_file "$TRAIN_FILE" \
        --test_file  "$TEST_FILE" \
        --verbose \
        --exp "$EXP"
done

echo "=== BSJ-DISJOINT hymba done ==="
