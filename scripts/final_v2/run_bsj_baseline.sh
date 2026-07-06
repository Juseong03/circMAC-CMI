#!/bin/bash
# run_bsj_baseline.sh
# Train strongest baselines (Hymba, Mamba) on BSJ-disjoint split × 10 seeds
# Usage: bash scripts/final_v2/run_bsj_baseline.sh <GPU>

GPU=${1:-0}
TRAIN_FILE="./data/df_train_bsj_disjoint.pkl"
TEST_FILE="./data/df_test_bsj_disjoint.pkl"

if [ -n "${SEEDS_OVERRIDE:-}" ]; then
    read -r -a SEEDS <<< "$SEEDS_OVERRIDE"
else
    SEEDS=(1 2 3 4 5 6 7 8 9 10)
fi

echo "=== BSJ-DISJOINT Baseline training (GPU=$GPU) ==="

for MODEL in hymba mamba; do
    for SEED in "${SEEDS[@]}"; do
        EXP="bsj_${MODEL}_s${SEED}"
        CKPT=$(find saved_models/${MODEL}/${EXP} -name "model.pth" 2>/dev/null | head -1)
        if [ -n "$CKPT" ]; then
            echo "  [SKIP] $EXP"
            continue
        fi
        echo "  [RUN]  $EXP"
        python training.py \
            --model_name $MODEL \
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
            --exp "$EXP"
    done
done

echo "=== BSJ-DISJOINT Baseline done ==="
