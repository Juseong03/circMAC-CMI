#!/bin/bash
# run_bsj_disjoint.sh
# Train CircMAC (v2_abl_full) on BSJ-disjoint split × 10 seeds
# Usage: bash scripts/final_v2/run_bsj_disjoint.sh <GPU>

GPU=${1:-0}
TRAIN_FILE="./data/df_train_bsj_disjoint.pkl"
TEST_FILE="./data/df_test_bsj_disjoint.pkl"

if [ -n "${SEEDS_OVERRIDE:-}" ]; then
    read -r -a SEEDS <<< "$SEEDS_OVERRIDE"
else
    SEEDS=(1 2 3 4 5 6 7 8 9 10)
fi

echo "=== BSJ-DISJOINT CircMAC training (GPU=$GPU) ==="
echo "    seeds: ${SEEDS[*]}"
echo "    train: $TRAIN_FILE"
echo "    test:  $TEST_FILE"

for SEED in "${SEEDS[@]}"; do
    EXP="bsj_circmac_s${SEED}"
    CKPT=$(find saved_models/circmac/${EXP} -name "model.pth" 2>/dev/null | head -1)
    if [ -n "$CKPT" ]; then
        echo "  [SKIP] $EXP — already done"
        continue
    fi
    echo "  [RUN]  $EXP (seed=$SEED)"
    python training.py \
        --model_name circmac \
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

echo "=== BSJ-DISJOINT done ==="
