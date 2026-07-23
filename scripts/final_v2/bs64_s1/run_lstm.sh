#!/bin/bash
# LSTM — iso + bsj, batch_size=64, seed=1
# Usage: bash scripts/final_v2/bs64_s1/run_lstm.sh <GPU>

GPU=${1:-0}
SEED=1

for SPLIT in iso bsj; do
    TRAIN_FILE="./data/df_train_${SPLIT}_disjoint.pkl"
    TEST_FILE="./data/df_test_${SPLIT}_disjoint.pkl"
    EXP="${SPLIT}_lstm_bs64_s${SEED}"

    CKPT=$(find saved_models/lstm/${EXP} -name "model.pth" 2>/dev/null | head -1)
    if [ -n "$CKPT" ]; then echo "  [SKIP] $EXP"; continue; fi
    echo "  [RUN]  $EXP"

    python training.py \
        --model_name lstm \
        --device $GPU \
        --task sites \
        --seed $SEED \
        --d_model 128 \
        --n_layer 6 \
        --batch_size 64 \
        --interaction cross_attention \
        --max_len 1022 \
        --train_file "$TRAIN_FILE" \
        --test_file  "$TEST_FILE" \
        --verbose \
        --exp "$EXP"
done

echo "=== lstm bs64 s1 done ==="
