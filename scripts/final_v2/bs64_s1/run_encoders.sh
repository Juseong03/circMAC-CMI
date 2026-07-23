#!/bin/bash
# Encoders (LSTM, Transformer, Mamba, Hymba) — iso + bsj, batch_size=64, seed=1
# Usage: bash scripts/final_v2/bs64_s1/run_encoders.sh <GPU>

GPU=${1:-0}; SEED=1

declare -A MODELS=(
    [lstm]=lstm
    [transformer]=transformer
    [mamba]=mamba
    [hymba]=hymba
)

for NAME in lstm transformer mamba hymba; do
    for SPLIT in iso bsj; do
        TRAIN_FILE="./data/df_train_${SPLIT}_disjoint.pkl"
        TEST_FILE="./data/df_test_${SPLIT}_disjoint.pkl"
        EXP="${SPLIT}_${NAME}_bs64_s${SEED}"

        if find saved_models/${NAME}/${EXP} -name "model.pth" 2>/dev/null | grep -q .; then
            echo "  [SKIP] $EXP"; continue
        fi
        echo "  [RUN]  $EXP"
        python training.py --model_name $NAME --device $GPU --task sites \
            --seed $SEED --d_model 128 --n_layer 6 --max_len 1022 \
            --batch_size 64 --interaction cross_attention \
            --train_file "$TRAIN_FILE" --test_file "$TEST_FILE" \
            --verbose --exp "$EXP"
    done
done

echo "=== encoders bs64 s1 done ==="
