#!/bin/bash
# RNAMSM fine-tuned — iso + bsj, seed=1
# Usage: bash scripts/final_v2/bs64_s1/run_rnamsm.sh <GPU>

GPU=${1:-0}; SEED=1

for SPLIT in iso bsj; do
    TRAIN_FILE="./data/df_train_${SPLIT}_disjoint.pkl"
    TEST_FILE="./data/df_test_${SPLIT}_disjoint.pkl"
    EXP="${SPLIT}_rnamsm_ft_s${SEED}"

    if find saved_models/rnamsm/${EXP} -name "model.pth" 2>/dev/null | grep -q .; then
        echo "  [SKIP] $EXP"; continue
    fi
    echo "  [RUN]  $EXP"
    python training.py --model_name rnamsm --device $GPU --task sites \
        --seed $SEED --d_model 128 --n_layer 6 --max_len 1022 \
        --batch_size 8 --interaction cross_attention \
        --trainable_pretrained \
        --train_file "$TRAIN_FILE" --test_file "$TEST_FILE" \
        --verbose --exp "$EXP"
done

echo "=== rnamsm done ==="
