#!/bin/bash
# RNA-FM fine-tuned — iso + bsj, seed=1
# Usage: bash scripts/final_v2/bs64_s1/run_rnafm.sh <GPU>

GPU=${1:-0}; SEED=1

for SPLIT in iso bsj; do
    TRAIN_FILE="./data/df_train_${SPLIT}_disjoint.pkl"
    TEST_FILE="./data/df_test_${SPLIT}_disjoint.pkl"
    EXP="${SPLIT}_rnafm_ft_bs64_s${SEED}"

    if find saved_models/rnafm/${EXP} -name "model.pth" 2>/dev/null | grep -q .; then
        echo "  [SKIP] $EXP"; continue
    fi
    echo "  [RUN]  $EXP"
    python training.py --model_name rnafm --device $GPU --task sites \
        --seed $SEED --d_model 128 --n_layer 6 --max_len 1022 \
        --batch_size 8 --interaction cross_attention \
        --trainable_pretrained \
        --train_file "$TRAIN_FILE" --test_file "$TEST_FILE" \
        --verbose --exp "$EXP"
done

echo "=== rnafm done ==="
