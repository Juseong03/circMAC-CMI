#!/bin/bash
# RNA-LMs fine-tuned (full length) — iso + bsj, seed=1
# batch_size=8 (same as pair split, no change needed)
# Usage: bash scripts/final_v2/bs64_s1/run_rna_lm_ft.sh <GPU>

GPU=${1:-0}
SEED=1

for SPLIT in iso bsj; do
    TRAIN_FILE="./data/df_train_${SPLIT}_disjoint.pkl"
    TEST_FILE="./data/df_test_${SPLIT}_disjoint.pkl"

    # RNABERT (max_len=438 native)
    EXP="${SPLIT}_rnabert_ft_s${SEED}"
    if find saved_models/rnabert/${EXP} -name "model.pth" 2>/dev/null | grep -q .; then
        echo "  [SKIP] $EXP"
    else
        echo "  [RUN]  $EXP"
        python training.py --model_name rnabert --device $GPU --task sites \
            --seed $SEED --d_model 128 --n_layer 6 --max_len 1022 \
            --batch_size 8 --interaction cross_attention \
            --trainable_pretrained \
            --train_file "$TRAIN_FILE" --test_file "$TEST_FILE" \
            --verbose --exp "$EXP"
    fi

    # RNAErnie (max_len=511 native)
    EXP="${SPLIT}_rnaernie_ft_s${SEED}"
    if find saved_models/rnaernie/${EXP} -name "model.pth" 2>/dev/null | grep -q .; then
        echo "  [SKIP] $EXP"
    else
        echo "  [RUN]  $EXP"
        python training.py --model_name rnaernie --device $GPU --task sites \
            --seed $SEED --d_model 128 --n_layer 6 --max_len 1022 \
            --batch_size 8 --interaction cross_attention \
            --trainable_pretrained \
            --train_file "$TRAIN_FILE" --test_file "$TEST_FILE" \
            --verbose --exp "$EXP"
    fi

    # RNAMSM
    EXP="${SPLIT}_rnamsm_ft_s${SEED}"
    if find saved_models/rnamsm/${EXP} -name "model.pth" 2>/dev/null | grep -q .; then
        echo "  [SKIP] $EXP"
    else
        echo "  [RUN]  $EXP"
        python training.py --model_name rnamsm --device $GPU --task sites \
            --seed $SEED --d_model 128 --n_layer 6 --max_len 1022 \
            --batch_size 8 --interaction cross_attention \
            --trainable_pretrained \
            --train_file "$TRAIN_FILE" --test_file "$TEST_FILE" \
            --verbose --exp "$EXP"
    fi

    # RNA-FM
    EXP="${SPLIT}_rnafm_ft_s${SEED}"
    if find saved_models/rnafm/${EXP} -name "model.pth" 2>/dev/null | grep -q .; then
        echo "  [SKIP] $EXP"
    else
        echo "  [RUN]  $EXP"
        python training.py --model_name rnafm --device $GPU --task sites \
            --seed $SEED --d_model 128 --n_layer 6 --max_len 1022 \
            --batch_size 8 --interaction cross_attention \
            --trainable_pretrained \
            --train_file "$TRAIN_FILE" --test_file "$TEST_FILE" \
            --verbose --exp "$EXP"
    fi

done

echo "=== RNA-LM ft (full-length) bs1 s1 done ==="
