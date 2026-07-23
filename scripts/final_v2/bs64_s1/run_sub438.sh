#!/bin/bash
# sub438 (max_len=438) — iso + bsj, seed=1
# RNA-LMs: batch_size=32, CircMAC: batch_size=64
# Usage: bash scripts/final_v2/bs64_s1/run_sub438.sh <GPU>

GPU=${1:-0}
SEED=1
PT_PATH="saved_models/circmac/v2_ptm_pairing/42/pretrain/model.pth"

if [ ! -f "$PT_PATH" ]; then
    echo "[ERROR] Pretrained checkpoint not found: $PT_PATH"; exit 1
fi

for SPLIT in iso bsj; do
    TRAIN_FILE="./data/df_train_${SPLIT}_disjoint.pkl"
    TEST_FILE="./data/df_test_${SPLIT}_disjoint.pkl"
    PREFIX="sub438_${SPLIT}"

    # RNABERT (native max=438, reuse full-length exp if exists)
    EXP="${SPLIT}_rnabert_ft_s${SEED}"
    if find saved_models/rnabert/${EXP} -name "model.pth" 2>/dev/null | grep -q .; then
        echo "  [SKIP] $EXP (reuse full-length)"
    else
        EXP="${PREFIX}_rnabert_ft_s${SEED}"
        if find saved_models/rnabert/${EXP} -name "model.pth" 2>/dev/null | grep -q .; then
            echo "  [SKIP] $EXP"
        else
            echo "  [RUN]  $EXP"
            python training.py --model_name rnabert --device $GPU --task sites \
                --seed $SEED --d_model 128 --n_layer 6 --max_len 438 \
                --batch_size 32 --interaction cross_attention \
                --trainable_pretrained \
                --train_file "$TRAIN_FILE" --test_file "$TEST_FILE" \
                --verbose --exp "$EXP"
        fi
    fi

    # RNAErnie (max_len=438 subset)
    EXP="${PREFIX}_rnaernie_ft_s${SEED}"
    if find saved_models/rnaernie/${EXP} -name "model.pth" 2>/dev/null | grep -q .; then
        echo "  [SKIP] $EXP"
    else
        echo "  [RUN]  $EXP"
        python training.py --model_name rnaernie --device $GPU --task sites \
            --seed $SEED --d_model 128 --n_layer 6 --max_len 438 \
            --batch_size 32 --interaction cross_attention \
            --trainable_pretrained \
            --train_file "$TRAIN_FILE" --test_file "$TEST_FILE" \
            --verbose --exp "$EXP"
    fi

    # RNAMSM (max_len=438 subset)
    EXP="${PREFIX}_rnamsm_ft_s${SEED}"
    if find saved_models/rnamsm/${EXP} -name "model.pth" 2>/dev/null | grep -q .; then
        echo "  [SKIP] $EXP"
    else
        echo "  [RUN]  $EXP"
        python training.py --model_name rnamsm --device $GPU --task sites \
            --seed $SEED --d_model 128 --n_layer 6 --max_len 438 \
            --batch_size 32 --interaction cross_attention \
            --trainable_pretrained \
            --train_file "$TRAIN_FILE" --test_file "$TEST_FILE" \
            --verbose --exp "$EXP"
    fi

    # RNA-FM (max_len=438 subset)
    EXP="${PREFIX}_rnafm_ft_s${SEED}"
    if find saved_models/rnafm/${EXP} -name "model.pth" 2>/dev/null | grep -q .; then
        echo "  [SKIP] $EXP"
    else
        echo "  [RUN]  $EXP"
        python training.py --model_name rnafm --device $GPU --task sites \
            --seed $SEED --d_model 128 --n_layer 6 --max_len 438 \
            --batch_size 32 --interaction cross_attention \
            --trainable_pretrained \
            --train_file "$TRAIN_FILE" --test_file "$TEST_FILE" \
            --verbose --exp "$EXP"
    fi

    # CircMAC (max_len=438, batch_size=64)
    EXP="${PREFIX}_circmac_pairing_bs64_s${SEED}"
    if find saved_models/circmac/${EXP} -name "model.pth" 2>/dev/null | grep -q .; then
        echo "  [SKIP] $EXP"
    else
        echo "  [RUN]  $EXP"
        python training.py --model_name circmac --device $GPU --task sites \
            --seed $SEED --d_model 128 --n_layer 6 --max_len 438 \
            --batch_size 64 --interaction cross_attention \
            --load_pretrained "$PT_PATH" \
            --train_file "$TRAIN_FILE" --test_file "$TEST_FILE" \
            --verbose --exp "$EXP"
    fi

done

echo "=== sub438 bs64 s1 done ==="
