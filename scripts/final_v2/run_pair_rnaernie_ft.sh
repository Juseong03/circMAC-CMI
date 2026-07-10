#!/bin/bash
# Train RNAErnie (trainable) on PAIR split x seeds
# Exp name: exp1_fair_trainable_rnaernie_s{seed}
# Usage: bash scripts/final_v2/run_pair_rnaernie_ft.sh <GPU>

GPU=${1:-0}

if [ -n "${SEEDS_OVERRIDE:-}" ]; then
    read -r -a SEEDS <<< "$SEEDS_OVERRIDE"
else
    SEEDS=(1 2 3)
fi

echo "=== PAIR rnaernie trainable (GPU=$GPU seeds=${SEEDS[*]}) ==="

for SEED in "${SEEDS[@]}"; do
    EXP="exp1_fair_trainable_rnaernie_s${SEED}"
    CKPT=$(find saved_models/rnaernie/${EXP} -name "model.pth" 2>/dev/null | head -1)
    if [ -n "$CKPT" ]; then echo "  [SKIP] $EXP"; continue; fi
    echo "  [RUN]  $EXP"
    python training.py \
        --model_name rnaernie \
        --device $GPU \
        --task sites \
        --seed $SEED \
        --d_model 128 \
        --n_layer 6 \
        --batch_size 8 \
        --interaction cross_attention \
        --max_len 511 \
        --trainable_pretrained \
        --verbose \
        --exp "$EXP"
done

echo "=== PAIR rnaernie done ==="
