#!/bin/bash
# Train RNA-MSM (trainable) on PAIR split x seeds
# Exp name: exp1_fair_trainable_rnamsm_s{seed}
# Usage: bash scripts/final_v2/run_pair_rnamsm_ft.sh <GPU>

GPU=${1:-0}

if [ -n "${SEEDS_OVERRIDE:-}" ]; then
    read -r -a SEEDS <<< "$SEEDS_OVERRIDE"
else
    SEEDS=(1 2 3)
fi

echo "=== PAIR rnamsm trainable (GPU=$GPU seeds=${SEEDS[*]}) ==="

for SEED in "${SEEDS[@]}"; do
    EXP="exp1_fair_trainable_rnamsm_s${SEED}"
    CKPT=$(find saved_models/rnamsm/${EXP} -name "model.pth" 2>/dev/null | head -1)
    if [ -n "$CKPT" ]; then echo "  [SKIP] $EXP"; continue; fi
    echo "  [RUN]  $EXP"
    python training.py \
        --model_name rnamsm \
        --device $GPU \
        --task sites \
        --seed $SEED \
        --d_model 128 \
        --n_layer 6 \
        --batch_size 8 \
        --interaction cross_attention \
        --max_len 1022 \
        --trainable_pretrained \
        --verbose \
        --exp "$EXP"
done

echo "=== PAIR rnamsm done ==="
