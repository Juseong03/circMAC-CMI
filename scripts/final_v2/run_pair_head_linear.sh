#!/bin/bash
# Train CircMAC (site_head=linear) on PAIR split x seeds
# Exp name: v2_head_linear_s{seed}
# Usage: bash scripts/final_v2/run_pair_head_linear.sh <GPU>

GPU=${1:-0}

if [ -n "${SEEDS_OVERRIDE:-}" ]; then
    read -r -a SEEDS <<< "$SEEDS_OVERRIDE"
else
    SEEDS=(1 2 3)
fi

echo "=== PAIR CircMAC head=linear (GPU=$GPU seeds=${SEEDS[*]}) ==="

for SEED in "${SEEDS[@]}"; do
    EXP="v2_head_linear_s${SEED}"
    CKPT=$(find saved_models/circmac/${EXP} -name "model.pth" 2>/dev/null | head -1)
    if [ -n "$CKPT" ]; then echo "  [SKIP] $EXP"; continue; fi
    echo "  [RUN]  $EXP"
    python training.py \
        --model_name circmac \
        --device $GPU \
        --task sites \
        --seed $SEED \
        --d_model 128 \
        --n_layer 6 \
        --batch_size 64 \
        --interaction cross_attention \
        --site_head_type linear \
        --max_len 1022 \
        --verbose \
        --exp "$EXP"
done

echo "=== PAIR CircMAC head=linear done ==="
