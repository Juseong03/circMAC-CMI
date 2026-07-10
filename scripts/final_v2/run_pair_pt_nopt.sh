#!/bin/bash
# Train CircMAC (no pretrain) on PAIR split x seeds — pretraining comparison baseline
# Exp name: v2_pt_nopt_s{seed}  (same as run_final_s2_10seed.sh)
# Usage: bash scripts/final_v2/run_pair_pt_nopt.sh <GPU>

GPU=${1:-0}

if [ -n "${SEEDS_OVERRIDE:-}" ]; then
    read -r -a SEEDS <<< "$SEEDS_OVERRIDE"
else
    SEEDS=(1 2 3)
fi

echo "=== PAIR CircMAC NoPT (GPU=$GPU seeds=${SEEDS[*]}) ==="

for SEED in "${SEEDS[@]}"; do
    EXP="v2_pt_nopt_s${SEED}"
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
        --max_len 1022 \
        --verbose \
        --exp "$EXP"
done

echo "=== PAIR CircMAC NoPT done ==="
