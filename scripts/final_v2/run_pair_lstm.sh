#!/bin/bash
# Train LSTM on PAIR-DISJOINT x seeds
# Encoder comparison — exp name: exp1_lstm_s{seed}
# Usage: bash scripts/final_v2/run_pair_lstm.sh <GPU>

GPU=${1:-0}

if [ -n "${SEEDS_OVERRIDE:-}" ]; then
    read -r -a SEEDS <<< "$SEEDS_OVERRIDE"
else
    SEEDS=(1 2 3)
fi

echo "=== PAIR LSTM (GPU=$GPU seeds=${SEEDS[*]}) ==="

for SEED in "${SEEDS[@]}"; do
    EXP="exp1_lstm_s${SEED}"
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
        --batch_size 128 \
        --interaction cross_attention \
        --max_len 1022 \
        --verbose \
        --exp "$EXP"
done

echo "=== PAIR LSTM done ==="
