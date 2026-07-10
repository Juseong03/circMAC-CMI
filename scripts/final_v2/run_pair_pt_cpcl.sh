#!/bin/bash
# Train CircMAC (pretrain=cpcl) on PAIR split x seeds
# Requires: saved_models/circmac/v2_ptm_cpcl/42/pretrain/model.pth
# Exp name: v2_pt_cpcl_s{seed}
# Usage: bash scripts/final_v2/run_pair_pt_cpcl.sh <GPU>

GPU=${1:-0}
PT_PATH="saved_models/circmac/v2_ptm_cpcl/42/pretrain/model.pth"

if [ -n "${SEEDS_OVERRIDE:-}" ]; then
    read -r -a SEEDS <<< "$SEEDS_OVERRIDE"
else
    SEEDS=(1 2 3)
fi

if [ ! -f "$PT_PATH" ]; then
    echo "[ERROR] Pretrained checkpoint not found: $PT_PATH"
    exit 1
fi

echo "=== PAIR CircMAC (pt=cpcl) GPU=$GPU seeds=${SEEDS[*]} ==="

for SEED in "${SEEDS[@]}"; do
    EXP="v2_pt_cpcl_s${SEED}"
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
        --load_pretrained "$PT_PATH" \
        --verbose \
        --exp "$EXP"
done

echo "=== PAIR CircMAC pt=cpcl done ==="
