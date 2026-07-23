#!/bin/bash
# Train CircMAC (pretrain=mlm_cpcl_ssp) on ISO-DISJOINT x 3 seeds
# Requires pretrained checkpoint: saved_models/circmac/v2_ptm_mlm_cpcl_ssp/42/pretrain/model.pth
# Usage: bash scripts/final_v2/run_iso_pt_mlm_cpcl_ssp.sh <GPU>

GPU=${1:-0}
TRAIN_FILE="./data/df_train_iso_disjoint.pkl"
TEST_FILE="./data/df_test_iso_disjoint.pkl"
PT_PATH="saved_models/circmac/v2_ptm_mlm_cpcl_ssp/42/pretrain/model.pth"

if [ -n "${SEEDS_OVERRIDE:-}" ]; then
    read -r -a SEEDS <<< "$SEEDS_OVERRIDE"
else
    SEEDS=(1 2 3)
fi

if [ ! -f "$PT_PATH" ]; then
    echo "[ERROR] Pretrained checkpoint not found: $PT_PATH"
    echo "        Run pretraining first (v2_ptm_mlm_cpcl_ssp)"
    exit 1
fi

echo "=== ISO-DISJOINT CircMAC (pt=mlm_cpcl_ssp) GPU=$GPU seeds=${SEEDS[*]} ==="

for SEED in "${SEEDS[@]}"; do
    EXP="iso_pt_mlm_cpcl_ssp_s${SEED}"
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
        --train_file "$TRAIN_FILE" \
        --test_file  "$TEST_FILE" \
        --exp "$EXP"
done

echo "=== ISO-DISJOINT CircMAC pt=mlm_cpcl_ssp done ==="
