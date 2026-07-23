#!/bin/bash
# CircMAC (Pairing) — iso + bsj, batch_size=64, seed=1
# Usage: bash scripts/final_v2/bs64_s1/run_circmac_pairing.sh <GPU>

GPU=${1:-0}
SEED=1
PT_PATH="saved_models/circmac/v2_ptm_pairing/42/pretrain/model.pth"

if [ ! -f "$PT_PATH" ]; then
    echo "[ERROR] Pretrained checkpoint not found: $PT_PATH"
    exit 1
fi

for SPLIT in iso bsj; do
    TRAIN_FILE="./data/df_train_${SPLIT}_disjoint.pkl"
    TEST_FILE="./data/df_test_${SPLIT}_disjoint.pkl"
    EXP="${SPLIT}_pt_pairing_bs64_s${SEED}"

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
        --train_file "$TRAIN_FILE" \
        --test_file  "$TEST_FILE" \
        --verbose \
        --exp "$EXP"
done

echo "=== circmac_pairing bs64 s1 done ==="
