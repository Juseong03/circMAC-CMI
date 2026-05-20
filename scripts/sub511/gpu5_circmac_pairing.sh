#!/bin/bash
# SUB511 — CircMAC + pairing pretrained, seeds 1 2 3
# Usage: ./scripts/sub511/gpu5_circmac_pairing.sh [GPU_ID]
set -e
GPU=${1:-5}; MODEL="circmac"; MAX_LEN=511; BS=128; PREFIX="sub511"
LR=1e-4; EPOCHS=150; EARLYSTOP=20; NUM_WORKERS=4; D_MODEL=128; N_LAYER=6

PT_PATH="saved_models/circmac/v2_ptm_pairing/42/pretrain/model.pth"

mkdir -p "logs/${PREFIX}" saved_models
TOTAL=0; SKIPPED=0; RAN=0
echo "=== SUB511 ${MODEL} pairing-pretrained (GPU $GPU) ==="

if [ ! -f "$PT_PATH" ]; then
    echo "[ERROR] Pretrained weights not found: $PT_PATH"
    exit 1
fi

for SEED in 1 2 3; do
    EXP="${PREFIX}_${MODEL}_pairing_s${SEED}"
    TOTAL=$((TOTAL+1))
    if find "saved_models/${MODEL}/${EXP}" -name "training.json" 2>/dev/null | grep -q .; then
        echo "[SKIP] $EXP"; SKIPPED=$((SKIPPED+1)); continue
    fi
    RAN=$((RAN+1)); echo "[RUN]  $EXP"
    python training.py \
        --model_name "$MODEL" --task sites --seed $SEED \
        --d_model $D_MODEL --n_layer $N_LAYER --max_len $MAX_LEN \
        --batch_size $BS --num_workers $NUM_WORKERS \
        --lr $LR --epochs $EPOCHS --earlystop $EARLYSTOP \
        --device $GPU --exp "$EXP" \
        --load_pretrained "$PT_PATH" \
        --interaction cross_attention --verbose \
        2>&1 | tee "logs/${PREFIX}/${EXP}.log"
done
echo "=== Done: $RAN ran, $SKIPPED skipped / $TOTAL total ==="
