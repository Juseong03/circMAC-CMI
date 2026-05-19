#!/bin/bash
#===============================================================================
# SUB438 — GPU6: CircMAC no pretrain  (seeds 1 2 3)
# Usage: ./scripts/sub438/gpu6_circmac_nopt.sh [GPU_ID]
#===============================================================================
set -e
GPU=${1:-6}

SEEDS=(1 2 3)
TASK="sites"
MODEL="circmac"
MAX_LEN=438
BS=128
LR=1e-4; EPOCHS=150; EARLYSTOP=20; NUM_WORKERS=4
D_MODEL=128; N_LAYER=6
PREFIX="sub438"

mkdir -p "logs/${PREFIX}" saved_models
TOTAL=0; SKIPPED=0; RAN=0

echo "=== SUB438 ${MODEL} no-pretrain (GPU $GPU) ==="

for SEED in "${SEEDS[@]}"; do
    EXP="${PREFIX}_${MODEL}_nopt_s${SEED}"
    TOTAL=$((TOTAL+1))
    if find "saved_models/${MODEL}/${EXP}" -name "training.json" 2>/dev/null | grep -q .; then
        echo "[SKIP] $EXP"; SKIPPED=$((SKIPPED+1)); continue
    fi
    RAN=$((RAN+1)); echo "[RUN]  $EXP"
    python training.py \
        --model_name "$MODEL" --task $TASK --seed $SEED \
        --d_model $D_MODEL --n_layer $N_LAYER --max_len $MAX_LEN \
        --batch_size $BS --num_workers $NUM_WORKERS \
        --lr $LR --epochs $EPOCHS --earlystop $EARLYSTOP \
        --device $GPU --exp "$EXP" \
        --interaction cross_attention --verbose \
        2>&1 | tee "logs/${PREFIX}/${EXP}.log"
done

echo "=== Done: $RAN ran, $SKIPPED skipped / $TOTAL total ==="
