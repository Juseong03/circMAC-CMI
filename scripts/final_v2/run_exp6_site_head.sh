#!/bin/bash
#===============================================================================
# EXP6 — Site Prediction Head Comparison
#
# exp6_conv1d  : Conv1D head (best)
# exp6_linear  : Linear head
#
# Seeds: 1 2 3
# Usage: ./scripts/final_v2/run_exp6_site_head.sh [GPU_ID]
#===============================================================================
set -e

GPU=${1:-0}
SEEDS=(1 2 3)
TASK="sites"
LR=1e-4; EPOCHS=150; EARLYSTOP=20; NUM_WORKERS=4
D_MODEL=128; N_LAYER=6; MAX_LEN=1022; BS=128

mkdir -p logs/exp6 saved_models
TOTAL=0; SKIPPED=0; RAN=0

echo "=== EXP6 Site Head (GPU $GPU) ==="

for HEAD in conv1d linear; do
    for SEED in "${SEEDS[@]}"; do
        EXP="exp6_${HEAD}_s${SEED}"
        TOTAL=$((TOTAL+1))
        if find "saved_models/circmac/${EXP}" -name "training.json" 2>/dev/null | grep -q .; then
            echo "[SKIP] $EXP"; SKIPPED=$((SKIPPED+1)); continue; fi
        RAN=$((RAN+1)); echo "[RUN]  $EXP"
        python training.py \
            --model_name circmac --task $TASK --seed $SEED \
            --d_model $D_MODEL --n_layer $N_LAYER --max_len $MAX_LEN \
            --batch_size $BS --num_workers $NUM_WORKERS \
            --lr $LR --epochs $EPOCHS --earlystop $EARLYSTOP \
            --device $GPU --exp "$EXP" \
            --interaction cross_attention \
            --site_head_type "$HEAD" --verbose \
            2>&1 | tee "logs/exp6/${EXP}.log"
    done
done

echo "=== Done: $RAN ran, $SKIPPED skipped / $TOTAL total ==="
