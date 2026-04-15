#!/bin/bash
#===============================================================================
# GPU 5 — EXP1 RNA Frozen: rnafm, rnamsm
#
# rnafm  (640M): max_len=1022, bs=64  × 3 seeds = 3 runs (~3h each = ~9h)
# rnamsm (100M): max_len=1022, bs=64  × 3 seeds = 3 runs (~2.5h each = ~7h)
# 합계: 6 runs, 예상 ~17h
#
# Usage: ./scripts/final_v2/run_gpu5_rna_frozen.sh [GPU_ID]
#===============================================================================
set -e

GPU=${1:-5}
SEEDS=(1 2 3)
TASK="sites"
LR=1e-4; EPOCHS=150; EARLYSTOP=20; NUM_WORKERS=4
D_MODEL=128; N_LAYER=6

mkdir -p logs/exp1 saved_models
TOTAL=0; SKIPPED=0; RAN=0

echo "=== GPU $GPU: EXP1 RNA Frozen (rnafm, rnamsm) ==="

declare -A ML_MAP=( [rnafm]=1022 [rnamsm]=1022 )
declare -A BS_MAP=( [rnafm]=64   [rnamsm]=64   )

for MODEL in rnafm rnamsm; do
    ML=${ML_MAP[$MODEL]}
    BS=${BS_MAP[$MODEL]}
    for SEED in "${SEEDS[@]}"; do
        EXP="exp1_${MODEL}_frozen_s${SEED}"
        TOTAL=$((TOTAL+1))
        if find "saved_models/${MODEL}/${EXP}" -name "training.json" 2>/dev/null | grep -q .; then
            echo "[SKIP] $EXP"; SKIPPED=$((SKIPPED+1)); continue; fi
        RAN=$((RAN+1)); echo "[RUN]  $EXP (max_len=$ML, bs=$BS)"
        python training.py \
            --model_name "$MODEL" --task $TASK --seed $SEED \
            --d_model $D_MODEL --n_layer $N_LAYER --max_len $ML \
            --batch_size $BS --num_workers $NUM_WORKERS \
            --lr $LR --epochs $EPOCHS --earlystop $EARLYSTOP \
            --device $GPU --exp "$EXP" \
            --interaction cross_attention --verbose \
            2>&1 | tee "logs/exp1/${EXP}.log"
    done
done

echo "=== Done: $RAN ran, $SKIPPED skipped / $TOTAL total ==="
