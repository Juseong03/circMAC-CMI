#!/bin/bash
#===============================================================================
# EXP1 — RNA LM Frozen Comparison
#
# Models: rnabert, rnaernie, rnafm, rnamsm  (frozen backbone)
# Seeds:  1 2 3
# Exp naming: exp1_{model}_frozen_s{seed}
#
# GPU 요구사항:
#   A100/H100 권장 (rnabert/rnaernie: bs=128, rnafm/rnamsm: bs=64)
#   Mamba 미사용이므로 H100도 가능
#
# Usage: ./scripts/final_v2/run_exp1_rna_frozen.sh [GPU_ID]
#===============================================================================
set -e

GPU=${1:-0}
SEEDS=(1 2 3)
TASK="sites"
LR=1e-4; EPOCHS=150; EARLYSTOP=20; NUM_WORKERS=4
D_MODEL=128; N_LAYER=6

# 모델별 max_len (position embedding 한계)
declare -A MAX_LENS=( [rnabert]=438 [rnaernie]=511 [rnafm]=1022 [rnamsm]=1022 )
declare -A BATCH_SIZES=( [rnabert]=128 [rnaernie]=128 [rnafm]=64 [rnamsm]=64 )

mkdir -p logs/exp1 saved_models
TOTAL=0; SKIPPED=0; RAN=0

echo "=== EXP1 RNA LM Frozen (GPU $GPU) ==="

for MODEL in rnabert rnaernie rnafm rnamsm; do
    ML=${MAX_LENS[$MODEL]}
    BS=${BATCH_SIZES[$MODEL]}
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
