#!/bin/bash
#===============================================================================
# EXP5 — Interaction Mechanism Comparison
#
# exp5_cross_attn   : Cross-Attention (best)
# exp5_concat       : Concatenation
# exp5_elementwise  : Element-wise product
#
# Seeds: 1 2 3
# Usage: ./scripts/final_v2/run_exp5_interaction.sh [GPU_ID]
#===============================================================================
set -e

GPU=${1:-0}
SEEDS=(1 2 3)
TASK="sites"
LR=1e-4; EPOCHS=150; EARLYSTOP=20; NUM_WORKERS=4
D_MODEL=128; N_LAYER=6; MAX_LEN=1022; BS=128

mkdir -p logs/exp5 saved_models
TOTAL=0; SKIPPED=0; RAN=0

echo "=== EXP5 Interaction Mechanism (GPU $GPU) ==="

for INTERACTION in cross_attn concat elementwise; do
    # training.py에 넘기는 실제 값 (cross_attn → cross_attention)
    INTERACTION_ARG=$INTERACTION
    [ "$INTERACTION" = "cross_attn" ] && INTERACTION_ARG="cross_attention"

    for SEED in "${SEEDS[@]}"; do
        EXP="exp5_${INTERACTION}_s${SEED}"
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
            --interaction "$INTERACTION_ARG" --verbose \
            2>&1 | tee "logs/exp5/${EXP}.log"
    done
done

echo "=== Done: $RAN ran, $SKIPPED skipped / $TOTAL total ==="
