#!/bin/bash
#===============================================================================
# Repair Script S2: EXP4 ablation  → Server2 GPU1
# Usage: ./scripts/final/exp_repair_s2.sh [GPU_ID]
#===============================================================================
set -e

GPU=${1:-0}
SEEDS=(1 2 3)
TASK="sites"

D_MODEL=128; N_LAYER=6; LR=1e-4; EPOCHS=150; EARLYSTOP=20; NUM_WORKERS=4
BS=128

mkdir -p logs/exp4 saved_models
TOTAL=0; SKIPPED=0; RAN=0

echo "=== Repair S2: EXP4 Ablation (GPU $GPU) ==="

declare -a EXP4_REPAIRS=(
    "mamba_only  1  --no_attn --no_conv"
    "mamba_only  2  --no_attn --no_conv"
    "mamba_only  3  --no_attn --no_conv"
    "attn_only   1  --no_mamba --no_conv"
    "cnn_only    1  --no_attn --no_mamba"
)

for entry in "${EXP4_REPAIRS[@]}"; do
    CFG=$(echo "$entry" | awk '{print $1}')
    SEED=$(echo "$entry" | awk '{print $2}')
    FLAGS=$(echo "$entry" | awk '{for(i=3;i<=NF;i++) printf $i" "; print ""}')
    TOTAL=$((TOTAL + 1))
    EXP_NAME="exp4_${CFG}_s${SEED}"
    if find "saved_models/circmac/${EXP_NAME}" -name "training.json" 2>/dev/null | grep -q .; then
        echo "[DONE] $EXP_NAME"; SKIPPED=$((SKIPPED + 1)); continue; fi
    RAN=$((RAN + 1)); echo "[RUN]  $EXP_NAME ($FLAGS)"
    python training.py \
        --model_name circmac --task "$TASK" --seed "$SEED" \
        --d_model "$D_MODEL" --n_layer "$N_LAYER" \
        --batch_size "$BS" --num_workers "$NUM_WORKERS" \
        --lr "$LR" --epochs "$EPOCHS" --earlystop "$EARLYSTOP" \
        --device "$GPU" --exp "$EXP_NAME" \
        --interaction cross_attention --verbose \
        $FLAGS \
        2>&1 | tee "logs/exp4/${EXP_NAME}.log"
done

echo "=== Repair S2 Complete: $RAN ran, $SKIPPED skipped / $TOTAL total ==="
