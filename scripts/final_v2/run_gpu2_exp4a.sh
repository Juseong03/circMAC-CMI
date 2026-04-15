#!/bin/bash
#===============================================================================
# GPU 2 — EXP4 Ablation (Part A)
#
# configs: full, no_attn, no_mamba, no_conv  × seeds 1 2 3  = 12 runs
# 예상 시간: ~30h (A100)
#
# Usage: ./scripts/final_v2/run_gpu2_exp4a.sh [GPU_ID]
#===============================================================================
set -e

GPU=${1:-2}
SEEDS=(1 2 3)
TASK="sites"
LR=1e-4; EPOCHS=150; EARLYSTOP=20; NUM_WORKERS=4
D_MODEL=128; N_LAYER=6; MAX_LEN=1022; BS=128

mkdir -p logs/exp4 saved_models
TOTAL=0; SKIPPED=0; RAN=0

declare -A CONFIGS=(
    [full]=""
    [no_attn]="--no_attn"
    [no_mamba]="--no_mamba"
    [no_conv]="--no_conv"
)
ORDERED=(full no_attn no_mamba no_conv)

echo "=== EXP4 Part-A (GPU $GPU) ==="

for CFG in "${ORDERED[@]}"; do
    FLAGS=${CONFIGS[$CFG]}
    for SEED in "${SEEDS[@]}"; do
        EXP="exp4_${CFG}_s${SEED}"
        TOTAL=$((TOTAL+1))
        if find "saved_models/circmac/${EXP}" -name "training.json" 2>/dev/null | grep -q .; then
            echo "[SKIP] $EXP"; SKIPPED=$((SKIPPED+1)); continue; fi
        RAN=$((RAN+1)); echo "[RUN]  $EXP  [$FLAGS]"
        eval python training.py \
            --model_name circmac --task $TASK --seed $SEED \
            --d_model $D_MODEL --n_layer $N_LAYER --max_len $MAX_LEN \
            --batch_size $BS --num_workers $NUM_WORKERS \
            --lr $LR --epochs $EPOCHS --earlystop $EARLYSTOP \
            --device $GPU --exp "$EXP" \
            --interaction cross_attention --verbose $FLAGS \
            2>&1 | tee "logs/exp4/${EXP}.log"
    done
done

echo "=== Done: $RAN ran, $SKIPPED skipped / $TOTAL total ==="
