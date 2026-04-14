#!/bin/bash
#===============================================================================
# EXP4 — CircMAC Ablation Study
#
# exp4_full          : Full CircMAC (baseline) ← 시각화/분석에서 main 모델로 사용
# exp4_no_attn       : Attention branch 제거
# exp4_no_mamba      : Mamba branch 제거
# exp4_no_conv       : CNN branch 제거
# exp4_no_circ_bias  : Circular relative bias 제거
# exp4_attn_only     : Attention만 (no Mamba, no Conv)
# exp4_mamba_only    : Mamba만 (no Attn, no Conv)
# exp4_cnn_only      : CNN만 (no Attn, no Mamba)
#
# Seeds: 1 2 3
# Usage: ./scripts/final_v2/run_exp4_ablation.sh [GPU_ID]
#===============================================================================
set -e

GPU=${1:-0}
SEEDS=(1 2 3)
TASK="sites"
LR=1e-4; EPOCHS=150; EARLYSTOP=20; NUM_WORKERS=4
D_MODEL=128; N_LAYER=6; MAX_LEN=1022; BS=128

mkdir -p logs/exp4 saved_models
TOTAL=0; SKIPPED=0; RAN=0

echo "=== EXP4 CircMAC Ablation (GPU $GPU) ==="

# config name → extra flags
declare -A CONFIGS=(
    [full]=""
    [no_attn]="--no_attn"
    [no_mamba]="--no_mamba"
    [no_conv]="--no_conv"
    [no_circ_bias]="--no_circular_rel_bias"
    [attn_only]="--no_mamba --no_conv"
    [mamba_only]="--no_attn --no_conv"
    [cnn_only]="--no_attn --no_mamba"
)
# 순서 유지
ORDERED=(full no_attn no_mamba no_conv no_circ_bias attn_only mamba_only cnn_only)

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
