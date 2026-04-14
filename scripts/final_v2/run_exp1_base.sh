#!/bin/bash
#===============================================================================
# EXP1 — Base Encoder Comparison
#
# Models: circmac, mamba, lstm, transformer, hymba
# Seeds:  1 2 3
# Exp naming: exp1_{model}_s{seed}
#
# Usage: ./scripts/final_v2/run_exp1_base.sh [GPU_ID]
#===============================================================================
set -e

GPU=${1:-0}
SEEDS=(1 2 3)
TASK="sites"
LR=1e-4; EPOCHS=150; EARLYSTOP=20; NUM_WORKERS=4
D_MODEL=128; N_LAYER=6; MAX_LEN=1022

mkdir -p logs/exp1 saved_models
TOTAL=0; SKIPPED=0; RAN=0

run() {
    local MODEL=$1 EXP=$2 BS=$3; shift 3
    TOTAL=$((TOTAL+1))
    if find "saved_models/${MODEL}/${EXP}" -name "training.json" 2>/dev/null | grep -q .; then
        echo "[SKIP] $EXP"; SKIPPED=$((SKIPPED+1)); return 0; fi
    RAN=$((RAN+1)); echo "[RUN]  $EXP"
    python training.py \
        --model_name "$MODEL" --task $TASK --seed $SEED \
        --d_model $D_MODEL --n_layer $N_LAYER --max_len $MAX_LEN \
        --batch_size "$BS" --num_workers $NUM_WORKERS \
        --lr $LR --epochs $EPOCHS --earlystop $EARLYSTOP \
        --device $GPU --exp "$EXP" \
        --interaction cross_attention --verbose "$@" \
        2>&1 | tee "logs/exp1/${EXP}.log"
}

echo "=== EXP1 Base Encoders (GPU $GPU) ==="

for SEED in "${SEEDS[@]}"; do
    run circmac    "exp1_circmac_s${SEED}"      128
    run mamba      "exp1_mamba_s${SEED}"        128
    run hymba      "exp1_hymba_s${SEED}"        128
    run lstm       "exp1_lstm_s${SEED}"         128
    run transformer "exp1_transformer_s${SEED}" 64
done

echo "=== Done: $RAN ran, $SKIPPED skipped / $TOTAL total ==="
