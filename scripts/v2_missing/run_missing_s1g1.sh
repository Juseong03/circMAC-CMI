#!/bin/bash
#===============================================================================
# Missing Experiments — Server1 GPU 1
#
# EXP5 (incomplete): v2_int_elementwise s2, s3
# EXP6 (all missing): v2_head_conv1d s1,2,3 + v2_head_linear s1,2,3
#
# Total: 2 + 6 = 8 finetune runs  (~4h on A100)
#
# Usage: ./scripts/v2_missing/run_missing_s1g1.sh [GPU_ID]
#===============================================================================
set -e

GPU=${1:-1}
SEEDS=(1 2 3)
TASK="sites"

D_MODEL=128; N_LAYER=6; MAX_LEN=1022; NUM_WORKERS=4
FT_BS=64; FT_LR=1e-4; FT_EP=150; FT_ES=20

mkdir -p logs/v2/int logs/v2/head saved_models
TOTAL=0; SKIPPED=0; RAN=0

echo "========================================"
echo "  Missing Experiments — Server1 GPU $GPU"
echo "  EXP5: v2_int_elementwise (s2, s3)"
echo "  EXP6: v2_head_conv1d/linear (s1,2,3)"
echo "  BS=$FT_BS LR=$FT_LR epochs=$FT_EP es=$FT_ES"
echo "========================================"

run_ft() {
    local MODEL=$1 EXP=$2 LOG=$3; shift 3
    TOTAL=$((TOTAL+1))
    if find "saved_models/${MODEL}/${EXP}" -name "training.json" 2>/dev/null | grep -q .; then
        echo "[SKIP] $EXP"; SKIPPED=$((SKIPPED+1)); return 0
    fi
    RAN=$((RAN+1)); echo "[RUN]  $EXP"
    python training.py \
        --model_name "$MODEL" --task $TASK \
        --d_model $D_MODEL --n_layer $N_LAYER --max_len $MAX_LEN \
        --batch_size $FT_BS --num_workers $NUM_WORKERS \
        --lr $FT_LR --epochs $FT_EP --earlystop $FT_ES \
        --device $GPU --exp "$EXP" \
        --interaction cross_attention --verbose "$@" \
        2>&1 | tee "$LOG"
}

# ══ EXP5: v2_int_elementwise (s2, s3 only, s1 already done) ══════════════════
echo ""
echo "━━━ EXP5: v2_int_elementwise (s2, s3) ━━━"
for SEED in 2 3; do
    EXP="v2_int_elementwise_s${SEED}"
    run_ft circmac "$EXP" "logs/v2/int/${EXP}.log" \
        --seed $SEED --interaction elementwise
done

# ══ EXP6: v2_head_conv1d ══════════════════════════════════════════════════════
echo ""
echo "━━━ EXP6: v2_head_conv1d (s1,2,3) ━━━"
for SEED in "${SEEDS[@]}"; do
    EXP="v2_head_conv1d_s${SEED}"
    run_ft circmac "$EXP" "logs/v2/head/${EXP}.log" \
        --seed $SEED --site_head_type conv1d
done

# ══ EXP6: v2_head_linear ══════════════════════════════════════════════════════
echo ""
echo "━━━ EXP6: v2_head_linear (s1,2,3) ━━━"
for SEED in "${SEEDS[@]}"; do
    EXP="v2_head_linear_s${SEED}"
    run_ft circmac "$EXP" "logs/v2/head/${EXP}.log" \
        --seed $SEED --site_head_type linear
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Done: $RAN ran, $SKIPPED skipped / $TOTAL total"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
