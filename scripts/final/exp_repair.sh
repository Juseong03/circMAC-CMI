#!/bin/bash
#===============================================================================
# Repair Script: Re-run failed / missing experiments
#
# Missing items:
#   [EXP4] mamba_only s1,s2,s3 | attn_only s1 | cnn_only s1
#   [EXP5] cross_attention s1,s2,s3 (CUDA bug on old server — fixed now)
#   [EXP3] rnafm trainable (OOM → bs=8 fix)
#
# Usage:  ./scripts/final/exp_repair.sh [GPU_ID]
#===============================================================================
set -e

GPU=${1:-0}
SEEDS=(1 2 3)
TASK="sites"

D_MODEL=128; N_LAYER=6; LR=1e-4; EPOCHS=150; EARLYSTOP=20; NUM_WORKERS=4
BS=128   # standard batch size
BS_FM=8  # RNA-FM trainable: OOM fix

mkdir -p logs/exp4 logs/exp5 logs/exp3 saved_models

TOTAL=0; SKIPPED=0; RAN=0

echo "=============================================="
echo "  Repair Script: Missing / Failed Experiments"
echo "  GPU: $GPU"
echo "=============================================="

# ──────────────────────────────────────────────
# EXP4: Missing ablation variants
# ──────────────────────────────────────────────
echo ""
echo "[EXP4] Ablation — mamba_only, attn_only s1, cnn_only s1"

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
    FLAGS=$(echo "$entry" | cut -d' ' -f3-)

    TOTAL=$((TOTAL + 1))
    EXP_NAME="exp4_${CFG}_s${SEED}"

    if find "saved_models/circmac/${EXP_NAME}" -name "training.json" 2>/dev/null | grep -q .; then
        echo "[DONE] $EXP_NAME"; SKIPPED=$((SKIPPED + 1)); continue
    fi

    RAN=$((RAN + 1))
    echo "[RUN]  $EXP_NAME  ($FLAGS)"
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

# ──────────────────────────────────────────────
# EXP5: cross_attention (fixed — was CUDA bug on old server)
# ──────────────────────────────────────────────
echo ""
echo "[EXP5] Interaction — cross_attention s1,s2,s3"

for SEED in "${SEEDS[@]}"; do
    TOTAL=$((TOTAL + 1))
    EXP_NAME="exp5_cross_attention_s${SEED}"

    if find "saved_models/circmac/${EXP_NAME}" -name "training.json" 2>/dev/null | grep -q .; then
        echo "[DONE] $EXP_NAME"; SKIPPED=$((SKIPPED + 1)); continue
    fi

    RAN=$((RAN + 1))
    echo "[RUN]  $EXP_NAME"
    python training.py \
        --model_name circmac --task "$TASK" --seed "$SEED" \
        --d_model "$D_MODEL" --n_layer "$N_LAYER" \
        --batch_size "$BS" --num_workers "$NUM_WORKERS" \
        --lr "$LR" --epochs "$EPOCHS" --earlystop "$EARLYSTOP" \
        --device "$GPU" --exp "$EXP_NAME" \
        --interaction cross_attention --verbose \
        2>&1 | tee "logs/exp5/${EXP_NAME}.log"
done

# ──────────────────────────────────────────────
# EXP3: RNA-FM trainable (OOM → bs=8)
# ──────────────────────────────────────────────
echo ""
echo "[EXP3] RNA-FM trainable — bs=8 (OOM fix)"

for SEED in "${SEEDS[@]}"; do
    TOTAL=$((TOTAL + 1))
    EXP_NAME="exp3_rnafm_trainable_s${SEED}"

    if find "saved_models/rnafm/${EXP_NAME}" -name "training.json" 2>/dev/null | grep -q .; then
        echo "[DONE] $EXP_NAME"; SKIPPED=$((SKIPPED + 1)); continue
    fi

    RAN=$((RAN + 1))
    echo "[RUN]  $EXP_NAME  (bs=$BS_FM)"
    python training.py \
        --model_name rnafm --task "$TASK" --seed "$SEED" \
        --batch_size "$BS_FM" --num_workers "$NUM_WORKERS" \
        --lr "$LR" --epochs "$EPOCHS" --earlystop "$EARLYSTOP" \
        --device "$GPU" --exp "$EXP_NAME" \
        --interaction cross_attention \
        --trainable_pretrained --verbose \
        2>&1 | tee "logs/exp3/${EXP_NAME}.log"
done

echo ""
echo "=============================================="
echo "  Repair Complete: $RAN ran, $SKIPPED skipped / $TOTAL total"
echo "=============================================="
