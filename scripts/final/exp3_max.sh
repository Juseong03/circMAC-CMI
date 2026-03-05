#!/bin/bash
#===============================================================================
# Exp3: Pretrained RNA Model Comparison — Max Performance (model-specific max_len)
# Skip RNABERT (same max_len as fair: 440)
#
# Models: RNA-FM (1024), RNAErnie (512), RNA-MSM (1024) + CircMAC-PT (1022)
# Modes:  frozen + trainable
#
# Runs: 3 models × 2 modes × 3 seeds + 3 CircMAC-PT = 21 runs
# Usage: ./scripts/final/exp3_max.sh [GPU_ID] [BEST_PT_CONFIG]
#===============================================================================
set -e

GPU=${1:-0}
BEST_PT=${2:-"mlm_ntp_cpcl_pair"}
SEEDS=(1 2 3)
TASK="sites"

D_MODEL=128; N_LAYER=6; LR=1e-4; EPOCHS=150; EARLYSTOP=20; NUM_WORKERS=4
BS=32; BS_CIRC=128

# Model-specific max lengths
declare -A MAX_LENS=( ["rnafm"]=1024 ["rnaernie"]=510 ["rnamsm"]=1024 )
MAX_MODELS=(rnafm rnaernie rnamsm)

mkdir -p logs/exp3/max saved_models

TOTAL=0; SKIPPED=0; RAN=0

echo "=============================================="
echo "  Exp3: Pretrained Models — Max (model-specific max_len)"
echo "  GPU: $GPU | Runs: 21"
echo "=============================================="

# ── 3D: Frozen (9 runs) ──────────────────────────────────────────────────────
echo ""
echo "--- 3D: Frozen (max) ---"
for MODEL in "${MAX_MODELS[@]}"; do
    ML=${MAX_LENS[$MODEL]}
    for SEED in "${SEEDS[@]}"; do
        TOTAL=$((TOTAL + 1))
        EXP_NAME="exp3_max_frozen_${MODEL}_s${SEED}"
        if find "saved_models/${MODEL}/${EXP_NAME}" -name "training.json" 2>/dev/null | grep -q .; then
            echo "[DONE] $EXP_NAME"; SKIPPED=$((SKIPPED + 1)); continue; fi
        RAN=$((RAN + 1))
        echo "[RUN]  $EXP_NAME (max_len=$ML)"
        python training.py \
            --model_name "$MODEL" --task "$TASK" --seed "$SEED" \
            --d_model "$D_MODEL" --max_len "$ML" \
            --batch_size "$BS" --num_workers "$NUM_WORKERS" \
            --lr "$LR" --epochs "$EPOCHS" --earlystop "$EARLYSTOP" \
            --device "$GPU" --exp "$EXP_NAME" \
            --interaction cross_attention --verbose \
            2>&1 | tee "logs/exp3/max/${EXP_NAME}.log"
    done
    echo "  Done: $MODEL (frozen)"
done

# ── 3E: Trainable (9 runs) ───────────────────────────────────────────────────
echo ""
echo "--- 3E: Trainable (max) ---"
for MODEL in "${MAX_MODELS[@]}"; do
    ML=${MAX_LENS[$MODEL]}
    if [ "$MODEL" = "rnafm" ] || [ "$MODEL" = "rnamsm" ]; then BS_RUN=16; else BS_RUN=$BS; fi
    for SEED in "${SEEDS[@]}"; do
        TOTAL=$((TOTAL + 1))
        EXP_NAME="exp3_max_trainable_${MODEL}_s${SEED}"
        if find "saved_models/${MODEL}/${EXP_NAME}" -name "training.json" 2>/dev/null | grep -q .; then
            echo "[DONE] $EXP_NAME"; SKIPPED=$((SKIPPED + 1)); continue; fi
        RAN=$((RAN + 1))
        echo "[RUN]  $EXP_NAME (max_len=$ML, bs=$BS_RUN)"
        python training.py \
            --model_name "$MODEL" --task "$TASK" --seed "$SEED" \
            --d_model "$D_MODEL" --max_len "$ML" \
            --batch_size "$BS_RUN" --num_workers "$NUM_WORKERS" \
            --lr "$LR" --epochs "$EPOCHS" --earlystop "$EARLYSTOP" \
            --device "$GPU" --exp "$EXP_NAME" \
            --interaction cross_attention \
            --trainable_pretrained --verbose \
            2>&1 | tee "logs/exp3/max/${EXP_NAME}.log"
    done
    echo "  Done: $MODEL (trainable)"
done

# ── 3F: CircMAC-PT max (3 runs) ──────────────────────────────────────────────
echo ""
echo "--- 3F: CircMAC-PT (max_len=1022) ---"
PT_PATH="saved_models/circmac/exp2_pt_${BEST_PT}/42/pretrain/model.pth"
if [ ! -f "$PT_PATH" ]; then
    echo "[WARN] CircMAC pretrain not found: $PT_PATH"
    echo "  Run Exp2 first. Skipping CircMAC-PT..."
else
    for SEED in "${SEEDS[@]}"; do
        TOTAL=$((TOTAL + 1))
        EXP_NAME="exp3_max_circmac_pt_s${SEED}"
        if find "saved_models/circmac/${EXP_NAME}" -name "training.json" 2>/dev/null | grep -q .; then
            echo "[DONE] $EXP_NAME"; SKIPPED=$((SKIPPED + 1)); continue; fi
        RAN=$((RAN + 1))
        echo "[RUN]  $EXP_NAME"
        python training.py \
            --model_name circmac --task "$TASK" --seed "$SEED" \
            --d_model "$D_MODEL" --n_layer "$N_LAYER" \
            --batch_size "$BS_CIRC" --num_workers "$NUM_WORKERS" \
            --lr "$LR" --epochs "$EPOCHS" --earlystop "$EARLYSTOP" \
            --device "$GPU" --exp "$EXP_NAME" \
            --load_pretrained "$PT_PATH" \
            --interaction cross_attention --verbose \
            2>&1 | tee "logs/exp3/max/${EXP_NAME}.log"
    done
    echo "  Done: CircMAC-PT"
fi

echo ""
echo "=============================================="
echo "  Exp3 Max Complete: $RAN ran, $SKIPPED skipped / $TOTAL total"
echo "=============================================="
