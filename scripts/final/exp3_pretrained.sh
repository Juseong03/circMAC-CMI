#!/bin/bash
#===============================================================================
# Exp3: Pretrained RNA Model Comparison
# RQ: Does CircMAC-PT outperform general-purpose pretrained RNA models?
#
# Each model uses its own maximum sequence length.
# RNABERT/RNAErnie are limited by positional encoding (noted with † in paper).
#
# Model    | max_len | data coverage
# ─────────────────────────────────
# RNABERT  |   440   |   37.6%  †
# RNAErnie |   510   |   48.2%  ††
# RNA-FM   |  1024   |  100.0%
# RNA-MSM  |  1024   |  100.0%
# CircMAC  |  1022   |  100.0%  ← fair vs RNA-FM/MSM
#
# Runs: 4 models × 2 modes × 3 seeds + 3 CircMAC-PT = 27 runs
# Usage: ./scripts/final/exp3_pretrained.sh [GPU_ID] [BEST_PT_CONFIG]
#===============================================================================
set -e

GPU=${1:-0}
BEST_PT=${2:-"mlm_ntp_cpcl_pair"}
SEEDS=(1 2 3)
TASK="sites"

D_MODEL=128; N_LAYER=6; LR=1e-4; EPOCHS=150; EARLYSTOP=20; NUM_WORKERS=4

# Model-specific max_len
declare -A MAX_LENS=( ["rnabert"]=440 ["rnaernie"]=510 ["rnafm"]=1024 ["rnamsm"]=1024 )
MODELS=(rnabert rnaernie rnafm rnamsm)

mkdir -p logs/exp3 saved_models

TOTAL=0; SKIPPED=0; RAN=0

echo "=============================================="
echo "  Exp3: Pretrained RNA Model Comparison"
echo "  GPU: $GPU | Runs: 27"
echo "  Each model at its maximum sequence length"
echo "=============================================="

# ── 3A: Frozen (12 runs) ─────────────────────────────────────────────────────
echo ""
echo "--- 3A: Frozen ---"
for MODEL in "${MODELS[@]}"; do
    ML=${MAX_LENS[$MODEL]}
    for SEED in "${SEEDS[@]}"; do
        TOTAL=$((TOTAL + 1))
        EXP_NAME="exp3_${MODEL}_frozen_s${SEED}"
        if find "saved_models/${MODEL}/${EXP_NAME}" -name "training.json" 2>/dev/null | grep -q .; then
            echo "[DONE] $EXP_NAME"; SKIPPED=$((SKIPPED + 1)); continue; fi
        RAN=$((RAN + 1))
        echo "[RUN]  $EXP_NAME (max_len=$ML)"
        python training.py \
            --model_name "$MODEL" --task "$TASK" --seed "$SEED" \
            --d_model "$D_MODEL" --max_len "$ML" \
            --batch_size 32 --num_workers "$NUM_WORKERS" \
            --lr "$LR" --epochs "$EPOCHS" --earlystop "$EARLYSTOP" \
            --device "$GPU" --exp "$EXP_NAME" \
            --interaction cross_attention --verbose \
            2>&1 | tee "logs/exp3/${EXP_NAME}.log"
    done
    echo "  Done: $MODEL (frozen, max_len=$ML)"
done

# ── 3B: Trainable (12 runs) ──────────────────────────────────────────────────
echo ""
echo "--- 3B: Trainable ---"
for MODEL in "${MODELS[@]}"; do
    ML=${MAX_LENS[$MODEL]}
    # rnafm/rnamsm: 95-96M params, need smaller batch to avoid OOM
    if [ "$MODEL" = "rnafm" ] || [ "$MODEL" = "rnamsm" ]; then BS=16; else BS=32; fi
    for SEED in "${SEEDS[@]}"; do
        TOTAL=$((TOTAL + 1))
        EXP_NAME="exp3_${MODEL}_trainable_s${SEED}"
        if find "saved_models/${MODEL}/${EXP_NAME}" -name "training.json" 2>/dev/null | grep -q .; then
            echo "[DONE] $EXP_NAME"; SKIPPED=$((SKIPPED + 1)); continue; fi
        RAN=$((RAN + 1))
        echo "[RUN]  $EXP_NAME (max_len=$ML, bs=$BS)"
        python training.py \
            --model_name "$MODEL" --task "$TASK" --seed "$SEED" \
            --d_model "$D_MODEL" --max_len "$ML" \
            --batch_size "$BS" --num_workers "$NUM_WORKERS" \
            --lr "$LR" --epochs "$EPOCHS" --earlystop "$EARLYSTOP" \
            --device "$GPU" --exp "$EXP_NAME" \
            --interaction cross_attention \
            --trainable_pretrained --verbose \
            2>&1 | tee "logs/exp3/${EXP_NAME}.log"
    done
    echo "  Done: $MODEL (trainable, max_len=$ML, bs=$BS)"
done

# ── 3C: CircMAC-PT (3 runs) ──────────────────────────────────────────────────
echo ""
echo "--- 3C: CircMAC-PT (max_len=1022, 100% data) ---"
PT_PATH="saved_models/circmac/exp2_pt_${BEST_PT}/42/pretrain/model.pth"
if [ ! -f "$PT_PATH" ]; then
    echo "[WARN] Pretrain model not found: $PT_PATH"
    echo "  → Run Exp2 first, then re-run this script."
    echo "  → Non-CircMAC rows above are already saved."
else
    for SEED in "${SEEDS[@]}"; do
        TOTAL=$((TOTAL + 1))
        EXP_NAME="exp3_circmac_pt_s${SEED}"
        if find "saved_models/circmac/${EXP_NAME}" -name "training.json" 2>/dev/null | grep -q .; then
            echo "[DONE] $EXP_NAME"; SKIPPED=$((SKIPPED + 1)); continue; fi
        RAN=$((RAN + 1))
        echo "[RUN]  $EXP_NAME (max_len=1022, 100% data)"
        python training.py \
            --model_name circmac --task "$TASK" --seed "$SEED" \
            --d_model "$D_MODEL" --n_layer "$N_LAYER" --max_len 1022 \
            --batch_size 32 --num_workers "$NUM_WORKERS" \
            --lr "$LR" --epochs "$EPOCHS" --earlystop "$EARLYSTOP" \
            --device "$GPU" --exp "$EXP_NAME" \
            --load_pretrained "$PT_PATH" \
            --interaction cross_attention --verbose \
            2>&1 | tee "logs/exp3/${EXP_NAME}.log"
    done
    echo "  Done: CircMAC-PT"
fi

echo ""
echo "=============================================="
echo "  Exp3 Complete: $RAN ran, $SKIPPED skipped / $TOTAL total"
echo "=============================================="

echo ""
echo "  Data coverage summary:"
echo "  RNABERT  (max_len=440):  37.6% of data  [†  limited by positional encoding]"
echo "  RNAErnie (max_len=510):  48.2% of data  [†† limited by positional encoding]"
echo "  RNA-FM   (max_len=1024): 100.0% of data"
echo "  RNA-MSM  (max_len=1024): 100.0% of data"
echo "  CircMAC  (max_len=1022): 100.0% of data  [fair comparison vs FM/MSM]"
