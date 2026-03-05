#!/bin/bash
#===============================================================================
# Exp3: Pretrained RNA Model Comparison — Fair Comparison (max_len=440)
# RQ: Does CircMAC-PT outperform general-purpose pretrained RNA models?
#
# All models trained/evaluated on identical data (max_len=440 = RNABERT limit).
# Modes: frozen (encoder fixed) + trainable (encoder fine-tuned)
# Models: RNABERT, RNA-FM, RNAErnie, RNA-MSM + CircMAC-PT
#
# Runs: 4 models × 2 modes × 3 seeds + 3 CircMAC-PT = 27 runs
# Usage: ./scripts/final/exp3_fair.sh [GPU_ID] [BEST_PT_CONFIG]
#===============================================================================
set -e

GPU=${1:-0}
BEST_PT=${2:-"mlm_ntp_cpcl_pair"}  # best config from Exp2
SEEDS=(1 2 3)
TASK="sites"
MAX_LEN=440  # RNABERT's limit — fair comparison

D_MODEL=128; N_LAYER=6; LR=1e-4; EPOCHS=150; EARLYSTOP=20; NUM_WORKERS=4
BS=32        # smaller batch for large pretrained models
BS_CIRC=128  # CircMAC uses larger batch

MODELS=(rnabert rnafm rnaernie rnamsm)

mkdir -p logs/exp3/fair saved_models

TOTAL=0; SKIPPED=0; RAN=0

echo "=============================================="
echo "  Exp3: Pretrained Models — Fair (max_len=$MAX_LEN)"
echo "  GPU: $GPU | Runs: 27"
echo "=============================================="

# ── 3A: Frozen pretrained models (12 runs) ───────────────────────────────────
echo ""
echo "--- 3A: Frozen (max_len=$MAX_LEN) ---"
for MODEL in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        TOTAL=$((TOTAL + 1))
        EXP_NAME="exp3_fair_frozen_${MODEL}_s${SEED}"
        if find "saved_models/${MODEL}/${EXP_NAME}" -name "training.json" 2>/dev/null | grep -q .; then
            echo "[DONE] $EXP_NAME"; SKIPPED=$((SKIPPED + 1)); continue; fi
        RAN=$((RAN + 1))
        echo "[RUN]  $EXP_NAME"
        python training.py \
            --model_name "$MODEL" --task "$TASK" --seed "$SEED" \
            --d_model "$D_MODEL" --max_len "$MAX_LEN" \
            --batch_size "$BS" --num_workers "$NUM_WORKERS" \
            --lr "$LR" --epochs "$EPOCHS" --earlystop "$EARLYSTOP" \
            --device "$GPU" --exp "$EXP_NAME" \
            --interaction cross_attention --verbose \
            2>&1 | tee "logs/exp3/fair/${EXP_NAME}.log"
    done
    echo "  Done: $MODEL (frozen)"
done

# ── 3B: Trainable pretrained models (12 runs) ────────────────────────────────
echo ""
echo "--- 3B: Trainable (max_len=$MAX_LEN) ---"
for MODEL in "${MODELS[@]}"; do
    # rnafm/rnamsm need smaller batch to avoid OOM in trainable mode
    if [ "$MODEL" = "rnafm" ] || [ "$MODEL" = "rnamsm" ]; then BS_RUN=16; else BS_RUN=$BS; fi
    for SEED in "${SEEDS[@]}"; do
        TOTAL=$((TOTAL + 1))
        EXP_NAME="exp3_fair_trainable_${MODEL}_s${SEED}"
        if find "saved_models/${MODEL}/${EXP_NAME}" -name "training.json" 2>/dev/null | grep -q .; then
            echo "[DONE] $EXP_NAME"; SKIPPED=$((SKIPPED + 1)); continue; fi
        RAN=$((RAN + 1))
        echo "[RUN]  $EXP_NAME (bs=$BS_RUN)"
        python training.py \
            --model_name "$MODEL" --task "$TASK" --seed "$SEED" \
            --d_model "$D_MODEL" --max_len "$MAX_LEN" \
            --batch_size "$BS_RUN" --num_workers "$NUM_WORKERS" \
            --lr "$LR" --epochs "$EPOCHS" --earlystop "$EARLYSTOP" \
            --device "$GPU" --exp "$EXP_NAME" \
            --interaction cross_attention \
            --trainable_pretrained --verbose \
            2>&1 | tee "logs/exp3/fair/${EXP_NAME}.log"
    done
    echo "  Done: $MODEL (trainable)"
done

# ── 3C: CircMAC-PT (3 runs) ──────────────────────────────────────────────────
echo ""
echo "--- 3C: CircMAC-PT (max_len=$MAX_LEN) ---"
PT_PATH="saved_models/circmac/exp2_pt_${BEST_PT}/42/pretrain/model.pth"
if [ ! -f "$PT_PATH" ]; then
    echo "[WARN] CircMAC pretrain not found: $PT_PATH"
    echo "  Run Exp2 first. Skipping CircMAC-PT..."
else
    for SEED in "${SEEDS[@]}"; do
        TOTAL=$((TOTAL + 1))
        EXP_NAME="exp3_fair_circmac_pt_s${SEED}"
        if find "saved_models/circmac/${EXP_NAME}" -name "training.json" 2>/dev/null | grep -q .; then
            echo "[DONE] $EXP_NAME"; SKIPPED=$((SKIPPED + 1)); continue; fi
        RAN=$((RAN + 1))
        echo "[RUN]  $EXP_NAME"
        python training.py \
            --model_name circmac --task "$TASK" --seed "$SEED" \
            --d_model "$D_MODEL" --n_layer "$N_LAYER" --max_len "$MAX_LEN" \
            --batch_size "$BS_CIRC" --num_workers "$NUM_WORKERS" \
            --lr "$LR" --epochs "$EPOCHS" --earlystop "$EARLYSTOP" \
            --device "$GPU" --exp "$EXP_NAME" \
            --load_pretrained "$PT_PATH" \
            --interaction cross_attention --verbose \
            2>&1 | tee "logs/exp3/fair/${EXP_NAME}.log"
    done
    echo "  Done: CircMAC-PT"
fi

echo ""
echo "=============================================="
echo "  Exp3 Fair Complete: $RAN ran, $SKIPPED skipped / $TOTAL total"
echo "=============================================="
