#!/bin/bash
#===============================================================================
# Experiment 1 - Part 2: Max Performance (model-specific max_len)
# Each model uses its own maximum sequence length.
# Skip rnabert (same as fair: max_len=438)
# Includes: frozen + trainable pretrained models + CircMAC-PT
#
# Runs: 21 (3 models × 3 seeds × 2 modes + 3 CircMAC-PT)
# Usage: ./scripts/2gpu/exp1_max.sh [GPU_ID]
#===============================================================================

set -e

GPU=${1:-1}
BEST_PT=${2:-"ss5_mlm_cpcl_bsj_pair"}
TASK="sites"
SEEDS=(1 2 3)
MAX_MODELS=("rnafm" "rnaernie" "rnamsm")  # Skip rnabert (same as fair)

# Hyperparameters
D_MODEL=128
N_LAYER=6
BATCH_SIZE=32
EPOCHS=150
EARLYSTOP=20
LR=1e-4
NUM_WORKERS=4

mkdir -p logs/exp1
mkdir -p saved_models

echo "=============================================="
echo "  Exp1 Part 2: Max Performance"
echo "  GPU: $GPU | Runs: 21"
echo "=============================================="

# 2A: Frozen pretrained models (9 runs)
echo ""
echo "--- 2A: Frozen Pretrained Models (max) ---"
for MODEL in "${MAX_MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        EXP_NAME="exp1_max_frozen_${MODEL}_s${SEED}"
        echo "  $EXP_NAME"

        python training.py \
            --model_name "$MODEL" \
            --task "$TASK" \
            --seed "$SEED" \
            --d_model "$D_MODEL" \
            --batch_size "$BATCH_SIZE" \
            --num_workers "$NUM_WORKERS" \
            --lr "$LR" \
            --epochs "$EPOCHS" \
            --earlystop "$EARLYSTOP" \
            --device "$GPU" \
            --exp "$EXP_NAME" \
            --interaction cross_attention \
            --verbose \
            2>&1 | tee "logs/exp1/${EXP_NAME}.log"
    done
    echo "Completed: $MODEL (frozen, max)"
done

# 2B: Trainable pretrained models (9 runs)
echo ""
echo "--- 2B: Trainable Pretrained Models (max) ---"
for MODEL in "${MAX_MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        EXP_NAME="exp1_max_trainable_${MODEL}_s${SEED}"
        echo "  $EXP_NAME"

        python training.py \
            --model_name "$MODEL" \
            --task "$TASK" \
            --seed "$SEED" \
            --d_model "$D_MODEL" \
            --batch_size "$BATCH_SIZE" \
            --num_workers "$NUM_WORKERS" \
            --lr "$LR" \
            --epochs "$EPOCHS" \
            --earlystop "$EARLYSTOP" \
            --device "$GPU" \
            --exp "$EXP_NAME" \
            --interaction cross_attention \
            --trainable_pretrained \
            --verbose \
            2>&1 | tee "logs/exp1/${EXP_NAME}.log"
    done
    echo "Completed: $MODEL (trainable, max)"
done

# 2C: CircMAC-PT max (3 runs)
echo ""
echo "--- 2C: CircMAC-PT (max) ---"
PT_PATH="saved_models/circmac/exp2_pt_${BEST_PT}/42/pretrain/model.pth"
if [ ! -f "$PT_PATH" ]; then
    echo "Warning: Pretrained model not found: $PT_PATH"
    echo "  Skipping CircMAC-PT (max)..."
else
    for SEED in "${SEEDS[@]}"; do
        EXP_NAME="exp1_max_circmac_pt_s${SEED}"
        echo "  $EXP_NAME"

        python training.py \
            --model_name circmac \
            --task "$TASK" \
            --seed "$SEED" \
            --d_model "$D_MODEL" \
            --n_layer "$N_LAYER" \
            --batch_size "$BATCH_SIZE" \
            --num_workers "$NUM_WORKERS" \
            --lr "$LR" \
            --epochs "$EPOCHS" \
            --earlystop "$EARLYSTOP" \
            --device "$GPU" \
            --exp "$EXP_NAME" \
            --load_pretrained "$PT_PATH" \
            --interaction cross_attention \
            --verbose \
            2>&1 | tee "logs/exp1/${EXP_NAME}.log"
    done
    echo "Completed: CircMAC-PT (max)"
fi

echo ""
echo "=============================================="
echo "  Exp1 Part 2 (Max) Complete! [21 runs]"
echo "=============================================="
