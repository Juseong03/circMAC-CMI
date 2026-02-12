#!/bin/bash
#===============================================================================
# Experiment 1 - Part 1: Fair Comparison (max_len=440)
# All models use the same max_len so they train/evaluate on identical data.
# Includes: frozen + trainable pretrained models + CircMAC-PT
#
# Runs: 27 (4 models × 3 seeds × 2 modes + 3 CircMAC-PT)
# Usage: ./scripts/2gpu/exp1_fair.sh [GPU_ID]
#===============================================================================

set -e

GPU=${1:-0}
BEST_PT=${2:-"ss5_mlm_cpcl_bsj_pair"}
TASK="sites"
SEEDS=(1 2 3)
PRETRAINED_MODELS=("rnabert" "rnafm" "rnaernie" "rnamsm")
FAIR_MAX_LEN=440

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
echo "  Exp1 Part 1: Fair Comparison (max_len=$FAIR_MAX_LEN)"
echo "  GPU: $GPU | Runs: 27"
echo "=============================================="

# 1A: Frozen pretrained models (12 runs)
echo ""
echo "--- 1A: Frozen Pretrained Models ---"
for MODEL in "${PRETRAINED_MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        EXP_NAME="exp1_fair_frozen_${MODEL}_s${SEED}"
        echo "  $EXP_NAME"

        python training.py \
            --model_name "$MODEL" \
            --task "$TASK" \
            --seed "$SEED" \
            --max_len "$FAIR_MAX_LEN" \
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
    echo "Completed: $MODEL (frozen, fair)"
done

# 1B: Trainable pretrained models (12 runs)
echo ""
echo "--- 1B: Trainable Pretrained Models ---"
for MODEL in "${PRETRAINED_MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        EXP_NAME="exp1_fair_trainable_${MODEL}_s${SEED}"
        echo "  $EXP_NAME"

        python training.py \
            --model_name "$MODEL" \
            --task "$TASK" \
            --seed "$SEED" \
            --max_len "$FAIR_MAX_LEN" \
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
    echo "Completed: $MODEL (trainable, fair)"
done

# 1C: CircMAC-PT fair (3 runs)
echo ""
echo "--- 1C: CircMAC-PT (fair) ---"
PT_PATH="saved_models/circmac/exp2_pt_${BEST_PT}/best.pt"
if [ ! -f "$PT_PATH" ]; then
    echo "Warning: Pretrained model not found: $PT_PATH"
    echo "  Run Experiment 2 first! Skipping CircMAC-PT..."
else
    for SEED in "${SEEDS[@]}"; do
        EXP_NAME="exp1_fair_circmac_pt_s${SEED}"
        echo "  $EXP_NAME"

        python training.py \
            --model_name circmac \
            --task "$TASK" \
            --seed "$SEED" \
            --max_len "$FAIR_MAX_LEN" \
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
    echo "Completed: CircMAC-PT (fair)"
fi

echo ""
echo "=============================================="
echo "  Exp1 Part 1 (Fair) Complete! [27 runs]"
echo "=============================================="
