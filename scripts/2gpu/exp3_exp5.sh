#!/bin/bash
#===============================================================================
# Experiment 3 + 5: Encoder Comparison + Interaction Mechanism
#
# Exp3: lstm, transformer, mamba, hymba, circmac (15 runs)
# Exp5: concat, elementwise, cross_attention (9 runs)
#
# Runs: 24
# Usage: ./scripts/2gpu/exp3_exp5.sh [GPU_ID]
#===============================================================================

set -e

GPU=${1:-0}
SEEDS=(1 2 3)
TASK="sites"

# Hyperparameters
D_MODEL=128
N_LAYER=6
BATCH_SIZE=128
EPOCHS=150
EARLYSTOP=20
LR=1e-4
NUM_WORKERS=4

mkdir -p logs/exp3
mkdir -p logs/exp5
mkdir -p saved_models

#-----------------------------------------------
# Experiment 3: Encoder Architecture Comparison (15 runs)
#-----------------------------------------------
MODELS=("lstm" "transformer" "mamba" "hymba" "circmac")

echo "=============================================="
echo "  Experiment 3: Encoder Architecture"
echo "  GPU: $GPU | Runs: 15"
echo "=============================================="

for MODEL in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        EXP_NAME="exp3_${MODEL}_${TASK}_s${SEED}"
        echo "  $EXP_NAME"

        python training.py \
            --model_name "$MODEL" \
            --task "$TASK" \
            --seed "$SEED" \
            --d_model "$D_MODEL" \
            --n_layer "$N_LAYER" \
            --batch_size "$BATCH_SIZE" \
            --num_workers "$NUM_WORKERS" \
            --epochs "$EPOCHS" \
            --earlystop "$EARLYSTOP" \
            --lr "$LR" \
            --device "$GPU" \
            --exp "$EXP_NAME" \
            --interaction cross_attention \
            --verbose \
            2>&1 | tee "logs/exp3/${EXP_NAME}.log"
    done
    echo "Completed: $MODEL"
done

echo ""
echo "  Experiment 3 Complete! [15 runs]"
echo ""

#-----------------------------------------------
# Experiment 5: Interaction Mechanism (9 runs)
#-----------------------------------------------
INTERACTIONS=("concat" "elementwise" "cross_attention")

echo "=============================================="
echo "  Experiment 5: Interaction Mechanism"
echo "  GPU: $GPU | Runs: 9"
echo "=============================================="

for INTERACTION in "${INTERACTIONS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        EXP_NAME="exp5_${INTERACTION}_s${SEED}"
        echo "  $EXP_NAME"

        python training.py \
            --model_name circmac \
            --task "$TASK" \
            --seed "$SEED" \
            --d_model "$D_MODEL" \
            --n_layer "$N_LAYER" \
            --batch_size "$BATCH_SIZE" \
            --num_workers "$NUM_WORKERS" \
            --epochs "$EPOCHS" \
            --earlystop "$EARLYSTOP" \
            --lr "$LR" \
            --device "$GPU" \
            --exp "$EXP_NAME" \
            --interaction "$INTERACTION" \
            --verbose \
            2>&1 | tee "logs/exp5/${EXP_NAME}.log"
    done
    echo "Completed: $INTERACTION"
done

echo ""
echo "=============================================="
echo "  Exp3 + Exp5 Complete! [24 runs]"
echo "=============================================="
