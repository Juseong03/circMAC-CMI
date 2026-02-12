#!/bin/bash
#===============================================================================
# Experiment 2 - SS1: Single Secondary Structure
# Pretrain with df_circ_ss (1 SS per RNA) + Fine-tune
#
# Runs: 36 (9 pretrain + 27 finetune)
# Usage: ./scripts/2gpu/exp2_ss1.sh [GPU_ID]
#===============================================================================

set -e

GPU=${1:-0}
SEEDS=(1 2 3)
TASK="sites"
DATA_FILE="df_circ_ss"
DATA_TAG="ss1"

# Pretraining hyperparameters
PT_EPOCHS=300
PT_EARLYSTOP=30
PT_BATCH_SIZE=64
PT_LR=1e-4
D_MODEL=128
N_LAYER=6
MAX_LEN=1022
NUM_WORKERS=4

# Fine-tuning hyperparameters
FT_EPOCHS=150
FT_EARLYSTOP=20
FT_BATCH_SIZE=128
FT_LR=1e-4

mkdir -p logs/exp2/pretrain
mkdir -p logs/exp2/finetune
mkdir -p saved_models

echo "=============================================="
echo "  Exp2 SS1: Pretrain + Fine-tune (df_circ_ss)"
echo "  GPU: $GPU | Runs: 36"
echo "=============================================="

# Pretraining configurations
declare -A PT_CONFIGS
PT_CONFIGS["mlm"]="--mlm"
PT_CONFIGS["mlm_ntp"]="--mlm --ntp"
PT_CONFIGS["mlm_ssp"]="--mlm --ssp"
PT_CONFIGS["mlm_pairing"]="--mlm --pairing"
PT_CONFIGS["mlm_cpcl"]="--mlm --cpcl"
PT_CONFIGS["mlm_bsj"]="--mlm --bsj_mlm"
PT_CONFIGS["mlm_cpcl_bsj"]="--mlm --cpcl --bsj_mlm"
PT_CONFIGS["mlm_cpcl_bsj_pair"]="--mlm --cpcl --bsj_mlm --pairing"
PT_CONFIGS["full"]="--mlm --ntp --ssp --ss_labels_multi --pairing --cpcl --bsj_mlm"

#-----------------------------------------------
# Phase 1: Pretraining (9 runs)
#-----------------------------------------------
echo ""
echo "=== Phase 1: Pretraining ($DATA_TAG) ==="

for CONFIG_NAME in "${!PT_CONFIGS[@]}"; do
    CONFIG_FLAGS="${PT_CONFIGS[$CONFIG_NAME]}"
    EXP_NAME="exp2_pt_${DATA_TAG}_${CONFIG_NAME}"

    echo "  $EXP_NAME ($CONFIG_FLAGS)"

    python pretraining.py \
        --model_name circmac \
        --data_file "$DATA_FILE" \
        --max_len "$MAX_LEN" \
        --d_model "$D_MODEL" \
        --n_layer "$N_LAYER" \
        --batch_size "$PT_BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" \
        --lr "$PT_LR" \
        --epochs "$PT_EPOCHS" \
        --earlystop "$PT_EARLYSTOP" \
        --device "$GPU" \
        --exp "$EXP_NAME" \
        --verbose \
        $CONFIG_FLAGS \
        2>&1 | tee "logs/exp2/pretrain/${EXP_NAME}.log"

    echo "Completed pretrain: $CONFIG_NAME ($DATA_TAG)"
done

#-----------------------------------------------
# Phase 2: Fine-tuning with ss1 pretrained models (27 runs)
#-----------------------------------------------
echo ""
echo "=== Phase 2: Fine-tuning ($DATA_TAG) ==="

for CONFIG_NAME in "${!PT_CONFIGS[@]}"; do
    PT_PATH="saved_models/circmac/exp2_pt_${DATA_TAG}_${CONFIG_NAME}/best.pt"

    if [ ! -f "$PT_PATH" ]; then
        echo "Skipping ${DATA_TAG}_${CONFIG_NAME}: pretrained model not found"
        continue
    fi

    for SEED in "${SEEDS[@]}"; do
        EXP_NAME="exp2_${DATA_TAG}_${CONFIG_NAME}_${TASK}_s${SEED}"
        echo "  $EXP_NAME"

        python training.py \
            --model_name circmac \
            --task "$TASK" \
            --seed "$SEED" \
            --d_model "$D_MODEL" \
            --n_layer "$N_LAYER" \
            --batch_size "$FT_BATCH_SIZE" \
            --num_workers "$NUM_WORKERS" \
            --lr "$FT_LR" \
            --epochs "$FT_EPOCHS" \
            --earlystop "$FT_EARLYSTOP" \
            --device "$GPU" \
            --exp "$EXP_NAME" \
            --load_pretrained "$PT_PATH" \
            --interaction cross_attention \
            --verbose \
            2>&1 | tee "logs/exp2/finetune/${EXP_NAME}.log"
    done
done

echo ""
echo "=============================================="
echo "  Exp2 SS1 Complete! [36 runs]"
echo "=============================================="
