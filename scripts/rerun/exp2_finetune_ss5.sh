#!/bin/bash
#===============================================================================
# RERUN: Exp2 SS5 Fine-tune Only
# Pretrain phase already completed. This only runs the finetune phase.
# Automatically skips already-completed experiments.
#
# Usage: ./scripts/rerun/exp2_finetune_ss5.sh [GPU_ID]
#===============================================================================

set -e

GPU=${1:-1}
SEEDS=(1 2 3)
TASK="sites"
DATA_TAG="ss5"
PT_SEED=42

# Hyperparameters (same as original)
D_MODEL=128
N_LAYER=6
FT_BATCH_SIZE=128
FT_EPOCHS=150
FT_EARLYSTOP=20
FT_LR=1e-4
NUM_WORKERS=4

# Pretraining configurations (same as exp2_ss5.sh)
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

mkdir -p logs/exp2/finetune
mkdir -p saved_models

TOTAL=0
SKIPPED=0
RAN=0

echo "=============================================="
echo "  RERUN: Exp2 SS5 Fine-tune Only"
echo "  GPU: $GPU"
echo "=============================================="

for CONFIG_NAME in "${!PT_CONFIGS[@]}"; do
    PT_PATH="saved_models/circmac/exp2_pt_${DATA_TAG}_${CONFIG_NAME}/${PT_SEED}/pretrain/model.pth"

    if [ ! -f "$PT_PATH" ]; then
        echo "[SKIP] ${DATA_TAG}_${CONFIG_NAME}: pretrained model not found at $PT_PATH"
        continue
    fi

    for SEED in "${SEEDS[@]}"; do
        TOTAL=$((TOTAL + 1))
        EXP_NAME="exp2_${DATA_TAG}_${CONFIG_NAME}_${TASK}_s${SEED}"

        # Check if already completed (training.json exists)
        RESULT_DIR="saved_models/circmac/${EXP_NAME}"
        if find "$RESULT_DIR" -name "training.json" 2>/dev/null | grep -q .; then
            echo "[DONE] $EXP_NAME (already completed, skipping)"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        RAN=$((RAN + 1))
        echo "[RUN]  $EXP_NAME"

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
echo "  RERUN Complete: Exp2 SS5 Fine-tune"
echo "  Total: $TOTAL | Ran: $RAN | Skipped: $SKIPPED"
echo "=============================================="
