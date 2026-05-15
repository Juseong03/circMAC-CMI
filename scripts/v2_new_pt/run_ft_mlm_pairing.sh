#!/bin/bash
#===============================================================================
# FT: v2_ptm_mlm_pairing  (--mlm --pairing)  — seeds 1, 2, 3 (sequential)
#
# Pretraining source: saved_models/circmac/v2_ptm_mlm_pairing/42/pretrain/model.pth
# Fine-tuned outputs: saved_models/circmac/v2_pt_mlm_pairing_s{1,2,3}/
#
# Usage: ./scripts/v2_new_pt/run_ft_mlm_pairing.sh [GPU_ID]
#===============================================================================
set -e

GPU=${1:-0}
SEEDS=(1 2 3)
TASK="sites"
PT_EXP="v2_ptm_mlm_pairing"
PT_SEED=42
PT_PATH="saved_models/circmac/${PT_EXP}/${PT_SEED}/pretrain/model.pth"

D_MODEL=128; N_LAYER=6; MAX_LEN=1022; NUM_WORKERS=4
FT_BS=64; FT_LR=1e-4; FT_EP=150; FT_ES=20

mkdir -p logs/v2/pt saved_models
TOTAL=0; SKIPPED=0; RAN=0

echo "========================================"
echo "  FT: v2_ptm_mlm_pairing  (GPU $GPU)"
echo "  PT:  $PT_PATH"
echo "  BS=$FT_BS LR=$FT_LR EP=$FT_EP ES=$FT_ES"
echo "  Seeds: ${SEEDS[*]}"
echo "========================================"

if [ ! -f "$PT_PATH" ]; then
    echo "[ERROR] Pretrained model not found: $PT_PATH"
    exit 1
fi

for SEED in "${SEEDS[@]}"; do
    EXP="v2_pt_mlm_pairing_s${SEED}"
    TOTAL=$((TOTAL+1))

    if find "saved_models/circmac/${EXP}" -name "training.json" 2>/dev/null | grep -q .; then
        echo "[SKIP] $EXP (already done)"; SKIPPED=$((SKIPPED+1)); continue
    fi

    RAN=$((RAN+1))
    echo ""
    echo "[RUN]  $EXP  (seed=$SEED)"
    python training.py \
        --model_name circmac --task $TASK --seed $SEED \
        --d_model $D_MODEL --n_layer $N_LAYER --max_len $MAX_LEN \
        --batch_size $FT_BS --num_workers $NUM_WORKERS \
        --lr $FT_LR --epochs $FT_EP --earlystop $FT_ES \
        --device $GPU --exp $EXP \
        --interaction cross_attention \
        --load_pretrained "$PT_PATH" \
        --verbose \
        2>&1 | tee "logs/v2/pt/${EXP}.log"
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Done: $RAN ran, $SKIPPED skipped / $TOTAL total"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
