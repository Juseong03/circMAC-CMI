#!/bin/bash
#===============================================================================
# GPU 0: Finetune  mlm_cpcl + pairing  (seeds 1,2,3 each)
#
# PT models required (already exist):
#   saved_models/circmac/v2_ptm_mlm_cpcl/42/pretrain/model.pth
#   saved_models/circmac/v2_ptm_pairing/42/pretrain/model.pth
#
# Run in PARALLEL with run_g1_ft.sh (bsj + mlm_bsj on GPU 1).
#
# Usage: ./scripts/v2_ft_pending/run_g0_ft.sh [GPU_ID]
#===============================================================================
set -e

GPU=${1:-0}
SEEDS=(1 2 3)
TASK="sites"
PT_SEED=42
D_MODEL=128; N_LAYER=6; MAX_LEN=1022; NUM_WORKERS=4
FT_BS=64; FT_LR=1e-4; FT_EP=150; FT_ES=20

mkdir -p logs/v2/pt saved_models
TOTAL=0; SKIPPED=0; RAN=0

echo "========================================"
echo "  GPU $GPU — FT: mlm_cpcl + pairing  seeds=(${SEEDS[*]})"
echo "  FT: BS=$FT_BS LR=$FT_LR epochs=$FT_EP es=$FT_ES"
echo "========================================"

run_ft() {
    local EXP_KEY=$1
    local PT_EXP=$2
    local SEED=$3
    local EXP="v2_pt_${EXP_KEY}_s${SEED}"
    local PT_PATH="saved_models/circmac/${PT_EXP}/${PT_SEED}/pretrain/model.pth"
    TOTAL=$((TOTAL+1))

    if [ ! -f "$PT_PATH" ]; then
        echo "[SKIP] PT not found: $PT_PATH"; TOTAL=$((TOTAL-1)); return
    fi

    if find "saved_models/circmac/${EXP}" -name "training.json" 2>/dev/null | grep -q .; then
        echo "[SKIP] $EXP (already done)"; SKIPPED=$((SKIPPED+1)); return
    fi

    RAN=$((RAN+1))
    echo "[RUN]  $EXP"
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
}

echo ""
echo "━━━ mlm_cpcl ━━━"
for SEED in "${SEEDS[@]}"; do
    run_ft "mlm_cpcl" "v2_ptm_mlm_cpcl" $SEED
done

echo ""
echo "━━━ pairing ━━━"
for SEED in "${SEEDS[@]}"; do
    run_ft "pairing" "v2_ptm_pairing" $SEED
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Done: $RAN ran, $SKIPPED skipped / $TOTAL total"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
