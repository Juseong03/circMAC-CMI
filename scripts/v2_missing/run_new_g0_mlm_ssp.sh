#!/bin/bash
#===============================================================================
# GPU 0 — v2_pt_mlm_ssp  (MLM + SSP finetuning × 3 seeds)
#
# Pretrain: v2_ptm_mlm_ssp  ← 이미 완료 (epoch 76 best)
# Finetune: v2_pt_mlm_ssp_s1, _s2, _s3
#
# Usage: ./scripts/v2_missing/run_new_g0_mlm_ssp.sh [GPU_ID]
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
echo "  GPU $GPU — v2_pt_mlm_ssp (MLM + SSP)"
echo "  FT: BS=$FT_BS LR=$FT_LR epochs=$FT_EP es=$FT_ES"
echo "========================================"

PT_PATH="saved_models/circmac/v2_ptm_mlm_ssp/${PT_SEED}/pretrain/model.pth"

if [ ! -f "$PT_PATH" ]; then
    echo "ERROR: Pretrained model not found: $PT_PATH"
    echo "  v2_ptm_mlm_ssp 가 완료되지 않은 것 같습니다."
    exit 1
fi
echo "[OK] Pretrained model found: $PT_PATH"
echo ""

for SEED in "${SEEDS[@]}"; do
    EXP="v2_pt_mlm_ssp_s${SEED}"
    TOTAL=$((TOTAL+1))
    if find "saved_models/circmac/${EXP}" -name "training.json" 2>/dev/null | grep -q .; then
        echo "[SKIP] $EXP (already done)"; SKIPPED=$((SKIPPED+1)); continue
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
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Done: $RAN ran, $SKIPPED skipped / $TOTAL total"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
