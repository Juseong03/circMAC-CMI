#!/bin/bash
#===============================================================================
# GPU 3 — v2_pt_mlm_bsj  (MLM + BSJ_MLM pretrain → finetune × 3 seeds)
#
# Pretrain: v2_ptm_mlm_bsj  (--mlm --bsj_mlm, seed=42)
# Finetune: v2_pt_mlm_bsj_s1, _s2, _s3
#
# 의의: BSJ_MLM 단독(v2_pt_bsj)과 비교
#       circRNA BSJ-focused masking + MLM 조합이 binding site 예측에 효과적인지 검증
#       (BSJ 근처가 miRNA binding site와 관련이 높으므로 가장 기대되는 조합)
#
# Note: v2_pt_bsj (BSJ 단독)는 별도 GPU에서 실행 중
#
# Usage: ./scripts/v2_missing/run_new_g3_mlm_bsj.sh [GPU_ID]
#===============================================================================
set -e

GPU=${1:-3}
SEEDS=(1 2 3)
TASK="sites"
PT_SEED=42

D_MODEL=128; N_LAYER=6; MAX_LEN=1022; NUM_WORKERS=4
PT_DATA="df_pretrain"
PT_BS=64; PT_LR=1e-3; PT_WD=0.01; PT_EP=300; PT_ES=30
FT_BS=64; FT_LR=1e-4; FT_EP=150; FT_ES=20

mkdir -p logs/v2/ptm logs/v2/pt saved_models
TOTAL=0; SKIPPED=0; RAN=0

echo "========================================"
echo "  GPU $GPU — v2_pt_mlm_bsj (MLM + BSJ_MLM)"
echo "  PT: BS=$PT_BS LR=$PT_LR epochs=$PT_EP es=$PT_ES"
echo "  FT: BS=$FT_BS LR=$FT_LR epochs=$FT_EP es=$FT_ES"
echo "========================================"

# ── Pretraining ──────────────────────────────────────────────────────────────
PT_EXP="v2_ptm_mlm_bsj"
PT_PATH="saved_models/circmac/${PT_EXP}/${PT_SEED}/pretrain/model.pth"
TOTAL=$((TOTAL+1))

if [ -f "$PT_PATH" ]; then
    echo "[SKIP] pretrain: $PT_EXP (already done)"
    SKIPPED=$((SKIPPED+1))
else
    RAN=$((RAN+1))
    echo "[RUN]  pretrain: $PT_EXP"
    python pretraining.py \
        --model_name circmac --data_file "$PT_DATA" \
        --max_len $MAX_LEN --d_model $D_MODEL --n_layer $N_LAYER \
        --batch_size $PT_BS --num_workers $NUM_WORKERS \
        --optimizer adamw --lr $PT_LR --w_decay $PT_WD \
        --epochs $PT_EP --earlystop $PT_ES \
        --device $GPU --exp "$PT_EXP" --seed $PT_SEED \
        --verbose --mlm --bsj_mlm \
        2>&1 | tee "logs/v2/ptm/${PT_EXP}.log"
fi

if [ ! -f "$PT_PATH" ]; then
    echo "ERROR: Pretrained model not found after PT run: $PT_PATH"
    exit 1
fi

# ── Finetuning ────────────────────────────────────────────────────────────────
echo ""
echo "━━━ Finetuning: v2_pt_mlm_bsj s1,2,3 ━━━"
for SEED in "${SEEDS[@]}"; do
    EXP="v2_pt_mlm_bsj_s${SEED}"
    TOTAL=$((TOTAL+1))
    if find "saved_models/circmac/${EXP}" -name "training.json" 2>/dev/null | grep -q .; then
        echo "[SKIP] $EXP"; SKIPPED=$((SKIPPED+1)); continue
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
