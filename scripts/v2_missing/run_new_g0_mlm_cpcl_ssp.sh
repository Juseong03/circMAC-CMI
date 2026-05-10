#!/bin/bash
#===============================================================================
# GPU 0 — v2_pt_mlm_cpcl_ssp  (MLM + CPCL + SSP pretrain → finetune × 3 seeds)
#
# Pretrain: v2_ptm_mlm_cpcl_ssp  (--mlm --cpcl --ssp, seed=42)
# Finetune: v2_pt_mlm_cpcl_ssp_s1, _s2, _s3
#
# 의의: CPCL이 단독으로 최고 성능(0.7611)이고 MLM도 강력(0.7585)한 상황에서
#       MLM+CPCL+SSP 조합으로 추가 향상 가능성 검증
#       SSP(이차구조)가 circRNA binding site와 관련 높으므로 보완적 신호 기대
#
# Comparison targets:
#   v2_pt_cpcl       : 0.7611 (best single)
#   v2_pt_mlm        : 0.7585
#   v2_pt_mlm_cpcl   : (pending)
#   v2_pt_mlm_ssp    : 0.7389
#
# Usage: ./scripts/v2_missing/run_new_g0_mlm_cpcl_ssp.sh [GPU_ID]
#===============================================================================
set -e

GPU=${1:-0}
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
echo "  GPU $GPU — v2_pt_mlm_cpcl_ssp (MLM + CPCL + SSP)"
echo "  PT: BS=$PT_BS LR=$PT_LR epochs=$PT_EP es=$PT_ES"
echo "  FT: BS=$FT_BS LR=$FT_LR epochs=$FT_EP es=$FT_ES"
echo "========================================"

# ── Pretraining ──────────────────────────────────────────────────────────────
PT_EXP="v2_ptm_mlm_cpcl_ssp"
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
        --verbose --mlm --cpcl --ssp \
        2>&1 | tee "logs/v2/ptm/${PT_EXP}.log"
fi

if [ ! -f "$PT_PATH" ]; then
    echo "ERROR: Pretrained model not found after PT run: $PT_PATH"
    exit 1
fi

# ── Finetuning ────────────────────────────────────────────────────────────────
echo ""
echo "━━━ Finetuning: v2_pt_mlm_cpcl_ssp s1,2,3 ━━━"
for SEED in "${SEEDS[@]}"; do
    EXP="v2_pt_mlm_cpcl_ssp_s${SEED}"
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
