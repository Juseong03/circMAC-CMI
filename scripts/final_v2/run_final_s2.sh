#!/bin/bash
#===============================================================================
# Final Re-run — Script 2 (GPU 1)
#
# EXP2 Pretraining Strategy Comparison
#   Pretrain phase : 6 strategies × 1 seed (42) = 6 PT runs
#   Finetune phase : 7 strategies × 3 seeds    = 21 FT runs
#   (nopt = no pretrain, finetune only)
#   Total: 27 runs  (~20h on A100)
#
# Naming:
#   Pretrain model : v2_ptm_{strategy}  (seed=42)
#                    saved_models/circmac/v2_ptm_{strategy}/42/pretrain/model.pth
#   Finetune result: v2_pt_{strategy}_s{seed}
#
# Strategies:
#   nopt     — No pretraining (baseline)
#   mlm      — Masked Language Modeling
#   ntp      — Next Token Prediction
#   mlm_ssp  — MLM + Secondary Structure Prediction
#   mlm_cpcl — MLM + Circular Permutation Contrastive Learning
#   mlm_ntp  — MLM + NTP  (best in previous experiments)
#   all      — All: MLM+NTP+SSP+Pairing+CPCL+BSJ_MLM
#
# Finetune hyperparams (consistent with EXP1/EXP4/EXP5/EXP6):
#   BS=64, LR=1e-4, epochs=150, es=20
#
# Pretrain hyperparams:
#   BS=64, LR=1e-3, w_decay=0.01, epochs=300, es=30
#   d_model=128, n_layer=6, max_len=1022
#
# Usage: ./scripts/final_v2/run_final_s2.sh [GPU_ID]
#===============================================================================
set -e

GPU=${1:-0}
SEEDS=(1 2 3)
TASK="sites"
PT_SEED=42

D_MODEL=128; N_LAYER=6; MAX_LEN=1022; NUM_WORKERS=4

# Pretraining hyperparams
PT_DATA="df_pretrain"
PT_BS=64; PT_LR=1e-3; PT_WD=0.01; PT_EP=300; PT_ES=30

# Finetuning hyperparams (same as all other v2 experiments)
FT_BS=64; FT_LR=1e-4; FT_EP=150; FT_ES=20

mkdir -p logs/v2/ptm logs/v2/pt saved_models
TOTAL=0; SKIPPED=0; RAN=0

echo "========================================"
echo "  Final Re-run Script 2  (GPU $GPU)"
echo "  EXP2 Pretraining Strategy Comparison"
echo "  PT:  BS=$PT_BS LR=$PT_LR epochs=$PT_EP es=$PT_ES"
echo "  FT:  BS=$FT_BS LR=$FT_LR epochs=$FT_EP es=$FT_ES"
echo "========================================"

# ── Helper: pretraining ─────────────────────────────────────────────────────────
run_pretrain() {
    local STRATEGY=$1; shift   # remaining args: task flags (--mlm, --ntp, ...)
    local PT_EXP="v2_ptm_${STRATEGY}"
    TOTAL=$((TOTAL+1))
    local PT_MODEL="saved_models/circmac/${PT_EXP}/${PT_SEED}/pretrain/model.pth"
    if [ -f "$PT_MODEL" ]; then
        echo "[SKIP] pretrain: $PT_EXP"
        SKIPPED=$((SKIPPED+1))
        return 0
    fi
    RAN=$((RAN+1)); echo "[RUN]  pretrain: $PT_EXP"
    python pretraining.py \
        --model_name circmac \
        --data_file "$PT_DATA" \
        --max_len $MAX_LEN \
        --d_model $D_MODEL --n_layer $N_LAYER \
        --batch_size $PT_BS --num_workers $NUM_WORKERS \
        --optimizer adamw --lr $PT_LR --w_decay $PT_WD \
        --epochs $PT_EP --earlystop $PT_ES \
        --device $GPU --exp "$PT_EXP" --seed $PT_SEED \
        --verbose "$@" \
        2>&1 | tee "logs/v2/ptm/${PT_EXP}.log"
}

# ── Helper: finetuning ──────────────────────────────────────────────────────────
run_finetune() {
    local STRATEGY=$1
    local PT_PATH=$2    # "none" = no pretrain
    for SEED in "${SEEDS[@]}"; do
        local EXP="v2_pt_${STRATEGY}_s${SEED}"
        TOTAL=$((TOTAL+1))
        if find "saved_models/circmac/${EXP}" -name "training.json" 2>/dev/null | grep -q .; then
            echo "[SKIP] finetune: $EXP"; SKIPPED=$((SKIPPED+1)); continue; fi
        RAN=$((RAN+1)); echo "[RUN]  finetune: $EXP"
        FT_CMD="python training.py \
            --model_name circmac --task $TASK --seed $SEED \
            --d_model $D_MODEL --n_layer $N_LAYER --max_len $MAX_LEN \
            --batch_size $FT_BS --num_workers $NUM_WORKERS \
            --lr $FT_LR --epochs $FT_EP --earlystop $FT_ES \
            --device $GPU --exp $EXP \
            --interaction cross_attention --verbose"
        [ "$PT_PATH" != "none" ] && FT_CMD="$FT_CMD --load_pretrained $PT_PATH"
        eval "$FT_CMD" 2>&1 | tee "logs/v2/pt/${EXP}.log"
    done
}

# ══ [1/7] No Pretraining (baseline) ═══════════════════════════════════════════
echo ""
echo "━━━ [1/7] No PT (baseline) ━━━"
run_finetune "nopt" "none"

# ══ [2/7] MLM ════════════════════════════════════════════════════════════════
echo ""
echo "━━━ [2/7] MLM ━━━"
run_pretrain "mlm" --mlm
PT_PATH="saved_models/circmac/v2_ptm_mlm/${PT_SEED}/pretrain/model.pth"
[ -f "$PT_PATH" ] && run_finetune "mlm" "$PT_PATH"

# ══ [3/7] NTP ════════════════════════════════════════════════════════════════
echo ""
echo "━━━ [3/7] NTP ━━━"
run_pretrain "ntp" --ntp
PT_PATH="saved_models/circmac/v2_ptm_ntp/${PT_SEED}/pretrain/model.pth"
[ -f "$PT_PATH" ] && run_finetune "ntp" "$PT_PATH"

# ══ [4/7] MLM + SSP ══════════════════════════════════════════════════════════
echo ""
echo "━━━ [4/7] MLM+SSP ━━━"
run_pretrain "mlm_ssp" --mlm --ssp
PT_PATH="saved_models/circmac/v2_ptm_mlm_ssp/${PT_SEED}/pretrain/model.pth"
[ -f "$PT_PATH" ] && run_finetune "mlm_ssp" "$PT_PATH"

# ══ [5/7] MLM + CPCL ═════════════════════════════════════════════════════════
echo ""
echo "━━━ [5/7] MLM+CPCL ━━━"
run_pretrain "mlm_cpcl" --mlm --cpcl
PT_PATH="saved_models/circmac/v2_ptm_mlm_cpcl/${PT_SEED}/pretrain/model.pth"
[ -f "$PT_PATH" ] && run_finetune "mlm_cpcl" "$PT_PATH"

# ══ [6/7] MLM + NTP ══════════════════════════════════════════════════════════
echo ""
echo "━━━ [6/7] MLM+NTP ━━━"
run_pretrain "mlm_ntp" --mlm --ntp
PT_PATH="saved_models/circmac/v2_ptm_mlm_ntp/${PT_SEED}/pretrain/model.pth"
[ -f "$PT_PATH" ] && run_finetune "mlm_ntp" "$PT_PATH"

# ══ [7/7] All tasks ═══════════════════════════════════════════════════════════
echo ""
echo "━━━ [7/7] All (MLM+NTP+SSP+Pairing+CPCL+BSJ_MLM) ━━━"
run_pretrain "all" --mlm --ntp --ssp --pairing --cpcl --bsj_mlm
PT_PATH="saved_models/circmac/v2_ptm_all/${PT_SEED}/pretrain/model.pth"
[ -f "$PT_PATH" ] && run_finetune "all" "$PT_PATH"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Done: $RAN ran, $SKIPPED skipped / $TOTAL total"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
