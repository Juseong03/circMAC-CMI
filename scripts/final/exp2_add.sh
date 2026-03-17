#!/bin/bash
#===============================================================================
# Exp2 Additional: NTP necessity + BSJ_MLM + fixed CPCL
#
# Background:
#   - MLM alone < No-PT → MLM solo 효과 없음
#   - MLM+NTP = best (77.12 F1) → NTP 중요
#   - MLM+CPCL ≈ No-PT → CPCL mean pooling bug → token-level로 수정
#
# New configs (5 × 4 runs = 20 runs):
#   1. ntp          : NTP alone (MLM 없어도 되는지 확인)
#   2. mlm_bsj      : MLM + BSJ_MLM (BSJ 효과 확인)
#   3. mlm_ntp_bsj  : MLM + NTP + BSJ_MLM (현재 best 확장)
#   4. mlm_cpcl_fix : MLM + CPCL (token-level fixed)
#   5. mlm_ntp_cpcl : MLM + NTP + CPCL (token-level fixed)
#
# Usage: ./scripts/final/exp2_add.sh [GPU_ID]
#===============================================================================
set -e

GPU=${1:-0}
SEEDS=(1 2 3)
TASK="sites"; PT_SEED=42; PREFIX="exp2"

DATA_FILE="df_circ_ss_5"; D_MODEL=128; N_LAYER=6; MAX_LEN=1022; NUM_WORKERS=4
PT_BS=64;  PT_LR=5e-4; PT_WD=0.01; PT_EP=500; PT_ES=50
FT_BS=32; FT_LR=1e-4; FT_EP=150; FT_ES=20

mkdir -p logs/exp2/pretrain logs/exp2/finetune saved_models

TOTAL=0; SKIPPED=0; RAN=0

echo "=============================================="
echo "  Exp2 Additional Configs"
echo "  GPU: $GPU | 5 configs × 4 runs = 20 runs"
echo "=============================================="

# ── Helper: pretrain + finetune ───────────────────────────────────────────────
run_config() {
    local CONFIG_NAME=$1
    local CONFIG_FLAGS=$2

    echo ""
    echo "--- Config: ${CONFIG_NAME} ---"

    # Pretrain
    PT_EXP="${PREFIX}_pt_${CONFIG_NAME}"
    PT_MODEL="saved_models/circmac/${PT_EXP}/${PT_SEED}/pretrain/model.pth"
    TOTAL=$((TOTAL + 1))
    if [ -f "$PT_MODEL" ]; then
        echo "[DONE] Pretrain: $PT_EXP"; SKIPPED=$((SKIPPED + 1))
    else
        RAN=$((RAN + 1))
        echo "[RUN]  Pretrain: $PT_EXP ($CONFIG_FLAGS)"
        python pretraining.py --model_name circmac --data_file $DATA_FILE --max_len $MAX_LEN \
            --d_model $D_MODEL --n_layer $N_LAYER --batch_size $PT_BS --num_workers $NUM_WORKERS \
            --optimizer adamw --lr $PT_LR --w_decay $PT_WD \
            --epochs $PT_EP --earlystop $PT_ES --device $GPU \
            --exp "$PT_EXP" --seed $PT_SEED --verbose \
            $CONFIG_FLAGS \
            2>&1 | tee "logs/exp2/pretrain/${PT_EXP}.log"
    fi

    # Finetune (3 seeds)
    PT_PATH="saved_models/circmac/${PT_EXP}/${PT_SEED}/pretrain/model.pth"
    if [ ! -f "$PT_PATH" ]; then
        echo "[SKIP] Pretrain model not found: $PT_PATH"; return
    fi
    for SEED in "${SEEDS[@]}"; do
        TOTAL=$((TOTAL + 1))
        EXP_NAME="${PREFIX}_${CONFIG_NAME}_${TASK}_s${SEED}"
        if find "saved_models/circmac/${EXP_NAME}" -name "training.json" 2>/dev/null | grep -q .; then
            echo "[DONE] Finetune: $EXP_NAME"; SKIPPED=$((SKIPPED + 1)); continue
        fi
        RAN=$((RAN + 1))
        echo "[RUN]  Finetune: $EXP_NAME"
        python training.py --model_name circmac --task "$TASK" --seed "$SEED" \
            --d_model $D_MODEL --n_layer $N_LAYER --batch_size $FT_BS --num_workers $NUM_WORKERS \
            --lr $FT_LR --epochs $FT_EP --earlystop $FT_ES --device $GPU \
            --exp "$EXP_NAME" --load_pretrained "$PT_PATH" \
            --interaction cross_attention --verbose \
            2>&1 | tee "logs/exp2/finetune/${EXP_NAME}.log"
    done
}

# ── 5 Configs ─────────────────────────────────────────────────────────────────

# 1. NTP alone (MLM 없이 NTP만)
run_config "ntp" "--ntp"

# 2. MLM + BSJ_MLM
run_config "mlm_bsj" "--mlm --bsj_mlm"

# 3. MLM + NTP + BSJ_MLM  (현재 best인 MLM+NTP에 BSJ_MLM 추가)
run_config "mlm_ntp_bsj" "--mlm --ntp --bsj_mlm"

# 4. MLM + CPCL (token-level fixed)
run_config "mlm_cpcl_fix" "--mlm --cpcl"

# 5. MLM + NTP + CPCL (token-level fixed)
run_config "mlm_ntp_cpcl" "--mlm --ntp --cpcl"

# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=============================================="
echo "  Exp2 Add Complete: $RAN ran, $SKIPPED skipped / $TOTAL total"
echo "=============================================="
echo ""
echo "  Compare against baselines:"
echo "  No-PT    : F1=75.30, AUROC=90.00"
echo "  MLM      : F1=75.00, AUROC=89.46"
echo "  MLM+NTP  : F1=77.12, AUROC=90.90  ← current best"
