#!/bin/bash
#===============================================================================
# Pretraining Strategy Comparison  (EXP — Pretrain Comparison)
#
# 2-step pipeline:
#   [1] Pretraining  : seed=42, df_pretrain data → saves encoder weights
#   [2] Fine-tuning  : seeds 1,2,3 → load encoder → sites task
#
# Naming:
#   pretrain model : saved_models/circmac/ptcmp_{strategy}/42/pretrain/model.pth
#   finetune result: saved_models/circmac/ptcmp_{strategy}_s{seed}/{seed}/
#
# Strategies (8):
#   nopt     — No pretraining (fine-tune baseline)
#   mlm      — Masked Language Modeling
#   ntp      — Next Token Prediction
#   ssp      — Secondary Structure Prediction
#   pair     — Base Pairing Matrix
#   cpcl     — Circular Permutation Contrastive Learning
#   mlm_ntp  — MLM + NTP
#   all      — All tasks (MLM+NTP+SSP+Pairing+CPCL+BSJ_MLM)
#
# Usage:
#   ./scripts/final_v2/run_pretrain_comparison.sh [GPU_ID]
#
# One GPU, ~30h total on A100
#===============================================================================
set -e

GPU=${1:-0}
SEEDS=(1 2 3)
TASK="sites"
PT_SEED=42

# ── Model architecture (must match EXP1/EXP4) ─────────────────────────────────
D_MODEL=128; N_LAYER=6; MAX_LEN=1022

# ── Pretraining hyperparams ────────────────────────────────────────────────────
DATA_FILE="df_pretrain"
PT_BS=64; PT_LR=1e-3; PT_WD=0.01; PT_EP=300; PT_ES=30; NUM_WORKERS=4

# ── Fine-tuning hyperparams (same as EXP1 base) ────────────────────────────────
FT_BS=64; FT_LR=1e-4; FT_EP=150; FT_ES=20

PREFIX="ptcmp"
mkdir -p "logs/${PREFIX}/pretrain" "logs/${PREFIX}/finetune" saved_models
TOTAL=0; SKIPPED=0; RAN=0

echo "=== Pretrain Comparison (GPU $GPU) ==="
echo "    PT: BS=$PT_BS LR=$PT_LR epochs=$PT_EP es=$PT_ES"
echo "    FT: BS=$FT_BS LR=$FT_LR epochs=$FT_EP es=$FT_ES"

# ── Helper: run pretraining ─────────────────────────────────────────────────────
run_pretrain() {
    local STRATEGY=$1; shift   # remaining args = task flags
    local PT_EXP="${PREFIX}_pt_${STRATEGY}"
    TOTAL=$((TOTAL + 1))
    local PT_MODEL="saved_models/circmac/${PT_EXP}/${PT_SEED}/pretrain/model.pth"
    if [ -f "$PT_MODEL" ]; then
        echo "[SKIP] pretrain: $PT_EXP"
        SKIPPED=$((SKIPPED + 1))
        return 0
    fi
    RAN=$((RAN + 1))
    echo "[RUN]  pretrain: $PT_EXP"
    python pretraining.py \
        --model_name circmac \
        --data_file "$DATA_FILE" \
        --max_len $MAX_LEN \
        --d_model $D_MODEL --n_layer $N_LAYER \
        --batch_size $PT_BS --num_workers $NUM_WORKERS \
        --optimizer adamw --lr $PT_LR --w_decay $PT_WD \
        --epochs $PT_EP --earlystop $PT_ES \
        --device $GPU --exp "$PT_EXP" --seed $PT_SEED \
        --verbose "$@" \
        2>&1 | tee "logs/${PREFIX}/pretrain/${PT_EXP}.log"
}

# ── Helper: run fine-tuning ─────────────────────────────────────────────────────
run_finetune() {
    local STRATEGY=$1
    local PT_PATH=$2    # "none" for no-pretrain
    for SEED in "${SEEDS[@]}"; do
        local EXP="${PREFIX}_${STRATEGY}_s${SEED}"
        TOTAL=$((TOTAL + 1))
        if find "saved_models/circmac/${EXP}" -name "training.json" 2>/dev/null | grep -q .; then
            echo "[SKIP] finetune: $EXP"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi
        RAN=$((RAN + 1))
        echo "[RUN]  finetune: $EXP"
        FT_CMD="python training.py \
            --model_name circmac --task $TASK --seed $SEED \
            --d_model $D_MODEL --n_layer $N_LAYER --max_len $MAX_LEN \
            --batch_size $FT_BS --num_workers $NUM_WORKERS \
            --lr $FT_LR --epochs $FT_EP --earlystop $FT_ES \
            --device $GPU --exp $EXP \
            --interaction cross_attention --verbose"
        [ "$PT_PATH" != "none" ] && FT_CMD="$FT_CMD --load_pretrained $PT_PATH"
        eval "$FT_CMD" 2>&1 | tee "logs/${PREFIX}/finetune/${EXP}.log"
    done
}

# ── [1/8] No Pretraining baseline ──────────────────────────────────────────────
echo ""
echo "--- [1/8] No PT (fine-tune baseline) ---"
run_finetune "nopt" "none"

# ── [2/8] MLM ──────────────────────────────────────────────────────────────────
echo ""
echo "--- [2/8] MLM ---"
run_pretrain "mlm" --mlm
PT_PATH="saved_models/circmac/${PREFIX}_pt_mlm/${PT_SEED}/pretrain/model.pth"
[ -f "$PT_PATH" ] && run_finetune "mlm" "$PT_PATH"

# ── [3/8] NTP ──────────────────────────────────────────────────────────────────
echo ""
echo "--- [3/8] NTP ---"
run_pretrain "ntp" --ntp
PT_PATH="saved_models/circmac/${PREFIX}_pt_ntp/${PT_SEED}/pretrain/model.pth"
[ -f "$PT_PATH" ] && run_finetune "ntp" "$PT_PATH"

# ── [4/8] SSP ──────────────────────────────────────────────────────────────────
echo ""
echo "--- [4/8] SSP ---"
run_pretrain "ssp" --ssp
PT_PATH="saved_models/circmac/${PREFIX}_pt_ssp/${PT_SEED}/pretrain/model.pth"
[ -f "$PT_PATH" ] && run_finetune "ssp" "$PT_PATH"

# ── [5/8] Pairing ──────────────────────────────────────────────────────────────
echo ""
echo "--- [5/8] Pairing ---"
run_pretrain "pair" --pairing
PT_PATH="saved_models/circmac/${PREFIX}_pt_pair/${PT_SEED}/pretrain/model.pth"
[ -f "$PT_PATH" ] && run_finetune "pair" "$PT_PATH"

# ── [6/8] CPCL ─────────────────────────────────────────────────────────────────
echo ""
echo "--- [6/8] CPCL ---"
run_pretrain "cpcl" --cpcl
PT_PATH="saved_models/circmac/${PREFIX}_pt_cpcl/${PT_SEED}/pretrain/model.pth"
[ -f "$PT_PATH" ] && run_finetune "cpcl" "$PT_PATH"

# ── [7/8] MLM + NTP ────────────────────────────────────────────────────────────
echo ""
echo "--- [7/8] MLM+NTP ---"
run_pretrain "mlm_ntp" --mlm --ntp
PT_PATH="saved_models/circmac/${PREFIX}_pt_mlm_ntp/${PT_SEED}/pretrain/model.pth"
[ -f "$PT_PATH" ] && run_finetune "mlm_ntp" "$PT_PATH"

# ── [8/8] All tasks ────────────────────────────────────────────────────────────
echo ""
echo "--- [8/8] All (MLM+NTP+SSP+Pairing+CPCL+BSJ_MLM) ---"
run_pretrain "all" --mlm --ntp --ssp --pairing --cpcl --bsj_mlm
PT_PATH="saved_models/circmac/${PREFIX}_pt_all/${PT_SEED}/pretrain/model.pth"
[ -f "$PT_PATH" ] && run_finetune "all" "$PT_PATH"

echo ""
echo "=== Done: $RAN ran, $SKIPPED skipped / $TOTAL total ==="
