#!/bin/bash
#===============================================================================
# Exp2 v3: Pretraining Strategy — Single-task comparison + All combined
#
# Experiments:
#   Single:   No-PT, MLM*, NTP, SSP, Pairing, CPCL       (* MLM: bug fix rerun)
#   Combined: MLM+NTP*, MLM+NTP+SSP+Pair+CPCL             (* rerun)
#
# Usage: ./scripts/final/exp2_v3.sh [GPU_ID]
#===============================================================================
set -e

GPU=${1:-0}
SEEDS=(1 2 3)
TASK="sites"; PT_SEED=42; PREFIX="exp2v3"

# Hyperparameters
DATA_FILE="df_circ_ss"; D_MODEL=128; N_LAYER=6; MAX_LEN=1022; NUM_WORKERS=4
PT_BS=64; PT_LR=5e-4; PT_WD=0.01; PT_EP=1000; PT_ES=100
FT_BS=32; FT_LR=1e-4; FT_EP=150; FT_ES=20

mkdir -p logs/exp2v3/pretrain logs/exp2v3/finetune saved_models

TOTAL=0; SKIPPED=0; RAN=0

echo "=== Exp2 v3: Pretraining Strategy Comparison (GPU $GPU) ==="
echo "    PT: EP=$PT_EP, ES=$PT_ES, LR=$PT_LR (cosine)"
echo "    FT: EP=$FT_EP, ES=$FT_ES"

# ── Helper: run pretrain ──────────────────────────────────────────────────────
run_pretrain() {
    local PT_EXP=$1; shift
    local PT_FLAGS="$@"
    TOTAL=$((TOTAL + 1))
    PT_MODEL="saved_models/circmac/${PT_EXP}/${PT_SEED}/pretrain/model.pth"
    if [ -f "$PT_MODEL" ]; then
        echo "[DONE] pretrain: $PT_EXP"; SKIPPED=$((SKIPPED + 1)); return 0
    fi
    RAN=$((RAN + 1))
    echo "[RUN]  pretrain: $PT_EXP ($PT_FLAGS)"
    python pretraining.py --model_name circmac --data_file $DATA_FILE --max_len $MAX_LEN \
        --d_model $D_MODEL --n_layer $N_LAYER --batch_size $PT_BS --num_workers $NUM_WORKERS \
        --optimizer adamw --lr $PT_LR --w_decay $PT_WD \
        --epochs $PT_EP --earlystop $PT_ES --device $GPU \
        --exp "$PT_EXP" --seed $PT_SEED --verbose \
        $PT_FLAGS \
        2>&1 | tee "logs/exp2v3/pretrain/${PT_EXP}.log"
}

# ── Helper: run finetune ──────────────────────────────────────────────────────
run_finetune() {
    local PT_NAME=$1
    local PT_PATH=$2
    for SEED in "${SEEDS[@]}"; do
        TOTAL=$((TOTAL + 1))
        EXP_NAME="${PREFIX}_${PT_NAME}_${TASK}_s${SEED}"
        if find "saved_models/circmac/${EXP_NAME}" -name "training.json" 2>/dev/null | grep -q .; then
            echo "[DONE] finetune: $EXP_NAME"; SKIPPED=$((SKIPPED + 1)); continue
        fi
        RAN=$((RAN + 1))
        echo "[RUN]  finetune: $EXP_NAME"
        FT_ARGS="python training.py --model_name circmac --task $TASK --seed $SEED \
            --d_model $D_MODEL --n_layer $N_LAYER --batch_size $FT_BS --num_workers $NUM_WORKERS \
            --lr $FT_LR --epochs $FT_EP --earlystop $FT_ES --device $GPU \
            --exp $EXP_NAME --interaction cross_attention --verbose"
        if [ "$PT_PATH" != "none" ]; then
            FT_ARGS="$FT_ARGS --load_pretrained $PT_PATH"
        fi
        eval "$FT_ARGS" 2>&1 | tee "logs/exp2v3/finetune/${EXP_NAME}.log"
    done
}

# ── 1. No PT Baseline ─────────────────────────────────────────────────────────
echo ""
echo "--- [1/8] No PT Baseline ---"
run_finetune "nopt" "none"

# ── 2. MLM (bug fix rerun) ────────────────────────────────────────────────────
echo ""
echo "--- [2/8] MLM ---"
run_pretrain "${PREFIX}_pt_mlm" --mlm
PT_PATH="saved_models/circmac/${PREFIX}_pt_mlm/${PT_SEED}/pretrain/model.pth"
[ -f "$PT_PATH" ] && run_finetune "mlm" "$PT_PATH" || echo "[SKIP] MLM pretrain not found"

# ── 3. NTP ────────────────────────────────────────────────────────────────────
echo ""
echo "--- [3/8] NTP ---"
run_pretrain "${PREFIX}_pt_ntp" --ntp
PT_PATH="saved_models/circmac/${PREFIX}_pt_ntp/${PT_SEED}/pretrain/model.pth"
[ -f "$PT_PATH" ] && run_finetune "ntp" "$PT_PATH" || echo "[SKIP] NTP pretrain not found"

# ── 4. SSP ────────────────────────────────────────────────────────────────────
echo ""
echo "--- [4/8] SSP ---"
run_pretrain "${PREFIX}_pt_ssp" --ssp
PT_PATH="saved_models/circmac/${PREFIX}_pt_ssp/${PT_SEED}/pretrain/model.pth"
[ -f "$PT_PATH" ] && run_finetune "ssp" "$PT_PATH" || echo "[SKIP] SSP pretrain not found"

# ── 5. Pairing ────────────────────────────────────────────────────────────────
echo ""
echo "--- [5/8] Pairing ---"
run_pretrain "${PREFIX}_pt_pair" --pairing
PT_PATH="saved_models/circmac/${PREFIX}_pt_pair/${PT_SEED}/pretrain/model.pth"
[ -f "$PT_PATH" ] && run_finetune "pair" "$PT_PATH" || echo "[SKIP] Pairing pretrain not found"

# ── 6. CPCL ───────────────────────────────────────────────────────────────────
echo ""
echo "--- [6/8] CPCL ---"
run_pretrain "${PREFIX}_pt_cpcl" --cpcl
PT_PATH="saved_models/circmac/${PREFIX}_pt_cpcl/${PT_SEED}/pretrain/model.pth"
[ -f "$PT_PATH" ] && run_finetune "cpcl" "$PT_PATH" || echo "[SKIP] CPCL pretrain not found"

# ── 7. MLM+NTP (bug fix rerun) ────────────────────────────────────────────────
echo ""
echo "--- [7/8] MLM+NTP ---"
run_pretrain "${PREFIX}_pt_mlm_ntp" --mlm --ntp
PT_PATH="saved_models/circmac/${PREFIX}_pt_mlm_ntp/${PT_SEED}/pretrain/model.pth"
[ -f "$PT_PATH" ] && run_finetune "mlm_ntp" "$PT_PATH" || echo "[SKIP] MLM+NTP pretrain not found"

# ── 8. All Combined ───────────────────────────────────────────────────────────
echo ""
echo "--- [8/8] All Combined (MLM+NTP+SSP+Pair+CPCL) ---"
run_pretrain "${PREFIX}_pt_all" --mlm --ntp --ssp --pairing --cpcl
PT_PATH="saved_models/circmac/${PREFIX}_pt_all/${PT_SEED}/pretrain/model.pth"
[ -f "$PT_PATH" ] && run_finetune "all" "$PT_PATH" || echo "[SKIP] All pretrain not found"

echo ""
echo "=== Exp2 v3 Complete: $RAN ran, $SKIPPED skipped / $TOTAL total ==="
