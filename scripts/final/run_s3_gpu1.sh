#!/bin/bash
#===============================================================================
# Server3 GPU1: EXP1 (Fair/Max trainable RNA) + EXP3 CircMAC-PT
#
# EXP1 fair trainable:  rnabert/rnaernie/rnafm/rnamsm @ max_len=438  (4×3=12)
# EXP1 max  trainable:  rnaernie/rnafm/rnamsm @ max_len=1022  (3×3=9)
#   ↑ rnafm/rnamsm max trainable OOM'd at full BS → BS=8
# EXP3 CircMAC-PT: circmac finetuned with best pretrained weights  (3 seeds)
#   → Requires BEST_PT_EXP arg (e.g., "exp2v4_pt_mlm") after exp2v4 completes
#
# Usage: ./scripts/final/run_s3_gpu1.sh [GPU_ID] [BEST_PT_EXP]
#===============================================================================
set -e

GPU=${1:-1}
BEST_PT_EXP=${2:-""}
SEEDS=(1 2 3)
TASK="sites"
LR=1e-4; EPOCHS=150; EARLYSTOP=20; NUM_WORKERS=4
D_MODEL=128; N_LAYER=6

mkdir -p logs/exp1 logs/exp3 saved_models
TOTAL=0; SKIPPED=0; RAN=0

echo "=== S3 GPU${GPU}: EXP1 Trainable RNA + EXP3 CircMAC-PT ==="

run_ft() {
    local MODEL=$1; local EXP=$2; local LOG=$3; shift 3
    TOTAL=$((TOTAL+1))
    if find "saved_models/${MODEL}/${EXP}" -name "training.json" 2>/dev/null | grep -q .; then
        echo "[SKIP] $EXP"; SKIPPED=$((SKIPPED+1)); return 0; fi
    RAN=$((RAN+1)); echo "[RUN]  $EXP"
    python training.py --model_name "$MODEL" --task $TASK \
        --d_model $D_MODEL --n_layer $N_LAYER \
        --lr $LR --epochs $EPOCHS --earlystop $EARLYSTOP \
        --num_workers $NUM_WORKERS --device $GPU \
        --exp "$EXP" --interaction cross_attention \
        --trainable_pretrained --verbose "$@" \
        2>&1 | tee "$LOG"
}

# ── EXP1D: Fair Trainable RNA (all at max_len=438) ────────────────────────────
echo "--- [1/3] EXP1D: Fair Trainable RNA (max_len=438) ---"
FAIR_MAX_LEN=438
declare -A FAIR_BS=( ["rnabert"]=32 ["rnaernie"]=32 ["rnafm"]=16 ["rnamsm"]=16 )
for MODEL in rnabert rnaernie rnafm rnamsm; do
    BS=${FAIR_BS[$MODEL]}
    for SEED in "${SEEDS[@]}"; do
        EXP="exp1_fair_trainable_${MODEL}_s${SEED}"
        run_ft "$MODEL" "$EXP" "logs/exp1/${EXP}.log" \
            --seed $SEED --batch_size $BS --max_len $FAIR_MAX_LEN
    done
done

# ── EXP1E: Max Trainable RNA (max_len=1022, capped by each model) ─────────────
# NOTE: rnafm/rnamsm OOM'd at large BS → use BS=8 on a clean dedicated GPU
echo "--- [2/3] EXP1E: Max Trainable RNA (max_len=1022) ---"
declare -A MAX_BS=( ["rnaernie"]=32 ["rnafm"]=8 ["rnamsm"]=8 )
for MODEL in rnaernie rnafm rnamsm; do
    BS=${MAX_BS[$MODEL]}
    for SEED in "${SEEDS[@]}"; do
        EXP="exp1_max_trainable_${MODEL}_s${SEED}"
        run_ft "$MODEL" "$EXP" "logs/exp1/${EXP}.log" \
            --seed $SEED --batch_size $BS --max_len 1022
    done
done

# ── EXP3: CircMAC-PT ──────────────────────────────────────────────────────────
echo "--- [3/3] EXP3: CircMAC-PT ---"
if [ -z "$BEST_PT_EXP" ]; then
    echo "[WARN] BEST_PT_EXP not specified. Skipping CircMAC-PT."
    echo "  → After exp2v4 completes, rerun:"
    echo "  → ./scripts/final/run_s3_gpu1.sh $GPU <best_pt_exp>"
    echo "  → e.g., exp2v4_pt_mlm  or  exp2v4_pt_ssp"
else
    PT_PATH="saved_models/circmac/${BEST_PT_EXP}/42/pretrain/model.pth"
    if [ ! -f "$PT_PATH" ]; then
        echo "[WARN] Pretrain model not found: $PT_PATH"
        echo "  → Verify exp2v4 pretraining completed for: $BEST_PT_EXP"
    else
        echo "  Using pretrained: $PT_PATH"
        for SEED in "${SEEDS[@]}"; do
            EXP="exp3_circmac_pt_s${SEED}"
            TOTAL=$((TOTAL+1))
            if find "saved_models/circmac/${EXP}" -name "training.json" 2>/dev/null | grep -q .; then
                echo "[SKIP] $EXP"; SKIPPED=$((SKIPPED+1)); continue; fi
            RAN=$((RAN+1)); echo "[RUN]  $EXP"
            python training.py --model_name circmac --task $TASK --seed $SEED \
                --d_model $D_MODEL --n_layer $N_LAYER --max_len 1022 \
                --batch_size 32 --num_workers $NUM_WORKERS \
                --lr $LR --epochs $EPOCHS --earlystop $EARLYSTOP \
                --device $GPU --exp $EXP \
                --load_pretrained "$PT_PATH" \
                --interaction cross_attention --verbose \
                2>&1 | tee "logs/exp3/${EXP}.log"
        done
    fi
fi

echo "=== S3 GPU${GPU} Done: $RAN ran, $SKIPPED skipped / $TOTAL total ==="
