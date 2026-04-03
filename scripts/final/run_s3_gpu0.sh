#!/bin/bash
#===============================================================================
# Server3 GPU0: EXP1 (Base encoders + Fair/Max frozen RNA) + EXP4/5/6
#
# EXP1 base:   circmac, hymba, lstm, mamba, transformer  (5×3=15)
# EXP1 fair:   rnabert/rnaernie/rnafm/rnamsm frozen @ max_len=438  (4×3=12)
# EXP1 max:    rnaernie/rnafm/rnamsm frozen @ max_len=1022  (3×3=9)
# EXP4:        Ablation (8 configs × 3 seeds = 24)
# EXP5:        Interaction (3 configs × 3 seeds = 9)
# EXP6:        Site head (2 configs × 3 seeds = 6)
#
# Usage: ./scripts/final/run_s3_gpu0.sh [GPU_ID]
#===============================================================================
set -e

GPU=${1:-0}
SEEDS=(1 2 3)
TASK="sites"
LR=1e-4; EPOCHS=150; EARLYSTOP=20; NUM_WORKERS=4
D_MODEL=128; N_LAYER=6

mkdir -p logs/exp1 logs/exp4 logs/exp5 logs/exp6 saved_models
TOTAL=0; SKIPPED=0; RAN=0

echo "=== S3 GPU${GPU}: EXP1/4/5/6 ==="

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
        --exp "$EXP" --interaction cross_attention --verbose "$@" \
        2>&1 | tee "$LOG"
}

# ── EXP1A: Base Encoders ──────────────────────────────────────────────────────
echo "--- [1/6] EXP1A: Base Encoders ---"
for MODEL in lstm transformer mamba hymba circmac; do
    BS=128; [ "$MODEL" = "transformer" ] && BS=64
    for SEED in "${SEEDS[@]}"; do
        EXP="exp1_${MODEL}_s${SEED}"
        run_ft "$MODEL" "$EXP" "logs/exp1/${EXP}.log" \
            --seed $SEED --batch_size $BS --max_len 1022
    done
done

# ── EXP1B: Fair Frozen RNA (all at max_len=438 for fair rnabert comparison) ──
echo "--- [2/6] EXP1B: Fair Frozen RNA (max_len=438) ---"
FAIR_MAX_LEN=438
for MODEL in rnabert rnaernie rnafm rnamsm; do
    BS=128; [ "$MODEL" = "rnafm" ] || [ "$MODEL" = "rnamsm" ] && BS=64
    for SEED in "${SEEDS[@]}"; do
        EXP="exp1_fair_frozen_${MODEL}_s${SEED}"
        run_ft "$MODEL" "$EXP" "logs/exp1/${EXP}.log" \
            --seed $SEED --batch_size $BS --max_len $FAIR_MAX_LEN
    done
done

# ── EXP1C: Max Frozen RNA (max_len=1022, capped by each model's limit) ────────
echo "--- [3/6] EXP1C: Max Frozen RNA (max_len=1022) ---"
for MODEL in rnaernie rnafm rnamsm; do
    BS=64; [ "$MODEL" = "rnafm" ] || [ "$MODEL" = "rnamsm" ] && BS=32
    for SEED in "${SEEDS[@]}"; do
        EXP="exp1_max_frozen_${MODEL}_s${SEED}"
        run_ft "$MODEL" "$EXP" "logs/exp1/${EXP}.log" \
            --seed $SEED --batch_size $BS --max_len 1022
    done
done

# ── EXP4: Ablation ────────────────────────────────────────────────────────────
echo "--- [4/6] EXP4: Ablation ---"
mkdir -p logs/exp4
declare -a ABL_CONFIGS=(
    "full                 "
    "no_attn              --no_attn"
    "no_mamba             --no_mamba"
    "no_conv              --no_conv"
    "no_circ_bias         --no_circular_rel_bias"
    "attn_only            --no_mamba --no_conv"
    "mamba_only           --no_attn --no_conv"
    "cnn_only             --no_attn --no_mamba"
)
for cfg in "${ABL_CONFIGS[@]}"; do
    CFG_NAME=$(echo "$cfg" | awk '{print $1}')
    CFG_FLAGS=$(echo "$cfg" | sed "s/^${CFG_NAME}[[:space:]]*//" | sed 's/^[[:space:]]*//')
    for SEED in "${SEEDS[@]}"; do
        EXP="exp4_${CFG_NAME}_s${SEED}"
        TOTAL=$((TOTAL+1))
        if find "saved_models/circmac/${EXP}" -name "training.json" 2>/dev/null | grep -q .; then
            echo "[SKIP] $EXP"; SKIPPED=$((SKIPPED+1)); continue; fi
        RAN=$((RAN+1)); echo "[RUN]  $EXP  [$CFG_FLAGS]"
        eval python training.py --model_name circmac --task $TASK --seed $SEED \
            --d_model $D_MODEL --n_layer $N_LAYER --max_len 1022 \
            --batch_size 128 --num_workers $NUM_WORKERS \
            --lr $LR --epochs $EPOCHS --earlystop $EARLYSTOP \
            --device $GPU --exp $EXP \
            --interaction cross_attention --verbose $CFG_FLAGS \
            2>&1 | tee "logs/exp4/${EXP}.log"
    done
done

# ── EXP5: Interaction ─────────────────────────────────────────────────────────
echo "--- [5/6] EXP5: Interaction ---"
mkdir -p logs/exp5
for INTERACTION in concat elementwise cross_attention; do
    for SEED in "${SEEDS[@]}"; do
        EXP="exp5_${INTERACTION}_s${SEED}"
        TOTAL=$((TOTAL+1))
        if find "saved_models/circmac/${EXP}" -name "training.json" 2>/dev/null | grep -q .; then
            echo "[SKIP] $EXP"; SKIPPED=$((SKIPPED+1)); continue; fi
        RAN=$((RAN+1)); echo "[RUN]  $EXP"
        python training.py --model_name circmac --task $TASK --seed $SEED \
            --d_model $D_MODEL --n_layer $N_LAYER --max_len 1022 \
            --batch_size 128 --num_workers $NUM_WORKERS \
            --lr $LR --epochs $EPOCHS --earlystop $EARLYSTOP \
            --device $GPU --exp $EXP \
            --interaction "$INTERACTION" --verbose \
            2>&1 | tee "logs/exp5/${EXP}.log"
    done
done

# ── EXP6: Site Head ───────────────────────────────────────────────────────────
echo "--- [6/6] EXP6: Site Head ---"
mkdir -p logs/exp6
for HEAD in conv1d linear; do
    for SEED in "${SEEDS[@]}"; do
        EXP="exp6_${HEAD}_s${SEED}"
        TOTAL=$((TOTAL+1))
        if find "saved_models/circmac/${EXP}" -name "training.json" 2>/dev/null | grep -q .; then
            echo "[SKIP] $EXP"; SKIPPED=$((SKIPPED+1)); continue; fi
        RAN=$((RAN+1)); echo "[RUN]  $EXP"
        python training.py --model_name circmac --task $TASK --seed $SEED \
            --d_model $D_MODEL --n_layer $N_LAYER --max_len 1022 \
            --batch_size 128 --num_workers $NUM_WORKERS \
            --lr $LR --epochs $EPOCHS --earlystop $EARLYSTOP \
            --device $GPU --exp $EXP \
            --interaction cross_attention \
            --site_head_type "$HEAD" --verbose \
            2>&1 | tee "logs/exp6/${EXP}.log"
    done
done

echo "=== S3 GPU${GPU} Done: $RAN ran, $SKIPPED skipped / $TOTAL total ==="
