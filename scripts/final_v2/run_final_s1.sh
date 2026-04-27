#!/bin/bash
#===============================================================================
# Final Re-run — Script 1 (GPU 0)
#
# EXP1 Base Encoder Comparison  : 5 models × 3 seeds = 15 runs
# EXP5 Interaction Mechanism    : 3 types  × 3 seeds =  9 runs
# EXP6 Site Prediction Head     : 2 types  × 3 seeds =  6 runs
# Total: 30 runs  (~15h on A100)
#
# Naming: v2_enc_{model}_s{seed}
#         v2_int_{type}_s{seed}
#         v2_head_{type}_s{seed}
#
# Hyperparams (all consistent):
#   BS=64, LR=1e-4, epochs=150, es=20
#   d_model=128, n_layer=6, max_len=1022
#   interaction: cross_attention (default for EXP1, EXP6)
#   site_head_type: conv1d (default for EXP1, EXP5)
#
# NOTE: v2_enc_circmac = CircMAC baseline used as reference in EXP4/EXP5/EXP6
#
# Usage: ./scripts/final_v2/run_final_s1.sh [GPU_ID]
#===============================================================================
set -e

GPU=${1:-0}
SEEDS=(1 2 3)
TASK="sites"

D_MODEL=128; N_LAYER=6; MAX_LEN=1022
BS=64; LR=1e-4; EPOCHS=150; EARLYSTOP=20; NUM_WORKERS=4

mkdir -p logs/v2/enc logs/v2/int logs/v2/head saved_models
TOTAL=0; SKIPPED=0; RAN=0

echo "========================================"
echo "  Final Re-run Script 1  (GPU $GPU)"
echo "  EXP1 Base + EXP5 Interaction + EXP6 Head"
echo "  BS=$BS LR=$LR epochs=$EPOCHS es=$EARLYSTOP"
echo "========================================"

# ── Helper ─────────────────────────────────────────────────────────────────────
run_ft() {
    local MODEL=$1 EXP=$2 LOG=$3; shift 3
    TOTAL=$((TOTAL+1))
    if find "saved_models/${MODEL}/${EXP}" -name "training.json" 2>/dev/null | grep -q .; then
        echo "[SKIP] $EXP"; SKIPPED=$((SKIPPED+1)); return 0; fi
    RAN=$((RAN+1)); echo "[RUN]  $EXP"
    python training.py \
        --model_name "$MODEL" --task $TASK \
        --d_model $D_MODEL --n_layer $N_LAYER --max_len $MAX_LEN \
        --batch_size $BS --num_workers $NUM_WORKERS \
        --lr $LR --epochs $EPOCHS --earlystop $EARLYSTOP \
        --device $GPU --exp "$EXP" \
        --interaction cross_attention --verbose "$@" \
        2>&1 | tee "$LOG"
}

# ══ EXP1 — Base Encoder Comparison ════════════════════════════════════════════
echo ""
echo "━━━ EXP1: Base Encoder Comparison ━━━"

declare -A ML_MAP=(
    [circmac]=1022 [mamba]=1022 [hymba]=1022 [lstm]=1022 [transformer]=1022
)

for MODEL in circmac mamba hymba lstm transformer; do
    ML=${ML_MAP[$MODEL]}
    for SEED in "${SEEDS[@]}"; do
        EXP="v2_enc_${MODEL}_s${SEED}"
        run_ft "$MODEL" "$EXP" "logs/v2/enc/${EXP}.log" \
            --seed $SEED --max_len $ML
    done
done

# ══ EXP5 — Interaction Mechanism ══════════════════════════════════════════════
echo ""
echo "━━━ EXP5: Interaction Mechanism ━━━"
# cross_attn: v2_enc_circmac (이미 위에서 실행, 동일 설정)
# concat, elementwise: 추가 실행

for INTERACTION in cross_attn concat elementwise; do
    # cross_attention은 training.py에서 cross_attention으로 전달
    IARG="$INTERACTION"
    [ "$INTERACTION" = "cross_attn" ] && IARG="cross_attention"

    for SEED in "${SEEDS[@]}"; do
        EXP="v2_int_${INTERACTION}_s${SEED}"
        TOTAL=$((TOTAL+1))
        if find "saved_models/circmac/${EXP}" -name "training.json" 2>/dev/null | grep -q .; then
            echo "[SKIP] $EXP"; SKIPPED=$((SKIPPED+1)); continue; fi
        RAN=$((RAN+1)); echo "[RUN]  $EXP"
        python training.py \
            --model_name circmac --task $TASK --seed $SEED \
            --d_model $D_MODEL --n_layer $N_LAYER --max_len $MAX_LEN \
            --batch_size $BS --num_workers $NUM_WORKERS \
            --lr $LR --epochs $EPOCHS --earlystop $EARLYSTOP \
            --device $GPU --exp "$EXP" \
            --interaction "$IARG" --verbose \
            2>&1 | tee "logs/v2/int/${EXP}.log"
    done
done

# ══ EXP6 — Site Prediction Head ═══════════════════════════════════════════════
echo ""
echo "━━━ EXP6: Site Prediction Head ━━━"
# conv1d: v2_enc_circmac (동일 설정), linear: 추가 실행
# 단, 명확한 비교를 위해 conv1d도 별도 실행

for HEAD in conv1d linear; do
    for SEED in "${SEEDS[@]}"; do
        EXP="v2_head_${HEAD}_s${SEED}"
        TOTAL=$((TOTAL+1))
        if find "saved_models/circmac/${EXP}" -name "training.json" 2>/dev/null | grep -q .; then
            echo "[SKIP] $EXP"; SKIPPED=$((SKIPPED+1)); continue; fi
        RAN=$((RAN+1)); echo "[RUN]  $EXP"
        python training.py \
            --model_name circmac --task $TASK --seed $SEED \
            --d_model $D_MODEL --n_layer $N_LAYER --max_len $MAX_LEN \
            --batch_size $BS --num_workers $NUM_WORKERS \
            --lr $LR --epochs $EPOCHS --earlystop $EARLYSTOP \
            --device $GPU --exp "$EXP" \
            --interaction cross_attention \
            --site_head_type "$HEAD" --verbose \
            2>&1 | tee "logs/v2/head/${EXP}.log"
    done
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Done: $RAN ran, $SKIPPED skipped / $TOTAL total"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
