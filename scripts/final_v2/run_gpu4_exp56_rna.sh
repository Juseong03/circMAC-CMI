#!/bin/bash
#===============================================================================
# GPU 4 — EXP5 + EXP6 + RNA Frozen (rnabert, rnaernie)
#
# EXP5 Interaction: cross_attn, concat, elementwise × 3 seeds = 9 runs
# EXP6 Site Head:   conv1d, linear                 × 3 seeds = 6 runs
# EXP1 RNA Frozen:  rnabert, rnaernie              × 3 seeds = 6 runs
# 합계: 21 runs, 예상 ~20h
#
# Usage: ./scripts/final_v2/run_gpu4_exp56_rna.sh [GPU_ID]
#===============================================================================
set -e

GPU=${1:-4}
SEEDS=(1 2 3)
TASK="sites"
LR=1e-4; EPOCHS=150; EARLYSTOP=20; NUM_WORKERS=4
D_MODEL=128; N_LAYER=6; MAX_LEN=1022; BS=64

mkdir -p logs/exp5 logs/exp6 logs/exp1 saved_models
TOTAL=0; SKIPPED=0; RAN=0

echo "=== GPU $GPU: EXP5 + EXP6 + RNA Frozen (rnabert/rnaernie) ==="

# ── EXP5: Interaction ─────────────────────────────────────────────────────────
echo "--- EXP5: Interaction ---"
for INTERACTION in cross_attn concat elementwise; do
    IARG=$INTERACTION
    [ "$INTERACTION" = "cross_attn" ] && IARG="cross_attention"
    for SEED in "${SEEDS[@]}"; do
        EXP="exp5_${INTERACTION}_s${SEED}"
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
            2>&1 | tee "logs/exp5/${EXP}.log"
    done
done

# ── EXP6: Site Head ───────────────────────────────────────────────────────────
echo "--- EXP6: Site Head ---"
for HEAD in conv1d linear; do
    for SEED in "${SEEDS[@]}"; do
        EXP="exp6_${HEAD}_s${SEED}"
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
            2>&1 | tee "logs/exp6/${EXP}.log"
    done
done

# ── EXP1 RNA Frozen: rnabert, rnaernie ────────────────────────────────────────
echo "--- EXP1 RNA Frozen: rnabert, rnaernie ---"
declare -A ML_MAP=( [rnabert]=438 [rnaernie]=511 )
for MODEL in rnabert rnaernie; do
    ML=${ML_MAP[$MODEL]}
    for SEED in "${SEEDS[@]}"; do
        EXP="exp1_${MODEL}_frozen_s${SEED}"
        TOTAL=$((TOTAL+1))
        if find "saved_models/${MODEL}/${EXP}" -name "training.json" 2>/dev/null | grep -q .; then
            echo "[SKIP] $EXP"; SKIPPED=$((SKIPPED+1)); continue; fi
        RAN=$((RAN+1)); echo "[RUN]  $EXP (max_len=$ML)"
        python training.py \
            --model_name "$MODEL" --task $TASK --seed $SEED \
            --d_model $D_MODEL --n_layer $N_LAYER --max_len $ML \
            --batch_size 128 --num_workers $NUM_WORKERS \
            --lr $LR --epochs $EPOCHS --earlystop $EARLYSTOP \
            --device $GPU --exp "$EXP" \
            --interaction cross_attention --verbose \
            2>&1 | tee "logs/exp1/${EXP}.log"
    done
done

echo "=== Done: $RAN ran, $SKIPPED skipped / $TOTAL total ==="
