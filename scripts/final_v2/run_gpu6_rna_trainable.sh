#!/bin/bash
#===============================================================================
# GPU 6 — EXP1 RNA Trainable (rnabert, rnaernie, rnamsm)
#
# NOTE: rnafm trainable은 논문 Figure에 없으므로 제외
#
# rnabert  (125M, max_len=438): bs=32  × 3 seeds (~4h each  = ~12h)
# rnaernie (125M, max_len=511): bs=32  × 3 seeds (~4h each  = ~12h)
# rnamsm   (100M, max_len=1022): bs=8  × 3 seeds (~8h each  = ~24h)
# 합계: 9 runs, 예상 ~35h  ← bottleneck
#
# 만약 GPU가 8개 이상이면 rnamsm을 GPU7로 분리 권장
#
# Usage: ./scripts/final_v2/run_gpu6_rna_trainable.sh [GPU_ID]
#===============================================================================
set -e

GPU=${1:-6}
SEEDS=(1 2 3)
TASK="sites"
LR=1e-4; EPOCHS=150; EARLYSTOP=20; NUM_WORKERS=4
D_MODEL=128; N_LAYER=6

mkdir -p logs/exp1 saved_models
TOTAL=0; SKIPPED=0; RAN=0

echo "=== GPU $GPU: EXP1 RNA Trainable (rnabert, rnaernie, rnamsm) ==="

declare -A ML_MAP=( [rnabert]=438 [rnaernie]=511 [rnamsm]=1022 )
declare -A BS_MAP=( [rnabert]=32  [rnaernie]=32  [rnamsm]=8    )

for MODEL in rnabert rnaernie rnamsm; do
    ML=${ML_MAP[$MODEL]}
    BS=${BS_MAP[$MODEL]}
    for SEED in "${SEEDS[@]}"; do
        EXP="exp1_${MODEL}_trainable_s${SEED}"
        TOTAL=$((TOTAL+1))
        if find "saved_models/${MODEL}/${EXP}" -name "training.json" 2>/dev/null | grep -q .; then
            echo "[SKIP] $EXP"; SKIPPED=$((SKIPPED+1)); continue; fi
        RAN=$((RAN+1)); echo "[RUN]  $EXP (max_len=$ML, bs=$BS)"
        python training.py \
            --model_name "$MODEL" --task $TASK --seed $SEED \
            --d_model $D_MODEL --n_layer $N_LAYER --max_len $ML \
            --batch_size $BS --num_workers $NUM_WORKERS \
            --lr $LR --epochs $EPOCHS --earlystop $EARLYSTOP \
            --device $GPU --exp "$EXP" \
            --trainable_pretrained \
            --interaction cross_attention --verbose \
            2>&1 | tee "logs/exp1/${EXP}.log"
    done
done

echo "=== Done: $RAN ran, $SKIPPED skipped / $TOTAL total ==="
