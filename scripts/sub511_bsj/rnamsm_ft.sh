#!/bin/bash
# SUB511 BSJ-DISJOINT — RNAMSM fine-tuned, seeds 1 2 3
# Usage: bash scripts/sub511_bsj/rnamsm_ft.sh [GPU_ID]
set -e
GPU=${1:-1}
MODEL="rnamsm"; MAX_LEN=511; BS=32; PREFIX="sub511_bsj"
LR=1e-4; EPOCHS=150; EARLYSTOP=20; NUM_WORKERS=4; D_MODEL=128; N_LAYER=6
TRAIN_FILE="./data/df_train_bsj_disjoint.pkl"
TEST_FILE="./data/df_test_bsj_disjoint.pkl"

mkdir -p "logs/${PREFIX}"
TOTAL=0; SKIPPED=0; RAN=0
echo "=== SUB511-BSJ ${MODEL} fine-tuned (GPU=$GPU) ==="

for SEED in 1 2 3; do
    EXP="${PREFIX}_${MODEL}_ft_s${SEED}"
    TOTAL=$((TOTAL+1))
    if find "saved_models/${MODEL}/${EXP}" -name "training.json" 2>/dev/null | grep -q .; then
        echo "  [SKIP] $EXP"; SKIPPED=$((SKIPPED+1)); continue
    fi
    RAN=$((RAN+1)); echo "  [RUN]  $EXP"
    python training.py \
        --model_name "$MODEL" --task sites --seed $SEED \
        --d_model $D_MODEL --n_layer $N_LAYER --max_len $MAX_LEN \
        --batch_size $BS --num_workers $NUM_WORKERS \
        --lr $LR --epochs $EPOCHS --earlystop $EARLYSTOP \
        --device $GPU --exp "$EXP" \
        --trainable_pretrained \
        --train_file "$TRAIN_FILE" --test_file "$TEST_FILE" \
        --interaction cross_attention --verbose \
        2>&1 | tee "logs/${PREFIX}/${EXP}.log"
done
echo "=== Done: $RAN ran, $SKIPPED skipped / $TOTAL total ==="
