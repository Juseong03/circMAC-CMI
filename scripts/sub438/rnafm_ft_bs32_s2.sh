#!/bin/bash
# SUB438 — RNA-FM fine-tuned BS=32, seed 2
# Usage: ./scripts/sub438/rnafm_ft_bs32_s2.sh [GPU_ID]
set -e
GPU=${1:-0}
SEED=2; MODEL="rnafm"; MAX_LEN=438; BS=32
LR=1e-4; EPOCHS=150; EARLYSTOP=20; NUM_WORKERS=4
D_MODEL=128; N_LAYER=6; PREFIX="sub438"
EXP="${PREFIX}_${MODEL}_ft_bs32_s${SEED}"

mkdir -p "logs/${PREFIX}" saved_models
if find "saved_models/${MODEL}/${EXP}" -name "training.json" 2>/dev/null | grep -q .; then
    echo "[SKIP] $EXP"; exit 0
fi
echo "[RUN]  $EXP (GPU=$GPU)"
python training.py \
    --model_name "$MODEL" --task sites --seed $SEED \
    --d_model $D_MODEL --n_layer $N_LAYER --max_len $MAX_LEN \
    --batch_size $BS --num_workers $NUM_WORKERS \
    --lr $LR --epochs $EPOCHS --earlystop $EARLYSTOP \
    --device $GPU --exp "$EXP" \
    --trainable_pretrained \
    --interaction cross_attention --verbose \
    2>&1 | tee "logs/${PREFIX}/${EXP}.log"
