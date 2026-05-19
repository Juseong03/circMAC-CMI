#!/bin/bash
#===============================================================================
# SUB438 — GPU7: CircMAC + pairing pretrained fine-tune  (seeds 1 2 3)
#
# pretrained weights 경로:
#   saved_models/circmac/v2_ptm_pairing/42/pretrain/model.pth
#   (서버에 없으면 먼저 run_pretrain_pairing_only.sh 실행 필요)
#
# Usage: ./scripts/sub438/gpu7_circmac_pairing.sh [GPU_ID]
#===============================================================================
set -e
GPU=${1:-7}

SEEDS=(1 2 3)
TASK="sites"
MODEL="circmac"
MAX_LEN=438
BS=128
LR=1e-4; EPOCHS=150; EARLYSTOP=20; NUM_WORKERS=4
D_MODEL=128; N_LAYER=6
PREFIX="sub438"

# Pairing pretrained weights (full-length pretraining 결과 재사용)
PT_SEED=42
PT_EXP="v2_ptm_pairing"
PT_PATH="saved_models/${MODEL}/${PT_EXP}/${PT_SEED}/pretrain/model.pth"

mkdir -p "logs/${PREFIX}" saved_models
TOTAL=0; SKIPPED=0; RAN=0

echo "=== SUB438 ${MODEL} pairing-pretrained (GPU $GPU) ==="

# pretrained weights 존재 여부 확인
if [ ! -f "$PT_PATH" ]; then
    echo "[ERROR] Pretrained weights not found: $PT_PATH"
    echo "        먼저 pretraining 실행:"
    echo "        python pretraining.py --model_name circmac --data_file df_pretrain \\"
    echo "          --task pairing --max_len 1022 --d_model 128 --n_layer 6 \\"
    echo "          --batch_size 64 --epochs 300 --earlystop 30 \\"
    echo "          --seed $PT_SEED --device $GPU --exp $PT_EXP"
    exit 1
fi

echo "  Pretrained weights: $PT_PATH"

for SEED in "${SEEDS[@]}"; do
    EXP="${PREFIX}_${MODEL}_pairing_s${SEED}"
    TOTAL=$((TOTAL+1))
    if find "saved_models/${MODEL}/${EXP}" -name "training.json" 2>/dev/null | grep -q .; then
        echo "[SKIP] $EXP"; SKIPPED=$((SKIPPED+1)); continue
    fi
    RAN=$((RAN+1)); echo "[RUN]  $EXP"
    python training.py \
        --model_name "$MODEL" --task $TASK --seed $SEED \
        --d_model $D_MODEL --n_layer $N_LAYER --max_len $MAX_LEN \
        --batch_size $BS --num_workers $NUM_WORKERS \
        --lr $LR --epochs $EPOCHS --earlystop $EARLYSTOP \
        --device $GPU --exp "$EXP" \
        --load_pretrained "$PT_PATH" \
        --interaction cross_attention --verbose \
        2>&1 | tee "logs/${PREFIX}/${EXP}.log"
done

echo "=== Done: $RAN ran, $SKIPPED skipped / $TOTAL total ==="
