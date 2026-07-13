#!/bin/bash
#===============================================================================
# Pretraining: MLM + Pairing  (pair split finetune)
#
# Step 1: Pretrain circmac  → saved_models/circmac/v2_ptm_mlm_pairing/42/pretrain/model.pth
# Step 2: Finetune × seeds  → saved_models/circmac/v2_pt_mlm_pairing_s{seed}/
#
# Usage: bash scripts/final_v2/run_pretrain_mlm_pairing.sh <GPU>
# SEEDS_OVERRIDE="1 2 3" bash scripts/final_v2/run_pretrain_mlm_pairing.sh <GPU>
#===============================================================================
set -e

GPU=${1:-0}
if [ -n "${SEEDS_OVERRIDE:-}" ]; then
    read -r -a SEEDS <<< "$SEEDS_OVERRIDE"
else
    SEEDS=(1 2 3 4 5 6 7 8 9 10)
fi

TASK="sites"; PT_SEED=42
D_MODEL=128; N_LAYER=6; MAX_LEN=1022; NUM_WORKERS=4
PT_DATA="df_pretrain"
PT_BS=64; PT_LR=1e-3; PT_WD=0.01; PT_EP=300; PT_ES=30
FT_BS=64; FT_LR=1e-4; FT_EP=150; FT_ES=20

mkdir -p logs/v2/ptm logs/v2/pt saved_models

echo "=== Pretrain: MLM+Pairing → Finetune (GPU=$GPU seeds=${SEEDS[*]}) ==="

# ── Step 1: Pretraining ────────────────────────────────────────────────────────
PT_EXP="v2_ptm_mlm_pairing"
PT_MODEL="saved_models/circmac/${PT_EXP}/${PT_SEED}/pretrain/model.pth"
if [ -f "$PT_MODEL" ]; then
    echo "[SKIP] pretrain: $PT_EXP"
else
    echo "[RUN]  pretrain: $PT_EXP"
    python pretraining.py \
        --model_name circmac \
        --data_file "$PT_DATA" \
        --max_len $MAX_LEN \
        --d_model $D_MODEL --n_layer $N_LAYER \
        --batch_size $PT_BS --num_workers $NUM_WORKERS \
        --optimizer adamw --lr $PT_LR --w_decay $PT_WD \
        --epochs $PT_EP --earlystop $PT_ES \
        --device $GPU --exp "$PT_EXP" --seed $PT_SEED \
        --mlm --pairing \
        --verbose \
        2>&1 | tee "logs/v2/ptm/${PT_EXP}.log"
fi

# ── Step 2: Finetuning ─────────────────────────────────────────────────────────
PT_PATH="saved_models/circmac/${PT_EXP}/${PT_SEED}/pretrain/model.pth"
if [ ! -f "$PT_PATH" ]; then
    echo "[ERROR] Pretrain checkpoint not found: $PT_PATH"
    exit 1
fi

for SEED in "${SEEDS[@]}"; do
    EXP="v2_pt_mlm_pairing_s${SEED}"
    CKPT=$(find saved_models/circmac/${EXP} -name "model.pth" 2>/dev/null | head -1)
    if [ -n "$CKPT" ]; then echo "  [SKIP] $EXP"; continue; fi
    echo "  [RUN]  $EXP"
    python training.py \
        --model_name circmac --task $TASK --seed $SEED \
        --d_model $D_MODEL --n_layer $N_LAYER --max_len $MAX_LEN \
        --batch_size $FT_BS --num_workers $NUM_WORKERS \
        --lr $FT_LR --epochs $FT_EP --earlystop $FT_ES \
        --device $GPU --exp "$EXP" \
        --interaction cross_attention \
        --load_pretrained "$PT_PATH" \
        --verbose \
        2>&1 | tee "logs/v2/pt/${EXP}.log"
done

echo "=== MLM+Pairing done ==="
