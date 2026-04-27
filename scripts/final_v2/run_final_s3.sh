#!/bin/bash
#===============================================================================
# Final Re-run — Script 3 (GPU 2)
#
# EXP4 CircMAC Ablation Study
#   9 variants × 3 seeds = 27 runs  (~35h on A100)
#
# Naming: v2_abl_{variant}_s{seed}
#
# Variants (8):
#   full         — Full CircMAC (cross_attention, all branches)
#   no_attn      — Disable Attention branch
#   no_mamba     — Disable Mamba branch
#   no_conv      — Disable Conv branch
#   no_circ_bias — Disable circular relative bias in attention (--no_circular_rel_bias)
#   attn_only    — Attention branch only
#   mamba_only   — Mamba branch only
#   cnn_only     — CNN branch only
#
# NOTE: no_circ_pad excluded — flag not in training.py, effect negligible (0.7402 ≈ full)
#
# Hyperparams (consistent with all v2 experiments):
#   BS=64, LR=1e-4, epochs=150, es=20
#   d_model=128, n_layer=6, max_len=1022
#   interaction: cross_attention
#
# NOTE: v2_abl_full should match v2_enc_circmac (Script 1) as a sanity check.
#
# Usage: ./scripts/final_v2/run_final_s3.sh [GPU_ID]
#===============================================================================
set -e

GPU=${1:-0}
SEEDS=(1 2 3)
TASK="sites"

D_MODEL=128; N_LAYER=6; MAX_LEN=1022
BS=64; LR=1e-4; EPOCHS=150; EARLYSTOP=20; NUM_WORKERS=4

mkdir -p logs/v2/abl saved_models
TOTAL=0; SKIPPED=0; RAN=0

echo "========================================"
echo "  Final Re-run Script 3  (GPU $GPU)"
echo "  EXP4 CircMAC Ablation Study"
echo "  BS=$BS LR=$LR epochs=$EPOCHS es=$EARLYSTOP"
echo "========================================"

# ── Helper ─────────────────────────────────────────────────────────────────────
run_abl() {
    local VARIANT=$1; shift   # remaining args: ablation flags
    for SEED in "${SEEDS[@]}"; do
        local EXP="v2_abl_${VARIANT}_s${SEED}"
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
            --interaction cross_attention --verbose "$@" \
            2>&1 | tee "logs/v2/abl/${EXP}.log"
    done
}

# ══ [1/9] Full CircMAC (baseline) ════════════════════════════════════════════
echo ""
echo "━━━ [1/9] Full (CircMAC baseline) ━━━"
run_abl "full"

# ══ [2/9] No Attention branch ════════════════════════════════════════════════
echo ""
echo "━━━ [2/9] No Attention branch ━━━"
run_abl "no_attn" --no_attn

# ══ [3/9] No Mamba branch ════════════════════════════════════════════════════
echo ""
echo "━━━ [3/9] No Mamba branch ━━━"
run_abl "no_mamba" --no_mamba

# ══ [4/9] No Conv branch ═════════════════════════════════════════════════════
echo ""
echo "━━━ [4/9] No Conv branch ━━━"
run_abl "no_conv" --no_conv

# ══ [5/9] No Circular Relative Bias ══════════════════════════════════════════
echo ""
echo "━━━ [5/9] No Circular Relative Bias ━━━"
run_abl "no_circ_bias" --no_circular_rel_bias

# ══ [6/8] Attention Only ══════════════════════════════════════════════════════
echo ""
echo "━━━ [6/8] Attention Only ━━━"
run_abl "attn_only" --no_mamba --no_conv

# ══ [7/8] Mamba Only ═════════════════════════════════════════════════════════
echo ""
echo "━━━ [7/8] Mamba Only ━━━"
run_abl "mamba_only" --no_attn --no_conv

# ══ [8/8] CNN Only ═══════════════════════════════════════════════════════════
echo ""
echo "━━━ [8/8] CNN Only ━━━"
run_abl "cnn_only" --no_attn --no_mamba

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Done: $RAN ran, $SKIPPED skipped / $TOTAL total"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
