#!/bin/bash
# update_models_for_viz.sh
# saved_models/ → models_for_viz/ 최신 모델 복사 후 ROC 재계산
#
# Usage:
#   ./figures_paper/fig_roc_curves/update_models_for_viz.sh [GPU]
#   GPU: GPU index (default: 0)

set -e

GPU=${1:-0}
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

SEEDS=(1 2 3)
COPIED=0; MISSING=0

echo "=========================================="
echo " Update models_for_viz from saved_models"
echo "=========================================="

copy_model() {
    local MODEL_NAME=$1   # e.g. circmac
    local EXP_TPL=$2      # e.g. v2_abl_full
    local SEED=$3

    local EXP="${EXP_TPL}_s${SEED}"
    local SRC="saved_models/${MODEL_NAME}/${EXP}/${SEED}/train/model.pth"
    local DST="models_for_viz/${MODEL_NAME}/${EXP}/${SEED}/train/model.pth"

    if [ ! -f "$SRC" ]; then
        echo "  [MISSING] $SRC"
        MISSING=$((MISSING+1))
        return
    fi

    mkdir -p "$(dirname "$DST")"
    cp "$SRC" "$DST"
    COPIED=$((COPIED+1))
    echo "  [OK] $DST"
}

echo ""
echo "--- Encoder models ---"
for SEED in "${SEEDS[@]}"; do
    copy_model "lstm"        "v2_enc_lstm"        $SEED
    copy_model "transformer" "v2_enc_transformer" $SEED
    copy_model "mamba"       "v2_enc_mamba"       $SEED
    copy_model "hymba"       "v2_enc_hymba"       $SEED
    copy_model "circmac"     "v2_abl_full"        $SEED
done

echo ""
echo "--- Pretrained models ---"
for SEED in "${SEEDS[@]}"; do
    copy_model "rnabert"  "exp1_fair_trainable_rnabert"  $SEED
    copy_model "rnaernie" "exp1_fair_trainable_rnaernie" $SEED
    copy_model "rnamsm"   "exp1_fair_trainable_rnamsm"   $SEED
    copy_model "rnafm"    "exp1_fair_trainable_rnafm"    $SEED
    copy_model "circmac"  "v2_pt_pairing"                $SEED
done

echo ""
echo "--- Frozen models (for rna_lm group) ---"
for SEED in "${SEEDS[@]}"; do
    copy_model "rnabert"  "exp1_fair_frozen_rnabert"  $SEED
    copy_model "rnaernie" "exp1_fair_frozen_rnaernie" $SEED
    copy_model "rnamsm"   "exp1_fair_frozen_rnamsm"   $SEED
    copy_model "rnafm"    "exp1_fair_frozen_rnafm"    $SEED
done

echo ""
echo "=========================================="
echo " Copy done: $COPIED copied, $MISSING missing"
echo "=========================================="

if [ "$MISSING" -gt 0 ]; then
    echo " [WARN] Some models missing — ROC will skip them."
fi

echo ""
echo "=========================================="
echo " Running ROC computation — encoder"
echo "=========================================="
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python figures_paper/fig_roc_curves/compute_roc_data.py \
    --device "$GPU" --group encoder

echo ""
echo "=========================================="
echo " Running ROC computation — pretrained"
echo "=========================================="
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python figures_paper/fig_roc_curves/compute_roc_data.py \
    --device "$GPU" --group pretrained

echo ""
echo "=========================================="
echo " Generating figures"
echo "=========================================="
python figures_paper/fig_roc_curves/fig_roc_curves.py

echo ""
echo "Done! Figures → figures_paper/fig_roc_curves/"
