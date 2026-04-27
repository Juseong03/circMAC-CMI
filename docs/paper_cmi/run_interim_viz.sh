#!/bin/bash
#===============================================================================
# 중간 결과 시각화 스크립트
#
# 완료된 모델만 자동 탐지하여 binding site 시각화 실행
# (training.json 존재 여부로 완료 판단)
#
# Usage:
#   ./docs/paper_cmi/run_interim_viz.sh [DEVICE] [SEED] [CIRC_ID] [BSJ_W]
#
# 예시:
#   ./docs/paper_cmi/run_interim_viz.sh 0 3 "chr17|78802324" 20
#   ./docs/paper_cmi/run_interim_viz.sh 0 3 "chr10|103439746" 20
#===============================================================================

DEVICE=${1:-0}
SEED=${2:-3}
CIRC_ID=${3:-"chr17|78802324"}
BSJ_W=${4:-20}

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
BASE_DIR="${ROOT_DIR}/docs/paper_cmi"
LOG_DIR="${ROOT_DIR}/logs"
SAVED_DIR="${ROOT_DIR}/saved_models"

CSV_TAG=$(echo "$CIRC_ID" | sed 's/[|,: /\\]/_/g')
RUN_TAG="${CSV_TAG}_bsjw${BSJ_W}_s${SEED}"
OUT_DIR="${BASE_DIR}/results/${RUN_TAG}"
mkdir -p "$OUT_DIR"

echo "========================================"
echo "  Interim Visualization"
echo "  device  : $DEVICE"
echo "  seed    : $SEED"
echo "  circ_id : $CIRC_ID"
echo "  bsj_w   : $BSJ_W"
echo "  out_dir : $OUT_DIR"
echo "========================================"

# ── 완료된 모델 자동 탐지 ───────────────────────────────────────────────────
# 각 모델의 새 naming(exp1/exp4) → 구 naming(exp3) 순으로 탐색

check_model() {
    local MODEL=$1 EXP_NEW=$2 EXP_OLD=$3
    local new_log="${LOG_DIR}/${MODEL}/${EXP_NEW}_s${SEED}/${SEED}/training.json"
    local old_log="${LOG_DIR}/${MODEL}/${EXP_OLD}_s${SEED}/${SEED}/training.json"
    local new_ckpt="${SAVED_DIR}/${MODEL}/${EXP_NEW}_s${SEED}/${SEED}"
    local old_ckpt="${SAVED_DIR}/${MODEL}/${EXP_OLD}_s${SEED}/${SEED}"

    # checkpoint 경로 우선
    if [ -d "$new_ckpt" ]; then echo "$new_ckpt"; return 0; fi
    if [ -d "$old_ckpt" ]; then echo "$old_ckpt"; return 0; fi
    # log만 있고 checkpoint 없는 경우 (다른 서버에 저장된 경우)
    if [ -f "$new_log" ]; then echo "${new_ckpt}"; return 0; fi
    if [ -f "$old_log" ]; then echo "${old_ckpt}"; return 0; fi
    echo ""
}

MODEL_DIRS_ARG=""

add_model() {
    local LABEL=$1 MODEL=$2 EXP_NEW=$3 EXP_OLD=$4
    local PATH=$(check_model "$MODEL" "$EXP_NEW" "$EXP_OLD")
    if [ -n "$PATH" ]; then
        echo "  [✓] $LABEL → $PATH"
        MODEL_DIRS_ARG="$MODEL_DIRS_ARG ${LABEL}:${PATH}"
    else
        echo "  [ ] $LABEL — not ready (skip)"
    fi
}

echo ""
echo "[Auto-detect] Checking completed models (seed=$SEED)..."
add_model "circmac"    "circmac"     "exp4_full"          "exp1_circmac"
add_model "mamba"      "mamba"       "exp1_mamba"         "exp1_mamba"
add_model "lstm"       "lstm"        "exp1_lstm"          "exp1_lstm"
add_model "transformer" "transformer" "exp1_transformer"  "exp1_transformer"
add_model "hymba"      "hymba"       "exp1_hymba"         "exp1_hymba"
add_model "rnabert"    "rnabert"     "exp1_rnabert_frozen" "exp3_rnabert_frozen"
add_model "rnaernie"   "rnaernie"    "exp1_rnaernie_frozen" "exp3_rnaernie_frozen"
add_model "rnamsm"     "rnamsm"      "exp1_rnamsm_frozen"  "exp3_rnamsm_frozen"
add_model "rnafm"      "rnafm"       "exp1_rnafm_frozen"   "exp3_rnafm_frozen"

if [ -z "$MODEL_DIRS_ARG" ]; then
    echo ""
    echo "ERROR: 완료된 모델이 없습니다."
    exit 1
fi

echo ""
echo "[Step 1] Running inference for available models..."

cd "$ROOT_DIR" || exit 1

# shellcheck disable=SC2086
python docs/paper_cmi/plot_binding_visualization.py \
    --with_pred \
    --circ_id "$CIRC_ID" \
    --model_dirs $MODEL_DIRS_ARG

# CSV 이동
CSV_SRC="${BASE_DIR}/binding_visualization_${CSV_TAG}_with_pred.csv"
CSV_FILE="${OUT_DIR}/binding_visualization_${CSV_TAG}_with_pred.csv"

if [ ! -f "$CSV_SRC" ]; then
    echo "ERROR: CSV not found: $CSV_SRC"
    exit 1
fi

mv "$CSV_SRC" "$CSV_FILE"
for f in "${BASE_DIR}/binding_visualization_${CSV_TAG}"*.pdf \
          "${BASE_DIR}/binding_visualization_${CSV_TAG}"*.png; do
    [ -f "$f" ] && mv "$f" "$OUT_DIR/"
done

echo ""
echo "[Step 2] Generating visualizations from CSV..."

python docs/paper_cmi/plot_from_csv.py \
    --csv "$CSV_FILE" \
    --isoform "$CIRC_ID" \
    --top_mirna 12 \
    --bsj_w "$BSJ_W" \
    --zoom_w 50 \
    --out_dir "$OUT_DIR" \
    --plots all

echo ""
echo "========================================"
echo "  Done! → $OUT_DIR"
echo "========================================"
