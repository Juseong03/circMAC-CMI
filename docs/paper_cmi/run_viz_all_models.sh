#!/bin/bash
# 9개 모델 전체 binding site visualization 스크립트
#
# 모델 경로 규칙:
#   base 5종  : saved_models/{model}/v2_enc_{model}_s{SEED}/{SEED}/
#   RNA LM 4종: saved_models/{model}/exp1_{model}_frozen_s{SEED}/{SEED}/
#
# 사용법:
#   ./docs/paper_cmi/run_viz_all_models.sh [DEVICE] [SEED] [CIRC_ID] [BSJ_W]
#
# 예시:
#   ./docs/paper_cmi/run_viz_all_models.sh 0 1 "chr4|5565258"
#   ./docs/paper_cmi/run_viz_all_models.sh 0 1 "chr17|78802324" 30

DEVICE=${1:-0}
SEED=${2:-1}
CIRC_ID=${3:-"chr4|5565258"}
BSJ_W=${4:-20}
THRESHOLD=${5:-0.5}    # pred binarization threshold for region overlap
IOU_THRESH=${6:-0.3}   # IoU >= this → GT site detected
NO_PDF=${7:-0}         # 1 이면 PDF 저장 안 함

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
BASE_DIR="${ROOT_DIR}/docs/paper_cmi"

# 결과 폴더: results/{isoform_tag}_bsjw{BSJ_W}_s{SEED}/
CSV_TAG=$(echo "$CIRC_ID" | sed 's/[|,: /\\]/_/g')
RUN_TAG="${CSV_TAG}_bsjw${BSJ_W}_s${SEED}"
OUT_DIR="${BASE_DIR}/results/${RUN_TAG}"

mkdir -p "$OUT_DIR"

echo "========================================"
echo "  All-model Visualization"
echo "  device    : $DEVICE"
echo "  seed      : $SEED"
echo "  circ_id   : $CIRC_ID"
echo "  bsj_w     : $BSJ_W"
echo "  threshold : $THRESHOLD"
echo "  iou_thresh: $IOU_THRESH"
echo "  out_dir   : $OUT_DIR"
echo "========================================"

cd "$ROOT_DIR" || exit 1

# ── 모델 경로 확인 (없으면 경고만 출력, 계속 진행) ──────────────────────────
check_model() {
    local label=$1 path=$2
    if [ ! -d "$path" ]; then
        echo "  [WARN] $label not found: $path"
        return 1
    fi
    return 0
}

MODEL_ARGS=()
add_model() {
    local label=$1 path=$2
    if check_model "$label" "$path"; then
        MODEL_ARGS+=("${label}:${path}")
    fi
}

echo ""
echo "[Check] Model paths..."
add_model "circmac"     "./saved_models/circmac/v2_enc_circmac_s${SEED}/${SEED}"
add_model "mamba"       "./saved_models/mamba/v2_enc_mamba_s${SEED}/${SEED}"
add_model "lstm"        "./saved_models/lstm/v2_enc_lstm_s${SEED}/${SEED}"
add_model "transformer" "./saved_models/transformer/v2_enc_transformer_s${SEED}/${SEED}"
add_model "hymba"       "./saved_models/hymba/v2_enc_hymba_s${SEED}/${SEED}"
add_model "rnabert"     "./saved_models/rnabert/exp1_rnabert_frozen_s${SEED}/${SEED}"
add_model "rnaernie"    "./saved_models/rnaernie/exp1_rnaernie_frozen_s${SEED}/${SEED}"
add_model "rnamsm"      "./saved_models/rnamsm/exp1_rnamsm_frozen_s${SEED}/${SEED}"
add_model "rnafm"       "./saved_models/rnafm/exp1_rnafm_frozen_s${SEED}/${SEED}"

if [ ${#MODEL_ARGS[@]} -eq 0 ]; then
    echo "ERROR: No valid model paths found. Check saved_models/ and SEED=$SEED."
    exit 1
fi
echo "  Found ${#MODEL_ARGS[@]} model(s): ${MODEL_ARGS[*]%%:*}"

# ── Step 1: 모든 모델 inference → CSV 저장 ────────────────────────────────
echo ""
echo "[Step 1] Running inference for ${#MODEL_ARGS[@]} models..."

python docs/paper_cmi/plot_binding_visualization.py \
    --with_pred \
    --circ_id "$CIRC_ID" \
    --model_dirs "${MODEL_ARGS[@]}"

# CSV는 plot_binding_visualization.py가 BASE_DIR에 저장함 → OUT_DIR로 이동
CSV_SRC="${BASE_DIR}/binding_visualization_${CSV_TAG}_with_pred.csv"
CSV_FILE="${OUT_DIR}/binding_visualization_${CSV_TAG}_with_pred.csv"

if [ ! -f "$CSV_SRC" ]; then
    echo "ERROR: CSV not found: $CSV_SRC"
    echo "Check if circ_id matched any isoform."
    exit 1
fi

mv "$CSV_SRC" "$CSV_FILE"

# PDF/PNG도 OUT_DIR로 이동
for f in "${BASE_DIR}/binding_visualization_${CSV_TAG}"*.pdf \
          "${BASE_DIR}/binding_visualization_${CSV_TAG}"*.png; do
    [ -f "$f" ] && mv "$f" "$OUT_DIR/"
done

echo ""
echo "[Step 2] Generating visualizations from CSV..."
echo "  CSV: $CSV_FILE"

PDF_FLAG=""
[ "$NO_PDF" = "1" ] && PDF_FLAG="--no_pdf"

python docs/paper_cmi/plot_from_csv.py \
    --csv "$CSV_FILE" \
    --isoform "$CIRC_ID" \
    --top_mirna 12 \
    --bsj_w "$BSJ_W" \
    --zoom_w 50 \
    --threshold "$THRESHOLD" \
    --iou_thresh "$IOU_THRESH" \
    --out_dir "$OUT_DIR" \
    --plots all \
    $PDF_FLAG

echo ""
echo "========================================"
echo "  Done! Files saved to:"
echo "  $OUT_DIR"
echo ""
echo "  binding_visualization_*.csv      - raw predictions"
echo "  viz_heatmap_*.pdf/png            - GT + 모델별 히트맵"
echo "  viz_overlay_*.pdf/png            - 다중 모델 오버레이"
echo "  viz_model_{name}_*.pdf/png       - 모델별 개별 그림"
echo "  viz_bsj_zoom_*.pdf/png           - BSJ 근처 확대"
echo "  viz_model_summary_*.pdf/png      - 모델간 비교 bar chart"
echo "========================================"
echo ""
echo "  TIP: seed 1,2,3 중 원하는 것 지정 가능"
echo "    ./docs/paper_cmi/run_viz_all_models.sh 0 2 \"chr4|5565258\""
echo "========================================"
