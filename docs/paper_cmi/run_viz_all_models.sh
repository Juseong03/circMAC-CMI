#!/bin/bash
# 9개 모델 전체 binding site visualization 스크립트
#
# 사용법:
#   ./docs/paper_cmi/run_viz_all_models.sh [DEVICE] [SEED] [CIRC_ID] [BSJ_W]
#
# 예시:
#   ./docs/paper_cmi/run_viz_all_models.sh 0 3 "chr17|78802324"
#   ./docs/paper_cmi/run_viz_all_models.sh 0 3 "chr17|78802324" 30

DEVICE=${1:-0}
SEED=${2:-3}
CIRC_ID=${3:-"chr4|5565258"}
BSJ_W=${4:-20}

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
BASE_DIR="${ROOT_DIR}/docs/paper_cmi"

# 결과 폴더: results/{isoform_tag}_bsjw{BSJ_W}_s{SEED}/
CSV_TAG=$(echo "$CIRC_ID" | sed 's/[|,: /\\]/_/g')
RUN_TAG="${CSV_TAG}_bsjw${BSJ_W}_s${SEED}"
OUT_DIR="${BASE_DIR}/results/${RUN_TAG}"

mkdir -p "$OUT_DIR"

echo "========================================"
echo "  All-model Visualization"
echo "  device  : $DEVICE"
echo "  seed    : $SEED"
echo "  circ_id : $CIRC_ID"
echo "  bsj_w   : $BSJ_W"
echo "  out_dir : $OUT_DIR"
echo "========================================"

cd "$ROOT_DIR" || exit 1

# ── Step 1: 모든 모델 inference → CSV 저장 ────────────────────────────────
echo ""
echo "[Step 1] Running inference for all 9 models..."

python docs/paper_cmi/plot_binding_visualization.py \
    --with_pred \
    --circ_id "$CIRC_ID" \
    --model_dirs \
        circmac:./saved_models/circmac/exp4_full_s${SEED}/${SEED} \
        mamba:./saved_models/mamba/exp1_mamba_s${SEED}/${SEED} \
        lstm:./saved_models/lstm/exp1_lstm_s${SEED}/${SEED} \
        transformer:./saved_models/transformer/exp1_transformer_s${SEED}/${SEED} \
        hymba:./saved_models/hymba/exp1_hymba_s${SEED}/${SEED} \
        rnabert:./saved_models/rnabert/exp3_rnabert_frozen_s${SEED}/${SEED} \
        rnaernie:./saved_models/rnaernie/exp3_rnaernie_frozen_s${SEED}/${SEED} \
        rnamsm:./saved_models/rnamsm/exp3_rnamsm_frozen_s${SEED}/${SEED} \
        rnafm:./saved_models/rnafm/exp3_rnafm_frozen_s${SEED}/${SEED}

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
