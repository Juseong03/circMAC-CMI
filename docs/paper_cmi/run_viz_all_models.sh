#!/bin/bash
# 9개 모델 전체 binding site visualization 스크립트
#
# 모델 경로 규칙:
#   base 5종  : saved_models/{model}/v2_enc_{model}_s{SEED}/{SEED}/
#   RNA LM 4종: saved_models/{model}/exp1_{model}_frozen_s{SEED}/{SEED}/
#
# 사용법:
#   ./docs/paper_cmi/run_viz_all_models.sh [DEVICE] [SEED] [CIRC_ID] [BSJ_W] \
#       [THRESHOLD] [IOU_THRESH] [NO_PDF] [SPLIT] [ALL_PAIRS]
#
# 예시:
#   # 기본 (test set, binding pair만)
#   ./docs/paper_cmi/run_viz_all_models.sh 0 1 "chr11|102114144"
#
#   # train set 사용
#   ./docs/paper_cmi/run_viz_all_models.sh 0 1 "chr11|102114144" 20 0.5 0.3 0 train
#
#   # train+test 합쳐서
#   ./docs/paper_cmi/run_viz_all_models.sh 0 1 "chr11|102114144" 20 0.5 0.3 0 all
#
#   # non-binding pair도 포함
#   ./docs/paper_cmi/run_viz_all_models.sh 0 1 "chr11|102114144" 20 0.5 0.3 0 test 1
#
#   # PDF 없이
#   ./docs/paper_cmi/run_viz_all_models.sh 0 1 "chr11|102114144" 20 0.5 0.3 1

DEVICE=${1:-0}
SEED=${2:-1}
CIRC_ID=${3:-"chr11|102114144"}
BSJ_W=${4:-20}
THRESHOLD=${5:-0.5}    # pred binarization threshold for region overlap
IOU_THRESH=${6:-0.3}   # IoU >= this → GT site detected
NO_PDF=${7:-0}         # 1 이면 PDF 저장 안 함
SPLIT=${8:-"test"}     # test | train | all
ALL_PAIRS=${9:-0}      # 0=binding only (default), 1=모든 pair 포함

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
BASE_DIR="${ROOT_DIR}/docs/paper_cmi"

# 결과 폴더: results/{isoform_tag}_{split}_{pairs}_bsjw{BSJ_W}_s{SEED}/
CSV_TAG=$(echo "$CIRC_ID" | sed 's/[|,: /\\]/_/g')
PAIR_TAG="binding_only"
[ "$ALL_PAIRS" = "1" ] && PAIR_TAG="all_pairs"
RUN_TAG="${CSV_TAG}_${SPLIT}_${PAIR_TAG}_bsjw${BSJ_W}_s${SEED}"
OUT_DIR="${BASE_DIR}/results/${RUN_TAG}"

mkdir -p "$OUT_DIR"

echo "========================================"
echo "  All-model Visualization"
echo "  device    : $DEVICE"
echo "  seed      : $SEED"
echo "  circ_id   : $CIRC_ID"
echo "  split     : $SPLIT"
echo "  pairs     : $PAIR_TAG"
echo "  bsj_w     : $BSJ_W"
echo "  threshold : $THRESHOLD"
echo "  iou_thresh: $IOU_THRESH"
echo "  no_pdf    : $NO_PDF"
echo "  out_dir   : $OUT_DIR"
echo "========================================"

cd "$ROOT_DIR" || exit 1

# ── 모델 경로 확인 (없으면 경고만 출력, 계속 진행) ──────────────────────────
MODEL_ARGS=()
add_model() {
    local label=$1 path=$2
    if [ -d "$path" ]; then
        MODEL_ARGS+=("${label}:${path}")
    else
        echo "  [WARN] $label not found: $path"
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
echo "  Found ${#MODEL_ARGS[@]} model(s): $(echo "${MODEL_ARGS[@]%%:*}" | tr ' ' ',')"

# ── Step 1: 모든 모델 inference → CSV 저장 ────────────────────────────────
echo ""
echo "[Step 1] Running inference  (split=$SPLIT, $PAIR_TAG)..."

ALL_PAIRS_FLAG=""
[ "$ALL_PAIRS" = "1" ] && ALL_PAIRS_FLAG="--all_pairs"

python docs/paper_cmi/plot_binding_visualization.py \
    --with_pred \
    --circ_id "$CIRC_ID" \
    --split "$SPLIT" \
    --out_dir "$OUT_DIR" \
    --model_dirs "${MODEL_ARGS[@]}" \
    $ALL_PAIRS_FLAG

CSV_FILE="${OUT_DIR}/binding_visualization_${CSV_TAG}_with_pred.csv"

if [ ! -f "$CSV_FILE" ]; then
    echo "ERROR: CSV not found: $CSV_FILE"
    echo "  circ_id가 데이터에 없거나, split=$SPLIT 에 해당 pair가 없습니다."
    echo "  사용 가능한 circRNA 확인:"
    echo "    python -c \"import pickle,pandas as pd; df=pickle.load(open('data/df_test_final.pkl','rb')); print(df[df['binding']==1]['isoform_ID'].unique()[:10])\""
    exit 1
fi

# ── Step 2: 시각화 생성 ────────────────────────────────────────────────────
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
echo "  Done! Files saved to: $OUT_DIR"
echo ""
echo "  binding_visualization_*.csv    - raw predictions"
echo "  viz_heatmap_*                  - GT + 모델별 히트맵"
echo "  viz_overlay_*                  - 다중 모델 오버레이"
echo "  viz_model_{name}_*             - 모델별 개별 그림"
echo "  viz_bsj_zoom_*                 - BSJ 근처 확대"
echo "  viz_region_overlap_*           - Site Recall / IoU / Coverage"
echo "========================================"
echo ""
echo "  SPLIT 옵션: test(기본) | train | all(train+test)"
echo "  PAIR 옵션 : binding_only(기본) | all_pairs(non-binding 포함)"
echo ""
echo "  예시:"
echo "    ./docs/paper_cmi/run_viz_all_models.sh 0 1 \"chr11|102114144\" 20 0.5 0.3 0 train"
echo "    ./docs/paper_cmi/run_viz_all_models.sh 0 1 \"chr11|102114144\" 20 0.5 0.3 0 all 1"
echo "========================================"
