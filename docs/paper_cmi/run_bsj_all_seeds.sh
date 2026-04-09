#!/bin/bash
# ================================================================
# BSJ Analysis — Seed 1/2/3 모두 실행
#
# 각 seed별로 analyze_bsj.py를 실행하고 CSV를 수집한 뒤,
# plot_bsj_from_csv.py로 평균 + 오차범위 그래프를 생성합니다.
#
# 사용법:
#   cd ~/experiment/circMAC-CMI   (프로젝트 루트)
#   bash docs/paper_cmi/run_bsj_all_seeds.sh [device] [out_dir] [data_path]
#
# 예시:
#   bash docs/paper_cmi/run_bsj_all_seeds.sh 0 docs/paper_cmi/results_bsj /data/df_test_final.pkl
#
# 완료 후 CSV + 그림을 이 서버로 복사:
#   scp -r [server]:/path/cmi_mac/docs/paper_cmi/results_bsj \
#       /workspace/volume/cmi_mac/docs/paper_cmi/
# ================================================================
set -e

DEVICE=${1:-0}
OUT_DIR=${2:-docs/paper_cmi/results_bsj}
DATA_PATH=${3:-/path/to/df_test_final.pkl}   # ← 서버 실제 경로로 수정

mkdir -p "$OUT_DIR"

echo "============================================================"
echo " BSJ Analysis (Seed 1/2/3)  |  device=$DEVICE"
echo " out_dir=$OUT_DIR"
echo " data=$DATA_PATH"
echo "============================================================"

# ── 공통 모델 순서 ──────────────────────────────────────────────
MODELS="circmac mamba hymba lstm transformer rnabert rnaernie rnafm rnamsm"

run_seed() {
    local S=$1
    local SEED_DIR="${OUT_DIR}/seed${S}"
    mkdir -p "$SEED_DIR"

    echo ""
    echo "--- Seed $S ---"

    python docs/paper_cmi/analyze_bsj.py \
        --models $MODELS \
        --exps \
            exp1_circmac_s${S}          exp1_mamba_s${S} \
            exp1_hymba_s${S}            exp1_lstm_s${S} \
            exp1_transformer_s${S} \
            exp3_rnabert_frozen_s${S}   exp3_rnaernie_frozen_s${S} \
            exp3_rnafm_frozen_s${S}     exp3_rnamsm_frozen_s${S} \
        --seeds $S $S $S $S $S $S $S $S $S \
        --device "$DEVICE" \
        --data_path "$DATA_PATH" \
        --out_dir "$SEED_DIR"

    echo "[Seed $S] Done → ${SEED_DIR}/bsj_analysis_results.csv"
}

# ── Seed 1 / 2 / 3 순서대로 실행 ───────────────────────────────
run_seed 1
run_seed 2
run_seed 3

# ── 3개 CSV 합쳐서 mean±std 그래프 생성 ─────────────────────────
echo ""
echo "--- Plotting (mean ± std across 3 seeds) ---"

CSV1="${OUT_DIR}/seed1/bsj_analysis_results.csv"
CSV2="${OUT_DIR}/seed2/bsj_analysis_results.csv"
CSV3="${OUT_DIR}/seed3/bsj_analysis_results.csv"

if [ -f "$CSV1" ] && [ -f "$CSV2" ] && [ -f "$CSV3" ]; then
    python docs/paper_cmi/plot_bsj_from_csv.py \
        --csv "$CSV1" "$CSV2" "$CSV3" \
        --out_dir "$OUT_DIR"
    echo ""
    echo "=== All Done! ==="
    echo "CSVs:    ${OUT_DIR}/seed{1,2,3}/bsj_analysis_results.csv"
    echo "Figures: ${OUT_DIR}/"
    ls -lh "${OUT_DIR}"/*.pdf "${OUT_DIR}"/*.png 2>/dev/null || true
else
    echo "[WARN] 일부 CSV가 없어서 통합 플롯 생략"
    echo "  존재하는 CSV로 수동 실행:"
    echo "  python docs/paper_cmi/plot_bsj_from_csv.py --csv $CSV1 $CSV2 $CSV3 --out_dir $OUT_DIR"
fi
