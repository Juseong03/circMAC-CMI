#!/bin/bash
#===============================================================================
# RNA-LM Length-Stratified Experiments for ISO & BSJ Disjoint Splits
#
# 목적: iso/bsj에서 RNA-LM 길이별 공정 비교 (pair split의 fig_lm_length_comparison과 동일 방식)
#
# 실험 구성:
#   sub438: RNABERT, RNAErnie, RNAMSM, RNA-FM, CircMAC (max_len=438)
#   sub511: RNAErnie, RNAMSM, RNA-FM, CircMAC (max_len=511)
#   max   : 기존 실험 재사용 (iso/bsj pt_pairing, rnamsm_ft, rnafm_ft)
#
# GPU 배치 (8 GPU 사용):
#   GPU 0-4: sub438_iso
#   GPU 0-3: sub511_iso  (sub438 완료 후)
#   GPU 0-4: sub438_bsj  (별도 서버 또는 순차 실행)
#   GPU 0-3: sub511_bsj
#
# 권장 실행 방법 (8 GPU 서버):
#   [터미널 1] bash scripts/run_lm_length_disjoint.sh iso
#   [터미널 2] bash scripts/run_lm_length_disjoint.sh bsj
#
# 단일 터미널 순차 실행:
#   bash scripts/run_lm_length_disjoint.sh all
#===============================================================================
set -e
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
SCRIPT_DIR="scripts"

TARGET=${1:-all}

run_iso() {
    echo "========================================"
    echo " ISO-DISJOINT sub-length experiments"
    echo " $(date)"
    echo "========================================"
    echo ""
    echo "--- [1/2] sub438-ISO (GPU 0-4) ---"
    bash "$SCRIPT_DIR/sub438_iso/run_all.sh"
    echo ""
    echo "--- [2/2] sub511-ISO (GPU 0-3) ---"
    bash "$SCRIPT_DIR/sub511_iso/run_all.sh"
    echo ""
    echo "ISO 완료: $(date)"
}

run_bsj() {
    echo "========================================"
    echo " BSJ-DISJOINT sub-length experiments"
    echo " $(date)"
    echo "========================================"
    echo ""
    echo "--- [1/2] sub438-BSJ (GPU 0-4) ---"
    bash "$SCRIPT_DIR/sub438_bsj/run_all.sh"
    echo ""
    echo "--- [2/2] sub511-BSJ (GPU 0-3) ---"
    bash "$SCRIPT_DIR/sub511_bsj/run_all.sh"
    echo ""
    echo "BSJ 완료: $(date)"
}

case "$TARGET" in
    iso) run_iso ;;
    bsj) run_bsj ;;
    all)
        run_iso
        run_bsj
        ;;
    *)
        echo "Usage: $0 [iso|bsj|all]"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo " 모든 실험 완료: $(date)"
echo "========================================"
echo ""
echo "다음 단계:"
echo "  python scripts/logs2csv.py --only disjoint"
echo "  python figures_paper/update_figure_csvs.py --disjoint eval_results/disjoint_new_summary.csv"
