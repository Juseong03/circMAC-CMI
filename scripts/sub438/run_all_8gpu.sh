#!/bin/bash
#===============================================================================
# SUB438 — 8 GPU 병렬 실행 마스터 스크립트
#
# GPU 배치:
#   GPU 0: RNAErnie frozen
#   GPU 1: RNAErnie fine-tuned
#   GPU 2: RNAMSM frozen
#   GPU 3: RNAMSM fine-tuned
#   GPU 4: RNA-FM frozen
#   GPU 5: RNA-FM fine-tuned
#   GPU 6: CircMAC no pretrain
#   GPU 7: CircMAC + pairing pretrained
#
# Usage: ./scripts/sub438/run_all_8gpu.sh
#   각 GPU script는 백그라운드로 실행됨
#===============================================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")/$(basename "$(dirname "$SCRIPT_DIR")")" 2>/dev/null || true
# 프로젝트 루트로 이동
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

echo "=============================="
echo " SUB438 — 8 GPU 병렬 실행"
echo " $(date)"
echo "=============================="

bash "$SCRIPT_DIR/gpu0_rnaernie_frozen.sh"  0 > logs/sub438/gpu0.log 2>&1 &
bash "$SCRIPT_DIR/gpu1_rnaernie_ft.sh"      1 > logs/sub438/gpu1.log 2>&1 &
bash "$SCRIPT_DIR/gpu2_rnamsm_frozen.sh"    2 > logs/sub438/gpu2.log 2>&1 &
bash "$SCRIPT_DIR/gpu3_rnamsm_ft.sh"        3 > logs/sub438/gpu3.log 2>&1 &
bash "$SCRIPT_DIR/gpu4_rnafm_frozen.sh"     4 > logs/sub438/gpu4.log 2>&1 &
bash "$SCRIPT_DIR/gpu5_rnafm_ft.sh"         5 > logs/sub438/gpu5.log 2>&1 &
bash "$SCRIPT_DIR/gpu6_circmac_nopt.sh"     6 > logs/sub438/gpu6.log 2>&1 &
bash "$SCRIPT_DIR/gpu7_circmac_pairing.sh"  7 > logs/sub438/gpu7.log 2>&1 &

echo "모든 GPU 백그라운드 실행 시작"
echo "로그 확인: tail -f logs/sub438/gpu{0..7}.log"
echo ""
echo "진행 상황 모니터링:"
echo "  watch -n 30 'grep -h \"\\[RUN\\]\\|\\[SKIP\\]\\|Done\" logs/sub438/gpu*.log'"

wait
echo ""
echo "=============================="
echo " 전체 완료: $(date)"
echo "=============================="
