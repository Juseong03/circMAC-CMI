#!/bin/bash
#===============================================================================
# SUB438 ISO-DISJOINT — 5 GPU 병렬 실행
#
# GPU 배치:
#   GPU 0: RNABERT  ft  (max_len=438)
#   GPU 1: RNAErnie ft  (max_len=438)
#   GPU 2: RNAMSM   ft  (max_len=438)
#   GPU 3: RNA-FM   ft  (max_len=438)
#   GPU 4: CircMAC  pairing (max_len=438)
#
# Usage: bash scripts/sub438_iso/run_all.sh
#   또는 GPU 오프셋 지정: GPU_OFFSET=2 bash scripts/sub438_iso/run_all.sh
#===============================================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

OFFSET=${GPU_OFFSET:-0}
mkdir -p logs/sub438_iso

echo "=============================="
echo " SUB438-ISO — 5 GPU 병렬 실행"
echo " GPU offset: $OFFSET"
echo " $(date)"
echo "=============================="

bash "$SCRIPT_DIR/rnabert_ft.sh"       $((OFFSET+0)) > logs/sub438_iso/gpu0.log 2>&1 &
bash "$SCRIPT_DIR/rnaernie_ft.sh"      $((OFFSET+1)) > logs/sub438_iso/gpu1.log 2>&1 &
bash "$SCRIPT_DIR/rnamsm_ft.sh"        $((OFFSET+2)) > logs/sub438_iso/gpu2.log 2>&1 &
bash "$SCRIPT_DIR/rnafm_ft.sh"         $((OFFSET+3)) > logs/sub438_iso/gpu3.log 2>&1 &
bash "$SCRIPT_DIR/circmac_pairing.sh"  $((OFFSET+4)) > logs/sub438_iso/gpu4.log 2>&1 &

echo "모든 GPU 백그라운드 실행 시작"
echo "로그 확인: tail -f logs/sub438_iso/gpu{0..4}.log"

wait
echo "=============================="
echo " 전체 완료: $(date)"
echo "=============================="
