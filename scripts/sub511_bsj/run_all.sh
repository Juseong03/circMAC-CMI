#!/bin/bash
#===============================================================================
# SUB511 BSJ-DISJOINT — 4 GPU 병렬 실행
#
# GPU 배치:
#   GPU 0: RNAErnie ft  (max_len=511)
#   GPU 1: RNAMSM   ft  (max_len=511)
#   GPU 2: RNA-FM   ft  (max_len=511)
#   GPU 3: CircMAC  pairing (max_len=511)
#
# Usage: bash scripts/sub511_bsj/run_all.sh
#   또는 GPU 오프셋 지정: GPU_OFFSET=4 bash scripts/sub511_bsj/run_all.sh
#===============================================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

OFFSET=${GPU_OFFSET:-0}
mkdir -p logs/sub511_bsj

echo "=============================="
echo " SUB511-BSJ — 4 GPU 병렬 실행"
echo " GPU offset: $OFFSET"
echo " $(date)"
echo "=============================="

bash "$SCRIPT_DIR/rnaernie_ft.sh"     $((OFFSET+0)) > logs/sub511_bsj/gpu0.log 2>&1 &
bash "$SCRIPT_DIR/rnamsm_ft.sh"       $((OFFSET+1)) > logs/sub511_bsj/gpu1.log 2>&1 &
bash "$SCRIPT_DIR/rnafm_ft.sh"        $((OFFSET+2)) > logs/sub511_bsj/gpu2.log 2>&1 &
bash "$SCRIPT_DIR/circmac_pairing.sh" $((OFFSET+3)) > logs/sub511_bsj/gpu3.log 2>&1 &

echo "모든 GPU 백그라운드 실행 시작"
echo "로그 확인: tail -f logs/sub511_bsj/gpu{0..3}.log"

wait
echo "=============================="
echo " 전체 완료: $(date)"
echo "=============================="
