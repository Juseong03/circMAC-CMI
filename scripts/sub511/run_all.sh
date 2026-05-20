#!/bin/bash
#===============================================================================
# SUB511 — RNAMSM / RNA-FM / CircMAC  (max_len=511, seeds 1 2 3)
#
# RNAErnie는 native max=511이므로 기존 exp1 결과 재사용
# RNAMSM/RNA-FM max 결과도 기존 exp1(max_len=1022) 재사용
#
# GPU 배치 (15 runs):
#   GPU 0: RNAMSM frozen    s1,2,3
#   GPU 1: RNAMSM ft bs32   s1,2,3
#   GPU 2: RNA-FM frozen    s1,2,3
#   GPU 3: RNA-FM ft bs32   s1,2,3
#   GPU 4: CircMAC          s1,2,3
#
# Usage: ./scripts/sub511/run_all.sh
#===============================================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

mkdir -p logs/sub511

echo "=== SUB511 — 5 GPU 병렬 실행 $(date) ==="

bash "$SCRIPT_DIR/gpu0_rnamsm_frozen.sh"  0 > logs/sub511/gpu0.log 2>&1 &
bash "$SCRIPT_DIR/gpu1_rnamsm_ft.sh"      1 > logs/sub511/gpu1.log 2>&1 &
bash "$SCRIPT_DIR/gpu2_rnafm_frozen.sh"   2 > logs/sub511/gpu2.log 2>&1 &
bash "$SCRIPT_DIR/gpu3_rnafm_ft.sh"       3 > logs/sub511/gpu3.log 2>&1 &
bash "$SCRIPT_DIR/gpu4_circmac.sh"        4 > logs/sub511/gpu4.log 2>&1 &

echo "로그: tail -f logs/sub511/gpu{0..4}.log"
wait
echo "=== 전체 완료: $(date) ==="
