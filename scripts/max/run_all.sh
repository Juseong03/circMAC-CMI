#!/bin/bash
#===============================================================================
# MAX — 6 GPU 병렬 실행 (max_len=1022, fixed circmac.py)
#
#   GPU 0: RNAMSM frozen
#   GPU 1: RNAMSM fine-tuned
#   GPU 2: RNA-FM frozen
#   GPU 3: RNA-FM fine-tuned
#   GPU 4: CircMAC noPT
#   GPU 5: CircMAC pairing
#
# Usage: ./scripts/max/run_all.sh
#===============================================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

mkdir -p logs/max

echo "=== MAX — 6 GPU 병렬 실행 $(date) ==="

bash "$SCRIPT_DIR/gpu0_rnamsm_frozen.sh"    0 > logs/max/gpu0.log 2>&1 &
bash "$SCRIPT_DIR/gpu1_rnamsm_ft.sh"        1 > logs/max/gpu1.log 2>&1 &
bash "$SCRIPT_DIR/gpu2_rnafm_frozen.sh"     2 > logs/max/gpu2.log 2>&1 &
bash "$SCRIPT_DIR/gpu3_rnafm_ft.sh"         3 > logs/max/gpu3.log 2>&1 &
bash "$SCRIPT_DIR/gpu4_circmac_nopt.sh"     4 > logs/max/gpu4.log 2>&1 &
bash "$SCRIPT_DIR/gpu5_circmac_pairing.sh"  5 > logs/max/gpu5.log 2>&1 &

echo "로그: tail -f logs/max/gpu{0..5}.log"
wait
echo "=== 전체 완료: $(date) ==="
