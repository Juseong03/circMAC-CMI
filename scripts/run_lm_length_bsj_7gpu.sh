#!/bin/bash
#===============================================================================
# BSJ-DISJOINT RNA-LM Length-Stratified — 7 GPU 병렬 실행
#
# GPU 배치:
#   GPU 0: sub438  RNAErnie ft    (max_len=438)
#   GPU 1: sub438  RNAMSM   ft    (max_len=438)
#   GPU 2: sub438  RNA-FM   ft    (max_len=438)
#   GPU 3: sub438  CircMAC  pair  (max_len=438)
#   GPU 4: sub511  RNAMSM   ft    (max_len=511)
#   GPU 5: sub511  RNA-FM   ft    (max_len=511)
#   GPU 6: sub511  CircMAC  pair  (max_len=511)
#
# 재사용 (새 실험 불필요):
#   RNABERT sub438  → 기존 bsj_rnabert_ft  (native max=438)
#   RNAErnie sub511 → 기존 bsj_rnaernie_ft (native max=511)
#   RNAMSM/RNA-FM/CircMAC max → 기존 bsj_rnamsm_ft / bsj_rnafm_ft / bsj_pt_pairing
#
# Usage: bash scripts/run_lm_length_bsj_7gpu.sh [GPU_OFFSET]
#   GPU_OFFSET: GPU 번호 오프셋 (default=0)
#===============================================================================
set -e
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
OFFSET=${1:-0}

mkdir -p logs/sub438_bsj logs/sub511_bsj

echo "======================================"
echo " BSJ Length-Stratified — 7 GPU 병렬"
echo " GPU offset: $OFFSET  (GPU $OFFSET ~ $((OFFSET+6)))"
echo " $(date)"
echo "======================================"

bash scripts/sub438_bsj/rnaernie_ft.sh     $((OFFSET+0)) > logs/sub438_bsj/gpu0_rnaernie.log    2>&1 &
bash scripts/sub438_bsj/rnamsm_ft.sh       $((OFFSET+1)) > logs/sub438_bsj/gpu1_rnamsm.log      2>&1 &
bash scripts/sub438_bsj/rnafm_ft.sh        $((OFFSET+2)) > logs/sub438_bsj/gpu2_rnafm.log       2>&1 &
bash scripts/sub438_bsj/circmac_pairing.sh $((OFFSET+3)) > logs/sub438_bsj/gpu3_circmac.log     2>&1 &
bash scripts/sub511_bsj/rnamsm_ft.sh       $((OFFSET+4)) > logs/sub511_bsj/gpu4_rnamsm.log      2>&1 &
bash scripts/sub511_bsj/rnafm_ft.sh        $((OFFSET+5)) > logs/sub511_bsj/gpu5_rnafm.log       2>&1 &
bash scripts/sub511_bsj/circmac_pairing.sh $((OFFSET+6)) > logs/sub511_bsj/gpu6_circmac.log     2>&1 &

echo "7개 GPU 백그라운드 실행 시작"
echo "로그 확인:"
echo "  tail -f logs/sub438_bsj/gpu{0..3}_*.log"
echo "  tail -f logs/sub511_bsj/gpu{4..6}_*.log"

wait
echo ""
echo "======================================"
echo " BSJ 완료: $(date)"
echo "======================================"
