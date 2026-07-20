#!/bin/bash
#===============================================================================
# ISO-DISJOINT RNA-LM Length-Stratified — 7 GPU 병렬 실행
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
#   RNABERT sub438  → 기존 iso_rnabert_ft  (native max=438)
#   RNAErnie sub511 → 기존 iso_rnaernie_ft (native max=511)
#   RNAMSM/RNA-FM/CircMAC max → 기존 iso_rnamsm_ft / iso_rnafm_ft / iso_pt_pairing
#
# Usage: bash scripts/run_lm_length_iso_7gpu.sh [GPU_OFFSET]
#   GPU_OFFSET: GPU 번호 오프셋 (default=0)
#===============================================================================
set -e
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
OFFSET=${1:-0}

mkdir -p logs/sub438_iso logs/sub511_iso

echo "======================================"
echo " ISO Length-Stratified — 7 GPU 병렬"
echo " GPU offset: $OFFSET  (GPU $OFFSET ~ $((OFFSET+6)))"
echo " $(date)"
echo "======================================"

bash scripts/sub438_iso/rnaernie_ft.sh     $((OFFSET+0)) > logs/sub438_iso/gpu0_rnaernie.log    2>&1 &
bash scripts/sub438_iso/rnamsm_ft.sh       $((OFFSET+1)) > logs/sub438_iso/gpu1_rnamsm.log      2>&1 &
bash scripts/sub438_iso/rnafm_ft.sh        $((OFFSET+2)) > logs/sub438_iso/gpu2_rnafm.log       2>&1 &
bash scripts/sub438_iso/circmac_pairing.sh $((OFFSET+3)) > logs/sub438_iso/gpu3_circmac.log     2>&1 &
bash scripts/sub511_iso/rnamsm_ft.sh       $((OFFSET+4)) > logs/sub511_iso/gpu4_rnamsm.log      2>&1 &
bash scripts/sub511_iso/rnafm_ft.sh        $((OFFSET+5)) > logs/sub511_iso/gpu5_rnafm.log       2>&1 &
bash scripts/sub511_iso/circmac_pairing.sh $((OFFSET+6)) > logs/sub511_iso/gpu6_circmac.log     2>&1 &

echo "7개 GPU 백그라운드 실행 시작"
echo "로그 확인:"
echo "  tail -f logs/sub438_iso/gpu{0..3}_*.log"
echo "  tail -f logs/sub511_iso/gpu{4..6}_*.log"

wait
echo ""
echo "======================================"
echo " ISO 완료: $(date)"
echo "======================================"
