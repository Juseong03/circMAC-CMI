#!/bin/bash
# ================================================================
# BSJ Analysis — 서버에서 실행 후 결과 CSV 저장
#
# 사용법:
#   cd ~/cmi_mac   (프로젝트 루트)
#   bash docs/paper_cmi/run_bsj_server.sh [device] [out_dir]
#
# 예시:
#   bash docs/paper_cmi/run_bsj_server.sh 0 docs/paper_cmi/results_server1
#
# 완료 후 CSV 파일을 이 서버로 복사:
#   scp [server]:/path/to/cmi_mac/docs/paper_cmi/results_server*/bsj_*.csv \
#       /workspace/volume/cmi_mac/docs/paper_cmi/
# ================================================================

DEVICE=${1:-0}
OUT_DIR=${2:-docs/paper_cmi/results_server}
DATA_PATH=${3:-/path/to/df_test_final.pkl}   # ← 서버 경로로 수정

mkdir -p "$OUT_DIR"

echo "============================================================"
echo " BSJ Analysis  |  device=$DEVICE  |  out=$OUT_DIR"
echo "============================================================"

# ── 실행 ──────────────────────────────────────────────────────
python docs/paper_cmi/analyze_bsj.py \
  --models circmac mamba hymba lstm transformer \
           rnabert rnaernie rnafm rnamsm \
  --exps   exp1_circmac_s3  exp1_mamba_s3   exp1_hymba_s3 \
           exp1_lstm_s3     exp1_transformer_s3 \
           exp3_rnabert_frozen_s3  exp3_rnaernie_frozen_s3 \
           exp3_rnafm_frozen_s3   exp3_rnamsm_frozen_s3 \
  --seeds  3 3 3 3 3 3 3 3 3 \
  --device "$DEVICE" \
  --data_path "$DATA_PATH" \
  --out_dir "$OUT_DIR"

echo ""
echo "=== Done! Results saved to: $OUT_DIR ==="
echo "Files:"
ls -lh "$OUT_DIR"/*.csv "$OUT_DIR"/*.pdf 2>/dev/null
