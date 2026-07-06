#!/bin/bash
#===============================================================================
# Final Re-run — 10-seed launcher for 6 GPUs
#
# Layout:
#   GPU0: run_final_s1.sh        (seeds 1-10)
#   GPU1: run_final_s2_10seed.sh (PRETRAIN_ONLY=1)
#   GPU2: run_final_s2_10seed.sh (SKIP_PRETRAIN=1, seeds 1-5)
#   GPU3: run_final_s2_10seed.sh (SKIP_PRETRAIN=1, seeds 6-10)
#   GPU4: run_final_s3.sh        (seeds 1-5)
#   GPU5: run_final_s3.sh        (seeds 6-10)
#
# Usage:
#   ./scripts/final_v2/run_final_10seed_6gpu.sh
#===============================================================================
set -e

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo "========================================"
echo "  Launching 10-seed rerun on 6 GPUs"
echo "========================================"

SEEDS_OVERRIDE="1 2 3 4 5 6 7 8 9 10" bash scripts/final_v2/run_final_s1.sh 0 &
PID0=$!

SEEDS_OVERRIDE="1 2 3 4 5" bash scripts/final_v2/run_final_s3.sh 4 &
PID4=$!
SEEDS_OVERRIDE="6 7 8 9 10" bash scripts/final_v2/run_final_s3.sh 5 &
PID5=$!

PRETRAIN_ONLY=1 bash scripts/final_v2/run_final_s2_10seed.sh 1

SKIP_PRETRAIN=1 SEEDS_OVERRIDE="1 2 3 4 5" bash scripts/final_v2/run_final_s2_10seed.sh 2 &
PID2=$!
SKIP_PRETRAIN=1 SEEDS_OVERRIDE="6 7 8 9 10" bash scripts/final_v2/run_final_s2_10seed.sh 3 &
PID3=$!

wait $PID0 $PID2 $PID3 $PID4 $PID5

echo ""
echo "All workers finished."
