#!/bin/bash
#===============================================================================
# Server1 GPU0: exp2v4-A — No PT baseline + MLM + NTP
# Usage: ./scripts/final/run_s1_gpu0.sh [GPU_ID]
#===============================================================================
./scripts/final/exp2_v4_a.sh ${1:-0}
