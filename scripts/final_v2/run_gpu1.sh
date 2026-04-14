#!/bin/bash
#===============================================================================
# GPU1 담당: EXP1 RNA LM (Frozen + Trainable)
#
# 총 실험 수:
#   EXP1 Frozen:    4 models × 3 seeds = 12
#   EXP1 Trainable: 4 models × 3 seeds = 12
#   합계: 24 runs
#
# 주의:
#   - Mamba 미사용 → H100에서도 실행 가능
#   - rnafm/rnamsm trainable은 bs=8로 실행 (VRAM 제한)
#
# Usage: ./scripts/final_v2/run_gpu1.sh [GPU_ID]
#   e.g., ./scripts/final_v2/run_gpu1.sh 1
#===============================================================================
set -e
GPU=${1:-1}

echo "=============================="
echo " GPU${GPU}: EXP1 RNA LM Frozen + Trainable"
echo "=============================="

bash scripts/final_v2/run_exp1_rna_frozen.sh    $GPU
bash scripts/final_v2/run_exp1_rna_trainable.sh $GPU

echo "=============================="
echo " GPU${GPU} ALL DONE"
echo "=============================="
