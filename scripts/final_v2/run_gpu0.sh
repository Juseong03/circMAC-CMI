#!/bin/bash
#===============================================================================
# GPU0 담당: EXP1 Base + EXP4 Ablation + EXP5 Interaction + EXP6 Site Head
#
# 총 실험 수:
#   EXP1 Base:    5 models × 3 seeds = 15
#   EXP4 Ablation: 8 configs × 3 seeds = 24
#   EXP5:          3 configs × 3 seeds = 9
#   EXP6:          2 configs × 3 seeds = 6
#   합계: 54 runs
#
# Usage: ./scripts/final_v2/run_gpu0.sh [GPU_ID]
#   e.g., ./scripts/final_v2/run_gpu0.sh 0
#===============================================================================
set -e
GPU=${1:-0}

echo "=============================="
echo " GPU${GPU}: EXP1-Base / EXP4 / EXP5 / EXP6"
echo "=============================="

bash scripts/final_v2/run_exp1_base.sh        $GPU
bash scripts/final_v2/run_exp4_ablation.sh    $GPU
bash scripts/final_v2/run_exp5_interaction.sh $GPU
bash scripts/final_v2/run_exp6_site_head.sh   $GPU

echo "=============================="
echo " GPU${GPU} ALL DONE"
echo "=============================="
