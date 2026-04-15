#!/bin/bash
# =========================================================================
# Final Experiments — GPU 분배 가이드
#
# 총 75 runs  |  권장: 7 GPU  |  가능: 6~9 GPU (A100)
#
# ┌──────┬────────────────────────────────────────────────┬───────┬────────┐
# │ GPU  │ 담당 실험                                       │ runs  │ 예상   │
# ├──────┼────────────────────────────────────────────────┼───────┼────────┤
# │ GPU0 │ EXP1 base: circmac + mamba  (× 3 seeds)        │   6   │ ~18h  │
# │ GPU1 │ EXP1 base: lstm + hymba + transformer (× 3)    │   9   │ ~13h  │
# │ GPU2 │ EXP4 ablation Part-A: full/no_attn/no_mamba/no_conv (× 3) │ 12 │ ~30h │
# │ GPU3 │ EXP4 ablation Part-B: no_circ_bias/attn_only/mamba_only/cnn_only (× 3) │ 12 │ ~25h │
# │ GPU4 │ EXP5(9) + EXP6(6) + RNA frozen: rnabert/rnaernie (6) │ 21 │ ~20h │
# │ GPU5 │ EXP1 RNA frozen: rnafm + rnamsm (× 3)         │   6   │ ~17h  │
# │ GPU6 │ EXP1 RNA trainable: rnabert/rnaernie/rnamsm(×3)│   9   │ ~35h* │
# └──────┴────────────────────────────────────────────────┴───────┴────────┘
# * bottleneck: rnamsm trainable (bs=8, max_len=1022)
#
# 전체 wall-clock: ~35h (GPU6 bottleneck 기준)
#
# ── 실행 명령어 ──────────────────────────────────────────────────────────
# GPU 번호 자유롭게 지정 가능 (아래는 예시)
#
#   nohup ./scripts/final_v2/run_gpu0.sh          0 > /dev/null 2>&1 &
#   nohup ./scripts/final_v2/run_gpu1.sh          1 > /dev/null 2>&1 &
#   nohup ./scripts/final_v2/run_gpu2_exp4a.sh    2 > /dev/null 2>&1 &
#   nohup ./scripts/final_v2/run_gpu3_exp4b.sh    3 > /dev/null 2>&1 &
#   nohup ./scripts/final_v2/run_gpu4_exp56_rna.sh 4 > /dev/null 2>&1 &
#   nohup ./scripts/final_v2/run_gpu5_rna_frozen.sh 5 > /dev/null 2>&1 &
#   nohup ./scripts/final_v2/run_gpu6_rna_trainable.sh 6 > /dev/null 2>&1 &
#
# ── 6 GPU로 줄이려면 ──────────────────────────────────────────────────────
# GPU5에 rnafm/rnamsm frozen 추가:
#   run_gpu5_rna_frozen.sh + run_gpu6_rna_trainable.sh 를 한 GPU에서 순차 실행
#   → 총 ~52h (GPU5 bottleneck)
#
# ── 8~9 GPU로 늘리려면 ───────────────────────────────────────────────────
# rnamsm trainable (×3)를 별도 GPU7로 분리:
#   GPU6: rnabert + rnaernie trainable × 3  (~12h)
#   GPU7: rnamsm trainable × 3              (~24h)
# → 전체 wall-clock ~30h 이하
#
# ── 실험명 규칙 ──────────────────────────────────────────────────────────
#   exp1_{model}_s{seed}              EXP1 base encoders
#   exp1_{model}_frozen_s{seed}       EXP1 RNA LMs frozen
#   exp1_{model}_trainable_s{seed}    EXP1 RNA LMs trainable
#   exp4_full_s{seed}                 CircMAC 최종 (viz/분석 기준)
#   exp4_{variant}_s{seed}            EXP4 ablation variants
#   exp5_{interaction}_s{seed}        EXP5 interaction
#   exp6_{head}_s{seed}               EXP6 site head
# =========================================================================
echo "README only — do not run directly"
