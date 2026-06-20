#!/bin/bash
#===============================================================================
# run_all_single_gpu.sh — Run ALL experiments on a single GPU (sequential)
#
# Covers every experiment in eval_full.py:
#   EXP1 Base    : circmac, mamba, hymba, lstm, transformer        (v2_enc_*)
#   EXP2 PT Str  : 12 pretraining strategies                       (v2_pt_*)
#   EXP4 Ablation: 8 CircMAC ablation variants                     (v2_abl_*)
#   EXP5 Interact: cross_attn, concat, elementwise                 (v2_int_*)
#   EXP6 Head    : conv1d, linear                                  (v2_head_*)
#   RNA Frozen   : rnabert, rnaernie, rnamsm, rnafm                (exp1_fair_frozen_*)
#   RNA Trainable: rnabert, rnaernie, rnamsm, rnafm                (exp1_fair_trainable_*)
#
# Skip logic: model.pth exists → skip
# Seeds: 1-10 (all), can be overridden with SEEDS_OVERRIDE
#
# Usage:
#   bash scripts/final_v2/run_all_single_gpu.sh [GPU_ID]
#   SEEDS_OVERRIDE="1 2 3" bash scripts/final_v2/run_all_single_gpu.sh 0
#===============================================================================
set -e

GPU=${1:-0}
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [ -n "${SEEDS_OVERRIDE:-}" ]; then
    read -r -a SEEDS <<< "$SEEDS_OVERRIDE"
else
    SEEDS=(1 2 3 4 5 6 7 8 9 10)
fi

echo "========================================"
echo "  All Experiments — Single GPU Runner"
echo "  GPU=$GPU  Seeds: ${SEEDS[*]}"
echo "========================================"

SEEDS_STR="${SEEDS[*]}"

# ── Phase 1: EXP1 Base + EXP5 Interaction + EXP6 Site Head ──────────────────
echo ""
echo "[Phase 1] EXP1 Base + EXP5 Interaction + EXP6 Site Head"
SEEDS_OVERRIDE="$SEEDS_STR" bash scripts/final_v2/run_final_s1.sh "$GPU"

# ── Phase 2: EXP4 Ablation ───────────────────────────────────────────────────
echo ""
echo "[Phase 2] EXP4 CircMAC Ablation"
SEEDS_OVERRIDE="$SEEDS_STR" bash scripts/final_v2/run_final_s3.sh "$GPU"

# ── Phase 3: EXP2 Pretraining Strategy ───────────────────────────────────────
echo ""
echo "[Phase 3] EXP2 Pretraining Strategy (pretrain + finetune)"
SEEDS_OVERRIDE="$SEEDS_STR" bash scripts/final_v2/run_final_s2_10seed.sh "$GPU"

# ── Phase 4: EXP2 Missing Pretraining Strategies ─────────────────────────────
echo ""
echo "[Phase 4] EXP2 Missing Pretraining Strategies (ssp, cpcl, bsj, pairing, mlm_cpcl_ssp)"
SEEDS_OVERRIDE="$SEEDS_STR" bash scripts/final_v2/run_missing_pretraining_10seed.sh "$GPU"

# ── Phase 5: RNA-LM Frozen ───────────────────────────────────────────────────
echo ""
echo "[Phase 5] RNA-LM Frozen (rnabert, rnaernie, rnamsm, rnafm)"
SEEDS_OVERRIDE="$SEEDS_STR" bash scripts/final_v2/run_rna_frozen_10seed.sh "$GPU"

# ── Phase 6: RNA-LM Trainable ────────────────────────────────────────────────
echo ""
echo "[Phase 6] RNA-LM Trainable (rnabert, rnaernie, rnamsm)"
SEEDS_OVERRIDE="$SEEDS_STR" bash scripts/final_v2/run_rna_trainable_10seed.sh "$GPU"

# ── Phase 7: RNA-FM Trainable ────────────────────────────────────────────────
echo ""
echo "[Phase 7] RNA-FM Trainable (rnafm)"
SEEDS_OVERRIDE="$SEEDS_STR" bash scripts/final_v2/run_rnafm_trainable_10seed.sh "$GPU"

echo ""
echo "========================================"
echo "  All experiments complete!"
echo "  Run: python scripts/check_progress.py"
echo "========================================"
