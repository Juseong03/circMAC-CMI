#!/bin/bash
# run_pita_all_splits.sh
# Runs PITA on pair / iso / bsj splits.
# Uses Python fallback (seed-match + ddG proxy) if PITA binary not installed.
#
# Usage:
#   bash scripts/baseline_tools/run_pita_all_splits.sh
#
# Prerequisites:
#   prepare_fasta.py must have been run for all splits.

set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

OUTDIR="results/baseline_tools"

# Try to install PITA + ViennaRNA if not already present
if ! micromamba env list 2>/dev/null | grep -q "pita_env"; then
    echo "=== Installing PITA env ==="
    micromamba create -n pita_env -c bioconda -c conda-forge pita perl viennarna python -y 2>&1 | tail -5
fi

for SPLIT in pair iso bsj; do
    OUT="${OUTDIR}/${SPLIT}/pita_preds.pkl"
    if [ -f "$OUT" ]; then
        echo "=== [${SPLIT}] SKIP (already exists: $OUT) ==="
        continue
    fi
    echo ""
    echo "=== Running PITA on ${SPLIT} split ==="
    micromamba run -n pita_env python scripts/baseline_tools/run_pita.py \
        --outdir "$OUTDIR" --split "$SPLIT"
done

echo ""
echo "=== All splits done. Running evaluate_tools + make_comparison_table ==="

for SPLIT in pair iso bsj; do
    echo ""
    echo "--- Evaluating ${SPLIT} ---"
    python scripts/baseline_tools/evaluate_tools.py \
        --outdir "${OUTDIR}/${SPLIT}"
done

echo ""
echo "--- Making full comparison table ---"
python scripts/baseline_tools/make_comparison_table.py

echo ""
echo "=== Done. ==="
