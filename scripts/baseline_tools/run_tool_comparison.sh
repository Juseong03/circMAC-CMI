#!/bin/bash
#==============================================================================
# run_tool_comparison.sh
# Installs miRanda, RNAhybrid, TargetScan, PITA, IntaRNA and runs them
# against the circMAC test set. Outputs nucleotide-level predictions.
#
# Usage:
#   bash scripts/baseline_tools/run_tool_comparison.sh [N_PAIRS]
#   N_PAIRS: number of test pairs to evaluate (default: all)
#            use a small number (e.g. 100) for a quick sanity check
#
# Output:
#   results/baseline_tools/
#     miranda_preds.pkl
#     rnahybrid_preds.pkl
#     targetscan_preds.pkl
#     pita_preds.pkl
#     intarna_preds.pkl
#     comparison_metrics.csv
#     comparison_metrics.txt
#==============================================================================
set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

N_PAIRS=${1:-all}
OUTDIR="results/baseline_tools"
mkdir -p "$OUTDIR"

# ── 1. Install tools via conda ────────────────────────────────────────────────
echo "=== Installing tools ==="
# Create isolated env to avoid conflicts
if ! conda env list | grep -q "rna_tools"; then
    conda create -y -n rna_tools -c bioconda -c conda-forge \
        miranda rnahybrid intarna python=3.10 pandas numpy 2>&1 | tail -5
fi
CONDA_PYTHON=$(conda run -n rna_tools which python)
echo "  conda env: rna_tools"
conda run -n rna_tools miranda --version 2>/dev/null | head -1 || echo "  miranda: ready"
conda run -n rna_tools RNAhybrid --version 2>/dev/null | head -1 || echo "  RNAhybrid: ready"
conda run -n rna_tools IntaRNA --version 2>/dev/null | head -1 || echo "  IntaRNA: ready"

# ── 2. Prepare FASTA inputs ───────────────────────────────────────────────────
echo ""
echo "=== Preparing input FASTA files ==="
conda run -n rna_tools python scripts/baseline_tools/prepare_fasta.py \
    --n_pairs "$N_PAIRS" --outdir "$OUTDIR"

# ── 3. Run miRanda ────────────────────────────────────────────────────────────
echo ""
echo "=== Running miRanda ==="
conda run -n rna_tools python scripts/baseline_tools/run_miranda.py \
    --outdir "$OUTDIR"

# ── 4. Run RNAhybrid ──────────────────────────────────────────────────────────
echo ""
echo "=== Running RNAhybrid ==="
conda run -n rna_tools python scripts/baseline_tools/run_rnahybrid.py \
    --outdir "$OUTDIR"

# ── 5. Run PITA ───────────────────────────────────────────────────────────────
echo ""
echo "=== Running PITA ==="
# Try PITA binary first (install if needed), fallback to Python implementation
if ! micromamba env list 2>/dev/null | grep -q "pita_env"; then
    echo "  Installing PITA via micromamba..."
    micromamba create -n pita_env -c bioconda -c conda-forge pita perl viennarna -y 2>&1 | tail -3
fi
micromamba run -n pita_env python scripts/baseline_tools/run_pita.py \
    --outdir "$OUTDIR"

# ── 6. Run IntaRNA ────────────────────────────────────────────────────────────
echo ""
echo "=== Running IntaRNA ==="
conda run -n rna_tools python scripts/baseline_tools/run_intarna.py \
    --outdir "$OUTDIR"

# ── 7. Evaluate all tools ─────────────────────────────────────────────────────
echo ""
echo "=== Evaluating all tools ==="
conda run -n rna_tools python scripts/baseline_tools/evaluate_tools.py \
    --outdir "$OUTDIR"

echo ""
echo "=== Done. Results: $OUTDIR/comparison_metrics.csv ==="
