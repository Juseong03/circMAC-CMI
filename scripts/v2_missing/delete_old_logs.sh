#!/bin/bash
#===============================================================================
# Delete old experiment logs from logs_0504 (or specified dir)
# Keeps: v2_* experiments and exp1_fair_* RNA LM comparisons
# Deletes: old exp1/2/3/4/5/6 experiments replaced by v2_* naming
#
# Usage: ./scripts/v2_missing/delete_old_logs.sh [logs_dir]
#        default logs_dir = logs_0504
#===============================================================================

LOGS_DIR="${1:-logs_0504}"

if [ ! -d "$LOGS_DIR" ]; then
    echo "ERROR: $LOGS_DIR not found"
    exit 1
fi

echo "========================================"
echo "  Deleting old logs from: $LOGS_DIR"
echo "  (keeping v2_* and exp1_fair_*)"
echo "========================================"

DRY_RUN="${DRY_RUN:-0}"
DELETED=0

delete_dir() {
    local d="$1"
    if [ -d "$d" ]; then
        if [ "$DRY_RUN" = "1" ]; then
            echo "  [DRY] rm -rf $d"
        else
            rm -rf "$d"
            echo "  [DEL] $d"
        fi
        DELETED=$((DELETED+1))
    fi
}

# ── Top-level old experiment dirs ─────────────────────────────────────────────
for OLD_DIR in exp1 exp2 exp2v2 exp2v3 exp2v4 exp3 exp4 exp5 exp6 default_model; do
    delete_dir "${LOGS_DIR}/${OLD_DIR}"
done

# ── Per-model old experiments ──────────────────────────────────────────────────
# circmac: keep v2_* and v2_ptm_*, delete everything else
for d in "${LOGS_DIR}/circmac"/exp1_* \
          "${LOGS_DIR}/circmac"/exp2_* \
          "${LOGS_DIR}/circmac"/exp2v2_* \
          "${LOGS_DIR}/circmac"/exp2v3_* \
          "${LOGS_DIR}/circmac"/exp2v4_* \
          "${LOGS_DIR}/circmac"/exp3_* \
          "${LOGS_DIR}/circmac"/exp4_* \
          "${LOGS_DIR}/circmac"/exp5_* \
          "${LOGS_DIR}/circmac"/exp6_*; do
    delete_dir "$d"
done

# mamba, hymba, lstm, transformer: keep only v2_enc_*
for MODEL in mamba hymba lstm transformer; do
    for d in "${LOGS_DIR}/${MODEL}"/exp1_* \
              "${LOGS_DIR}/${MODEL}"/exp3_*; do
        delete_dir "$d"
    done
done

# RNA LM models: keep only exp1_fair_*
for MODEL in rnabert rnaernie rnafm rnamsm; do
    for d in "${LOGS_DIR}/${MODEL}"/exp1_rna* \
              "${LOGS_DIR}/${MODEL}"/exp1_max_* \
              "${LOGS_DIR}/${MODEL}"/exp3_*; do
        delete_dir "$d"
    done
done

# ── GPU log file ───────────────────────────────────────────────────────────────
if [ -f "${LOGS_DIR}/gpu4.out" ]; then
    if [ "$DRY_RUN" = "1" ]; then
        echo "  [DRY] rm ${LOGS_DIR}/gpu4.out"
    else
        rm "${LOGS_DIR}/gpu4.out"
        echo "  [DEL] ${LOGS_DIR}/gpu4.out"
    fi
fi

echo ""
echo "  Deleted $DELETED directories"
echo ""
echo "Remaining:"
ls "$LOGS_DIR/"
echo ""
echo "To do a dry run first: DRY_RUN=1 ./scripts/v2_missing/delete_old_logs.sh $LOGS_DIR"
