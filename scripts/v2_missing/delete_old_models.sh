#!/bin/bash
#===============================================================================
# Delete old saved_models (mirrors delete_old_logs.sh)
# Keeps: v2_* experiments and exp1_fair_* RNA LM models
# Deletes: old exp1/2/3/4/5/6 models replaced by v2_* naming
#
# Usage: ./scripts/v2_missing/delete_old_models.sh [models_dir]
#        default models_dir = saved_models
#===============================================================================

MODELS_DIR="${1:-saved_models}"

if [ ! -d "$MODELS_DIR" ]; then
    echo "ERROR: $MODELS_DIR not found"
    exit 1
fi

echo "========================================"
echo "  Deleting old saved_models from: $MODELS_DIR"
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

# ── circmac: keep v2_* and v2_ptm_*, delete everything else ───────────────────
for d in "${MODELS_DIR}/circmac"/exp1_* \
          "${MODELS_DIR}/circmac"/exp2_* \
          "${MODELS_DIR}/circmac"/exp2v2_* \
          "${MODELS_DIR}/circmac"/exp2v3_* \
          "${MODELS_DIR}/circmac"/exp2v4_* \
          "${MODELS_DIR}/circmac"/exp3_* \
          "${MODELS_DIR}/circmac"/exp4_* \
          "${MODELS_DIR}/circmac"/exp5_* \
          "${MODELS_DIR}/circmac"/exp6_*; do
    delete_dir "$d"
done

# ── mamba, hymba, lstm, transformer: keep only v2_enc_* ───────────────────────
for MODEL in mamba hymba lstm transformer; do
    for d in "${MODELS_DIR}/${MODEL}"/exp1_* \
              "${MODELS_DIR}/${MODEL}"/exp3_*; do
        delete_dir "$d"
    done
done

# ── RNA LM models: keep only exp1_fair_* ──────────────────────────────────────
for MODEL in rnabert rnaernie rnafm rnamsm; do
    for d in "${MODELS_DIR}/${MODEL}"/exp1_rna* \
              "${MODELS_DIR}/${MODEL}"/exp1_max_* \
              "${MODELS_DIR}/${MODEL}"/exp3_*; do
        delete_dir "$d"
    done
done

# ── default_model ──────────────────────────────────────────────────────────────
delete_dir "${MODELS_DIR}/default_model"

echo ""
echo "  Deleted $DELETED directories"
echo ""
echo "Remaining top-level:"
ls "$MODELS_DIR/" 2>/dev/null
echo ""
echo "To do a dry run first: DRY_RUN=1 ./scripts/v2_missing/delete_old_models.sh $MODELS_DIR"
