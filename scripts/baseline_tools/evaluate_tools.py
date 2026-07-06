#!/usr/bin/env python3
"""
evaluate_tools.py
Evaluates miRanda, RNAhybrid, IntaRNA and circMAC predictions on the same
test pairs, computing:
  - Binding classification: AUROC, AUPRC, F1, Precision, Recall
  - Site-level (nucleotide): AUROC, AUPRC, F1 @ threshold=0.5

Output:
  {outdir}/comparison_metrics.csv
  {outdir}/comparison_metrics.txt   (human-readable)
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
)


def load_preds(path: Path) -> list[dict]:
    with open(path, "rb") as f:
        return pickle.load(f)


def eval_binding(results: list[dict]) -> dict:
    """Binding-level classification metrics."""
    y_true  = np.array([r["binding"]    for r in results])
    y_score = np.array([r["bind_score"] for r in results])
    y_pred  = np.array([r["pred_bind"]  for r in results])

    # Normalise scores to [0,1] for AUROC/AUPRC
    if y_score.max() > y_score.min():
        y_score_norm = (y_score - y_score.min()) / (y_score.max() - y_score.min())
    else:
        y_score_norm = y_score

    try:
        auroc = roc_auc_score(y_true, y_score_norm)
    except Exception:
        auroc = float("nan")
    try:
        auprc = average_precision_score(y_true, y_score_norm)
    except Exception:
        auprc = float("nan")

    f1   = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)

    return dict(bind_auroc=auroc, bind_auprc=auprc,
                bind_f1=f1, bind_prec=prec, bind_rec=rec)


def eval_sites(results: list[dict], thr: float = 0.5) -> dict:
    """Nucleotide-level site prediction metrics (positive pairs only)."""
    all_true  = []
    all_score = []
    all_pred  = []

    for r in results:
        if r["binding"] != 1:
            continue
        sites = r.get("sites")
        pred  = r.get("pred_sites")
        if sites is None or pred is None:
            continue
        sites = np.array(sites, dtype=int)
        pred  = np.array(pred,  dtype=float)
        if len(sites) != len(pred):
            continue
        all_true.extend(sites.tolist())
        all_score.extend(pred.tolist())
        all_pred.extend((pred >= thr).astype(int).tolist())

    if not all_true:
        return dict(site_auroc=float("nan"), site_auprc=float("nan"),
                    site_f1=float("nan"), site_prec=float("nan"),
                    site_rec=float("nan"))

    y_true  = np.array(all_true)
    y_score = np.array(all_score)
    y_pred  = np.array(all_pred)

    try:
        auroc = roc_auc_score(y_true, y_score)
    except Exception:
        auroc = float("nan")
    try:
        auprc = average_precision_score(y_true, y_score)
    except Exception:
        auprc = float("nan")

    f1   = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)

    return dict(site_auroc=auroc, site_auprc=auprc,
                site_f1=f1, site_prec=prec, site_rec=rec)


def load_circmac_preds(outdir: Path, pairs_index: list[dict]) -> list[dict] | None:
    """
    Try to load circMAC predictions from eval_results/preds/.
    Returns list[dict] in same format as tool predictions, or None if not found.
    """
    preds_dir = ROOT / "eval_results" / "preds"
    if not preds_dir.exists():
        return None

    # Find the best CircMAC experiment (v2_abl_full) predictions
    circ_dirs = sorted(preds_dir.glob("v2_abl_full_s*"))
    if not circ_dirs:
        return None

    # Use first available seed
    circ_dir = circ_dirs[0]
    test_pred_path = circ_dir / "test_preds.pkl"
    if not test_pred_path.exists():
        return None

    print(f"  Loading circMAC preds: {circ_dir.name}")
    with open(test_pred_path, "rb") as f:
        df_preds = pickle.load(f)

    # df_preds is expected to have: sample_idx, position, label, prob
    if not isinstance(df_preds, pd.DataFrame):
        return None

    # Build pair-keyed lookup
    pair_lookup = {}
    for item in pairs_index:
        key = (item["isoform_ID"], item["miRNA_ID"])
        pair_lookup[key] = item

    results = []
    for (iso_id, mir_id), grp in df_preds.groupby(["sample_idx", "position"]):
        # This might not match exactly — skip for now
        pass

    # Fallback: return None (circMAC eval needs specific mapping)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="results/baseline_tools")
    parser.add_argument("--thr",    type=float, default=0.5,
                        help="Threshold for site-level binary prediction")
    args = parser.parse_args()

    outdir = Path(args.outdir)

    with open(outdir / "pairs_index.pkl", "rb") as f:
        pairs_index = pickle.load(f)

    # ── Load predictions ──────────────────────────────────────────────────────
    tools = {}
    for name, fname in [
        ("miRanda",   "miranda_preds.pkl"),
        ("RNAhybrid", "rnahybrid_preds.pkl"),
        ("IntaRNA",   "intarna_preds.pkl"),
    ]:
        p = outdir / fname
        if p.exists():
            tools[name] = load_preds(p)
            print(f"  Loaded {name}: {len(tools[name]):,} pairs")
        else:
            print(f"  [SKIP] {name}: {fname} not found")

    if not tools:
        print("No prediction files found. Run the tool scripts first.")
        return

    # ── Evaluate ──────────────────────────────────────────────────────────────
    rows = []
    for name, preds in tools.items():
        b_metrics = eval_binding(preds)
        s_metrics = eval_sites(preds, thr=args.thr)
        n_pos_pred = sum(r["pred_bind"] for r in preds)
        n_total    = len(preds)
        rows.append({
            "Model":          name,
            "N_pairs":        n_total,
            "N_pos_predicted": n_pos_pred,
            "Bind_AUROC":     round(b_metrics["bind_auroc"],  4),
            "Bind_AUPRC":     round(b_metrics["bind_auprc"],  4),
            "Bind_F1":        round(b_metrics["bind_f1"],     4),
            "Bind_Prec":      round(b_metrics["bind_prec"],   4),
            "Bind_Rec":       round(b_metrics["bind_rec"],    4),
            "Site_AUROC":     round(s_metrics["site_auroc"],  4),
            "Site_AUPRC":     round(s_metrics["site_auprc"],  4),
            "Site_F1":        round(s_metrics["site_f1"],     4),
            "Site_Prec":      round(s_metrics["site_prec"],   4),
            "Site_Rec":       round(s_metrics["site_rec"],    4),
        })

    df = pd.DataFrame(rows)
    csv_path = outdir / "comparison_metrics.csv"
    df.to_csv(csv_path, index=False)

    # ── Pretty print ──────────────────────────────────────────────────────────
    txt_lines = []
    def tee(s):
        print(s); txt_lines.append(s)

    tee("\n" + "="*70)
    tee("  Tool Comparison Results")
    tee("="*70)
    tee(f"\n  Pairs evaluated: {len(pairs_index):,}")
    tee(f"  Site threshold : {args.thr}")

    tee(f"\n  {'Model':<12}  {'Bind_AUROC':>10}  {'Bind_AUPRC':>10}  "
        f"{'Bind_F1':>8}  {'Site_AUROC':>10}  {'Site_AUPRC':>10}  {'Site_F1':>8}")
    tee(f"  {'-'*76}")
    for _, r in df.iterrows():
        tee(f"  {r['Model']:<12}  {r['Bind_AUROC']:>10.4f}  {r['Bind_AUPRC']:>10.4f}  "
            f"{r['Bind_F1']:>8.4f}  {r['Site_AUROC']:>10.4f}  {r['Site_AUPRC']:>10.4f}  "
            f"{r['Site_F1']:>8.4f}")

    tee(f"\n  NOTE: circMAC results should be added from eval_results/ for")
    tee(f"        fair comparison on the same pairs.")
    tee(f"\n  Saved: {csv_path}")

    (outdir / "comparison_metrics.txt").write_text("\n".join(txt_lines))


if __name__ == "__main__":
    main()
