#!/usr/bin/env python3
"""
make_comparison_table.py
Combines baseline tool results (miRanda, RNAhybrid) with CircMAC results
across pair / iso / bsj splits.

Output:
  results/baseline_tools/full_comparison.csv
  results/baseline_tools/full_comparison.txt
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "results" / "baseline_tools"
EVAL   = ROOT / "eval_results"


def load_tool_results(split):
    """Load miRanda/RNAhybrid CSV for a split."""
    p = OUTDIR / split / "comparison_metrics.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["split"] = split
    return df


def load_circmac_pair():
    """Load CircMAC results on pair-disjoint split from eval_full."""
    rows = []
    for csv_name in ["eval_full_ablation.csv", "eval_full_summary.csv"]:
        p = EVAL / csv_name
        if not p.exists():
            continue
        df = pd.read_csv(p)
        circ = df[(df["label"].str.contains("CircMAC.*full", na=False)) &
                  (df["split"] == "test")]
        if len(circ) == 0:
            circ = df[(df["exp_tpl"] == "v2_abl_full") & (df["split"] == "test")]
        if len(circ) > 0:
            rows.append({
                "Model":        "CircMAC",
                "split":        "pair",
                "N_pairs":      circ["n_tokens"].mean() if "n_tokens" in circ else None,
                "Site_AUROC":   round(circ["auroc"].mean(), 4),
                "Site_AUPRC":   round(circ["auprc"].mean(), 4),
                "Site_F1":      round(circ["f1_pos"].mean(), 4),
                "Site_AUROC_std": round(circ["auroc"].std(), 4),
                "Site_AUPRC_std": round(circ["auprc"].std(), 4),
                "Site_F1_std":    round(circ["f1_pos"].std(), 4),
                "Bind_AUROC":   None,
                "Bind_AUPRC":   None,
                "Bind_F1":      None,
                "n_seeds":      len(circ),
            })
            break
    return pd.DataFrame(rows)


def load_circmac_disjoint():
    """Load CircMAC results on iso/bsj splits from disjoint_new_summary.csv."""
    p = EVAL / "disjoint_new_summary.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    circ = df[df["label"] == "CircMAC"]
    rows = []
    for _, r in circ.iterrows():
        rows.append({
            "Model":        "CircMAC",
            "split":        r["split"],
            "Site_AUROC":   r.get("auroc_mean"),
            "Site_AUPRC":   r.get("auprc_mean"),
            "Site_F1":      r.get("f1_pos_mean"),
            "Site_AUROC_std": r.get("auroc_std"),
            "Site_AUPRC_std": r.get("auprc_std"),
            "Site_F1_std":    r.get("f1_pos_std"),
            "Bind_AUROC":   None,
            "Bind_AUPRC":   None,
            "Bind_F1":      None,
            "n_seeds":      r.get("n_seeds"),
        })
    return pd.DataFrame(rows)


def main():
    # ── Load baseline tool results ─────────────────────────────────────────
    tool_rows = []
    for split in ["pair", "iso", "bsj"]:
        df = load_tool_results(split)
        if df.empty:
            continue
        for _, r in df.iterrows():
            tool_rows.append({
                "Model":        r["Model"],
                "split":        split,
                "N_pairs":      r.get("N_pairs"),
                "Bind_AUROC":   r.get("Bind_AUROC"),
                "Bind_AUPRC":   r.get("Bind_AUPRC"),
                "Bind_F1":      r.get("Bind_F1"),
                "Site_AUROC":   r.get("Site_AUROC"),
                "Site_AUPRC":   r.get("Site_AUPRC"),
                "Site_F1":      r.get("Site_F1"),
                "Site_AUROC_std": None,
                "Site_AUPRC_std": None,
                "Site_F1_std":    None,
                "n_seeds":      1,
            })
    df_tools = pd.DataFrame(tool_rows)

    # ── Load CircMAC results ───────────────────────────────────────────────
    df_circ_pair = load_circmac_pair()
    df_circ_disj = load_circmac_disjoint()
    df_circ = pd.concat([df_circ_pair, df_circ_disj], ignore_index=True)

    df_all = pd.concat([df_tools, df_circ], ignore_index=True)

    out_csv = OUTDIR / "full_comparison.csv"
    df_all.to_csv(out_csv, index=False)

    # ── Print table ────────────────────────────────────────────────────────
    lines = []
    def tee(s): print(s); lines.append(s)

    tee("\n" + "="*80)
    tee("  Baseline Tool vs CircMAC Comparison")
    tee("  Metrics: Site-level (nucleotide) on positive pairs")
    tee("="*80)

    def fmt(val, std=None):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "  —   "
        if std is not None and not np.isnan(std):
            return f"{val:.4f}±{std:.4f}"
        return f"{val:.4f}"

    MODEL_ORDER = ["miRanda", "RNAhybrid", "TargetScan", "PITA", "CircMAC"]
    SPLIT_ORDER = ["pair", "iso", "bsj"]
    SPLIT_LABEL = {"pair": "Pair-disjoint", "iso": "Iso-disjoint", "bsj": "BSJ-disjoint"}

    for split in SPLIT_ORDER:
        tee(f"\n  [{SPLIT_LABEL[split]}]")
        tee(f"  {'Model':<14}  {'Bind_AUROC':>12}  {'Site_AUROC':>16}  {'Site_AUPRC':>16}  {'Site_F1':>16}")
        tee("  " + "-"*78)
        sub = df_all[df_all["split"] == split]
        for model in MODEL_ORDER:
            r = sub[sub["Model"] == model]
            if r.empty:
                tee(f"  {model:<14}  {'(not available)':>62}")
                continue
            r = r.iloc[0]
            tee(f"  {model:<14}  "
                f"{fmt(r.get('Bind_AUROC')):>12}  "
                f"{fmt(r.get('Site_AUROC'), r.get('Site_AUROC_std')):>16}  "
                f"{fmt(r.get('Site_AUPRC'), r.get('Site_AUPRC_std')):>16}  "
                f"{fmt(r.get('Site_F1'),    r.get('Site_F1_std')):>16}")

    tee(f"\n  Note: Bind_AUROC for iso/bsj splits is affected by class imbalance")
    tee(f"        (fewer negatives: ~18% vs 56% in pair-disjoint).")
    tee(f"        Site-level metrics are evaluated on positive pairs only.")
    tee(f"\n  Saved: {out_csv}")

    (OUTDIR / "full_comparison.txt").write_text("\n".join(lines))
    print(f"\n  Saved txt: {OUTDIR / 'full_comparison.txt'}")


if __name__ == "__main__":
    main()
