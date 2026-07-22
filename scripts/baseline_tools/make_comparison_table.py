#!/usr/bin/env python3
"""
make_comparison_table.py
Combines baseline tool results (miRanda, RNAhybrid, TargetScan, PITA) with
CircMAC results across pair / iso / bsj splits.

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
    """Load baseline tool CSV for a split."""
    p = OUTDIR / split / "comparison_metrics.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["split"] = split
    return df


def load_circmac_pair():
    """Load CircMAC (NoPT) results on pair split.
    Priority: logs-based fig3_encoder_data.csv → eval_full CSVs.
    """
    # 1) logs 기반 (logs2csv.py 결과)
    logs_csv = ROOT / "figures_paper/fig3_encoder/fig3_encoder_data.csv"
    if logs_csv.exists():
        df = pd.read_csv(logs_csv)
        circ = df[df["model"] == "CircMAC"]
        if len(circ) > 0:
            return pd.DataFrame([{
                "Model":          "CircMAC (NoPT)",
                "split":          "pair",
                "N_pairs":        None,
                "Bind_AUROC":     None, "Bind_AUPRC": None, "Bind_F1": None,
                "Site_AUROC":     round(circ["roc_auc"].mean(), 4),
                "Site_AUPRC":     round(circ["auprc"].mean(), 4),
                "Site_F1":        round(circ["f1_pos"].mean(), 4) if "f1_pos" in circ else None,
                "Site_Prec":      round(circ["prec_pos"].mean(), 4) if "prec_pos" in circ else None,
                "Site_Rec":       round(circ["rec_pos"].mean(), 4)  if "rec_pos"  in circ else None,
                "Site_AUROC_std": round(circ["roc_auc"].std(), 4),
                "Site_AUPRC_std": round(circ["auprc"].std(), 4),
                "Site_F1_std":    round(circ["f1_pos"].std(), 4) if "f1_pos" in circ else None,
                "Site_Prec_std":  round(circ["prec_pos"].std(), 4) if "prec_pos" in circ else None,
                "Site_Rec_std":   round(circ["rec_pos"].std(), 4)  if "rec_pos"  in circ else None,
                "n_seeds":        len(circ),
                "source":         "logs",
            }])

    # 2) eval_full CSV fallback
    for csv_name in ["eval_full_pretrained.csv", "eval_full_ablation.csv", "eval_full_summary.csv"]:
        p = EVAL / csv_name
        if not p.exists():
            continue
        df = pd.read_csv(p)
        circ = df[(df["exp_tpl"] == "v2_abl_full") & (df["split"] == "test")]
        if len(circ) == 0:
            circ = df[(df["label"].str.contains("CircMAC.*NoPT|CircMAC.*full", na=False)) &
                      (df["split"] == "test")]
        if len(circ) > 0:
            return pd.DataFrame([{
                "Model":          "CircMAC (NoPT)",
                "split":          "pair",
                "N_pairs":        None,
                "Bind_AUROC":     None, "Bind_AUPRC": None, "Bind_F1": None,
                "Site_AUROC":     round(circ["auroc"].mean(), 4),
                "Site_AUPRC":     round(circ["auprc"].mean(), 4),
                "Site_F1":        round(circ["f1_pos"].mean(), 4),
                "Site_Prec":      round(circ["prec_pos"].mean(), 4) if "prec_pos" in circ else None,
                "Site_Rec":       round(circ["rec_pos"].mean(), 4)  if "rec_pos"  in circ else None,
                "Site_AUROC_std": round(circ["auroc"].std(), 4),
                "Site_AUPRC_std": round(circ["auprc"].std(), 4),
                "Site_F1_std":    round(circ["f1_pos"].std(), 4),
                "Site_Prec_std":  round(circ["prec_pos"].std(), 4) if "prec_pos" in circ else None,
                "Site_Rec_std":   round(circ["rec_pos"].std(), 4)  if "rec_pos"  in circ else None,
                "n_seeds":        len(circ),
                "source":         "eval_full",
            }])
    return pd.DataFrame()


def load_circmac_pairing_pair():
    """Load CircMAC (Pairing) results on pair split.
    Priority: logs-based fig1_rna_lm_data.csv → eval_full CSVs.
    """
    # 1) logs 기반
    logs_csv = ROOT / "figures_paper/fig1_rna_lm/fig1_rna_lm_data.csv"
    if logs_csv.exists():
        df = pd.read_csv(logs_csv)
        circ = df[(df["model"] == "CircMAC") & (df["mode"] == "proposed")]
        if len(circ) > 0:
            return pd.DataFrame([{
                "Model":          "CircMAC (Pairing)",
                "split":          "pair",
                "N_pairs":        None,
                "Bind_AUROC":     None, "Bind_AUPRC": None, "Bind_F1": None,
                "Site_AUROC":     round(circ["roc_auc"].mean(), 4),
                "Site_AUPRC":     round(circ["auprc"].mean(), 4),
                "Site_F1":        round(circ["f1_pos"].mean(), 4) if "f1_pos" in circ else None,
                "Site_Prec":      round(circ["prec_pos"].mean(), 4) if "prec_pos" in circ else None,
                "Site_Rec":       round(circ["rec_pos"].mean(), 4)  if "rec_pos"  in circ else None,
                "Site_AUROC_std": round(circ["roc_auc"].std(), 4),
                "Site_AUPRC_std": round(circ["auprc"].std(), 4),
                "Site_F1_std":    round(circ["f1_pos"].std(), 4) if "f1_pos" in circ else None,
                "Site_Prec_std":  round(circ["prec_pos"].std(), 4) if "prec_pos" in circ else None,
                "Site_Rec_std":   round(circ["rec_pos"].std(), 4)  if "rec_pos"  in circ else None,
                "n_seeds":        len(circ),
                "source":         "logs",
            }])

    # 2) eval_full CSV fallback
    for csv_name in ["eval_full_pretrained.csv", "eval_full_summary.csv"]:
        p = EVAL / csv_name
        if not p.exists():
            continue
        df = pd.read_csv(p)
        circ = df[(df["exp_tpl"] == "v2_pt_pairing") & (df["split"] == "test")]
        if len(circ) == 0:
            circ = df[(df["label"].str.contains("CircMAC.*Pairing", na=False)) &
                      (df["split"] == "test")]
        if len(circ) > 0:
            return pd.DataFrame([{
                "Model":          "CircMAC (Pairing)",
                "split":          "pair",
                "N_pairs":        None,
                "Bind_AUROC":     None,
                "Bind_AUPRC":     None,
                "Bind_F1":        None,
                "Site_AUROC":     round(circ["auroc"].mean(), 4),
                "Site_AUPRC":     round(circ["auprc"].mean(), 4),
                "Site_F1":        round(circ["f1_pos"].mean(), 4),
                "Site_Prec":      round(circ["prec_pos"].mean(), 4) if "prec_pos" in circ else None,
                "Site_Rec":       round(circ["rec_pos"].mean(), 4)  if "rec_pos"  in circ else None,
                "Site_AUROC_std": round(circ["auroc"].std(), 4),
                "Site_AUPRC_std": round(circ["auprc"].std(), 4),
                "Site_F1_std":    round(circ["f1_pos"].std(), 4),
                "Site_Prec_std":  round(circ["prec_pos"].std(), 4) if "prec_pos" in circ else None,
                "Site_Rec_std":   round(circ["rec_pos"].std(), 4)  if "rec_pos"  in circ else None,
                "n_seeds":        len(circ),
            }])
    return pd.DataFrame()


def load_circmac_disjoint():
    """Load CircMAC results on iso/bsj splits from disjoint_new_summary.csv."""
    p = EVAL / "disjoint_new_summary.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    rows = []
    for label_filter, model_name in [
        ("CircMAC (NoPT)",    "CircMAC (NoPT)"),
        ("CircMAC",           "CircMAC (NoPT)"),   # fallback
        ("CircMAC (Pairing)", "CircMAC (Pairing)"),
    ]:
        circ = df[df["label"] == label_filter]
        for _, r in circ.iterrows():
            rows.append({
                "Model":          model_name,
                "split":          r["split"],
                "N_pairs":        None,
                "Bind_AUROC":     None,
                "Bind_AUPRC":     None,
                "Bind_F1":        None,
                "Site_AUROC":     r.get("auroc_mean"),
                "Site_AUPRC":     r.get("auprc_mean"),
                "Site_F1":        r.get("f1_pos_mean"),
                "Site_Prec":      r.get("prec_pos_mean"),
                "Site_Rec":       r.get("rec_pos_mean"),
                "Site_AUROC_std": r.get("auroc_std"),
                "Site_AUPRC_std": r.get("auprc_std"),
                "Site_F1_std":    r.get("f1_pos_std"),
                "Site_Prec_std":  r.get("prec_pos_std"),
                "Site_Rec_std":   r.get("rec_pos_std"),
                "n_seeds":        r.get("n_seeds"),
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
                "Model":          r["Model"],
                "split":          split,
                "N_pairs":        r.get("N_pairs"),
                "Bind_AUROC":     r.get("Bind_AUROC"),
                "Bind_AUPRC":     r.get("Bind_AUPRC"),
                "Bind_F1":        r.get("Bind_F1"),
                "Site_AUROC":     r.get("Site_AUROC"),
                "Site_AUPRC":     r.get("Site_AUPRC"),
                "Site_F1":        r.get("Site_F1"),
                "Site_Prec":      r.get("Site_Prec"),
                "Site_Rec":       r.get("Site_Rec"),
                "Site_AUROC_std": None,
                "Site_AUPRC_std": None,
                "Site_F1_std":    None,
                "Site_Prec_std":  None,
                "Site_Rec_std":   None,
                "n_seeds":        1,
            })
    df_tools = pd.DataFrame(tool_rows)

    # ── Load CircMAC results ───────────────────────────────────────────────
    df_circ = pd.concat([
        load_circmac_pair(),
        load_circmac_pairing_pair(),
        load_circmac_disjoint(),
    ], ignore_index=True)

    df_all = pd.concat([df_tools, df_circ], ignore_index=True)

    out_csv = OUTDIR / "full_comparison.csv"
    df_all.to_csv(out_csv, index=False)

    # ── Print table ────────────────────────────────────────────────────────
    lines = []
    def tee(s): print(s); lines.append(s)

    tee("\n" + "="*90)
    tee("  Baseline Tool vs CircMAC — Site-level Comparison")
    tee("="*90)

    def fmt(val, std=None):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "  —   "
        if std is not None and not (isinstance(std, float) and np.isnan(std)):
            return f"{val:.4f}±{std:.4f}"
        return f"{val:.4f}"

    MODEL_ORDER = ["miRanda", "RNAhybrid", "TargetScan", "PITA",
                   "CircMAC (NoPT)", "CircMAC (Pairing)"]
    SPLIT_ORDER = ["pair", "iso", "bsj"]
    SPLIT_LABEL = {"pair": "Pair-disjoint", "iso": "Iso-disjoint", "bsj": "BSJ-disjoint"}

    for split in SPLIT_ORDER:
        tee(f"\n  [{SPLIT_LABEL[split]}]")
        hdr = f"  {'Model':<20}  {'AUROC':>14}  {'AUPRC':>14}  {'F1':>14}  {'Precision':>14}  {'Recall':>14}"
        tee(hdr)
        tee("  " + "-"*90)
        sub = df_all[df_all["split"] == split]
        for model in MODEL_ORDER:
            r = sub[sub["Model"] == model]
            if r.empty:
                tee(f"  {model:<20}  {'(not available)':>70}")
                continue
            r = r.iloc[0]
            tee(f"  {model:<20}  "
                f"{fmt(r.get('Site_AUROC'), r.get('Site_AUROC_std')):>14}  "
                f"{fmt(r.get('Site_AUPRC'), r.get('Site_AUPRC_std')):>14}  "
                f"{fmt(r.get('Site_F1'),    r.get('Site_F1_std')):>14}  "
                f"{fmt(r.get('Site_Prec'),  r.get('Site_Prec_std')):>14}  "
                f"{fmt(r.get('Site_Rec'),   r.get('Site_Rec_std')):>14}")

    tee(f"\n  Note: PITA/TargetScan produce ranking scores only (no binary site predictions)")
    tee(f"        → F1/Precision/Recall = 0 or N/A for those tools.")
    tee(f"        Site-level metrics evaluated on positive (binding=1) pairs only.")
    tee(f"\n  Saved: {out_csv}")

    (OUTDIR / "full_comparison.txt").write_text("\n".join(lines))
    print(f"  Saved txt: {OUTDIR / 'full_comparison.txt'}")


if __name__ == "__main__":
    main()
