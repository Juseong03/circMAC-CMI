#!/usr/bin/env python3
"""
update_figure_csvs.py
eval_results/eval_full_summary.csv (test split) → 각 figure 스크립트가 사용하는 data CSV 업데이트

Seeds: 1, 2, 3 only (fair comparison across all experiments)

Usage:
    python figures_paper/update_figure_csvs.py
    python figures_paper/update_figure_csvs.py --eval eval_results/eval_full_summary.csv
    python figures_paper/update_figure_csvs.py --disjoint eval_results/disjoint_new_summary.csv
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

ROOT  = Path(__file__).resolve().parents[1]
OUT   = Path(__file__).resolve().parent
SEEDS = [1, 2, 3]  # fixed for fair comparison across all experiments


def load_eval(csv_path):
    df = pd.read_csv(csv_path)
    if "auroc" in df.columns and "roc_auc" not in df.columns:
        df["roc_auc"] = df["auroc"]
    if "split" in df.columns:
        df = df[df["split"] == "test"].copy()
    return df


def get_seeds(df, exp_tpl, model_name=None):
    """exp_tpl 기준으로 seed 1-3 rows만 반환."""
    mask = df["exp_tpl"] == exp_tpl
    if model_name:
        mask &= df["model_name"] == model_name
    rows = df[mask & df["seed"].apply(lambda x: str(x).isdigit())]
    rows = rows[rows["seed"].astype(int).isin(SEEDS)]
    return rows


def _row(label, group, row, extra=None):
    d = {
        "model":    label,
        "group":    group,
        "seed":     int(row["seed"]),
        "f1_macro": row.get("f1_macro", ""),
        "roc_auc":  row.get("auroc", row.get("roc_auc", "")),
        "auprc":    row.get("auprc", ""),
        "f1_pos":   row.get("f1_pos", ""),
    }
    if extra:
        d.update(extra)
    return d


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1: RNA-LM comparison (frozen vs fine-tuned vs CircMAC)
# ─────────────────────────────────────────────────────────────────────────────
def update_fig1(df):
    specs = [
        ("RNABERT",  "rnabert",  "exp1_fair_frozen_rnabert",     "frozen"),
        ("RNAErnie", "rnaernie", "exp1_fair_frozen_rnaernie",    "frozen"),
        ("RNAMSM",   "rnamsm",   "exp1_fair_frozen_rnamsm",      "frozen"),
        ("RNA-FM",   "rnafm",    "exp1_fair_frozen_rnafm",       "frozen"),
        ("RNABERT",  "rnabert",  "exp1_fair_trainable_rnabert",  "fine-tuned"),
        ("RNAErnie", "rnaernie", "exp1_fair_trainable_rnaernie", "fine-tuned"),
        ("RNAMSM",   "rnamsm",   "exp1_fair_trainable_rnamsm",   "fine-tuned"),
        ("RNA-FM",   "rnafm",    "exp1_fair_trainable_rnafm",    "fine-tuned"),
        ("CircMAC",  "circmac",  "v2_pt_pairing",                "proposed"),
    ]
    merged = {}
    for label, model_name, exp_tpl, mode in specs:
        for _, row in get_seeds(df, exp_tpl, model_name).iterrows():
            key = (label, mode, int(row["seed"]))
            merged[key] = _row(label, mode, row, {"mode": mode})
    out_df = pd.DataFrame(list(merged.values()))
    p = OUT / "fig1_rna_lm" / "fig1_rna_lm_data.csv"
    out_df.to_csv(p, index=False)
    print(f"  [fig1] {p}  ({len(out_df)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2: Pretraining strategy (all strategies)
# ─────────────────────────────────────────────────────────────────────────────
def update_fig2(df):
    specs = [
        # (label, exp_tpl, group)
        ("No PT",       "v2_pt_nopt",        "baseline"),
        ("MLM",         "v2_pt_mlm",         "single"),
        ("NTP",         "v2_pt_ntp",         "single"),
        ("SSP",         "v2_pt_ssp",         "single"),
        ("Pairing",     "v2_pt_pairing",     "single"),
        ("MLM+NTP",     "v2_pt_mlm_ntp",     "combination"),
        ("MLM+SSP",     "v2_pt_mlm_ssp",     "combination"),
        ("MLM+Pairing", "v2_pt_mlm_pairing", "combination"),
        ("SSP+Pairing", "v2_pt_ssp_pairing", "combination"),
        ("MLM+NTP+SSP", "v2_pt_mlm_ntp_ssp", "combination"),
        ("All",         "v2_pt_all",         "combination"),
    ]
    rows = []
    for label, exp_tpl, group in specs:
        for _, row in get_seeds(df, exp_tpl, "circmac").iterrows():
            rows.append(_row(label, group, row))
    out_df = pd.DataFrame(rows)
    p = OUT / "fig2_pretraining" / "fig2_pretraining_data.csv"
    out_df.to_csv(p, index=False)
    print(f"  [fig2] {p}  ({len(out_df)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3: Encoder comparison
# ─────────────────────────────────────────────────────────────────────────────
def update_fig3(df):
    specs = [
        ("LSTM",        "lstm",        "v2_enc_lstm"),
        ("Transformer", "transformer", "v2_enc_transformer"),
        ("Mamba",       "mamba",       "v2_enc_mamba"),
        ("Hymba",       "hymba",       "v2_enc_hymba"),
        ("CircMAC",     "circmac",     "v2_abl_full"),
    ]
    rows = []
    for label, model_name, exp_tpl in specs:
        for _, row in get_seeds(df, exp_tpl, model_name).iterrows():
            rows.append(_row(label, "encoder", row))
    out_df = pd.DataFrame(rows)
    p = OUT / "fig3_encoder" / "fig3_encoder_data.csv"
    out_df.to_csv(p, index=False)
    print(f"  [fig3] {p}  ({len(out_df)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4: Ablation modules
# ─────────────────────────────────────────────────────────────────────────────
def update_fig4(df):
    specs = [
        ("CircMAC",    "v2_abl_full",        "full"),
        ("w/o Attn",   "v2_abl_no_attn",     "remove"),
        ("w/o Conv",   "v2_abl_no_conv",      "remove"),
        ("w/o Mamba",  "v2_abl_no_mamba",    "remove"),
        ("w/o CircBias","v2_abl_no_circ_bias","remove"),
        ("Mamba Only", "v2_abl_mamba_only",  "single"),
        ("CNN Only",   "v2_abl_cnn_only",    "single"),
        ("Attn Only",  "v2_abl_attn_only",   "single"),
    ]
    rows = []
    for label, exp_tpl, group in specs:
        for _, row in get_seeds(df, exp_tpl, "circmac").iterrows():
            rows.append(_row(label, group, row))
    out_df = pd.DataFrame(rows)
    p = OUT / "fig4_ablation_modules" / "fig4_ablation_modules_data.csv"
    out_df.to_csv(p, index=False)
    print(f"  [fig4] {p}  ({len(out_df)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5: Interaction & Site Head ablation
# ─────────────────────────────────────────────────────────────────────────────
def update_fig5(df):
    specs = [
        # (label, exp_tpl, ablation_type, is_best)
        ("Cross-Attn",  "v2_int_cross_attn",  "interaction", True),
        ("Concat",      "v2_int_concat",       "interaction", False),
        ("Elementwise", "v2_int_elementwise",  "interaction", False),
        ("Conv1D",      "v2_int_cross_attn",   "head",        True),
        ("Linear",      "v2_head_linear",      "head",        False),
    ]
    rows = []
    for label, exp_tpl, ablation, is_best in specs:
        for _, row in get_seeds(df, exp_tpl, "circmac").iterrows():
            rows.append(_row(label, ablation, row, {"is_best": is_best}))
    out_df = pd.DataFrame(rows)
    p = OUT / "fig5_ablation_int_head" / "fig5_ablation_int_head_data.csv"
    out_df.to_csv(p, index=False)
    print(f"  [fig5] {p}  ({len(out_df)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# Disjoint splits: iso / bsj  (from disjoint_new_summary.csv)
# ─────────────────────────────────────────────────────────────────────────────
def update_disjoint(disjoint_csv):
    """
    disjoint_new_summary.csv → figures_paper/fig_disjoint/fig_disjoint_data.csv

    Columns in summary: split, label, n_seeds, auroc_mean, auroc_std,
                        auprc_mean, auprc_std, f1_pos_mean, f1_pos_std, ...
    """
    p_out = OUT / "fig_disjoint"
    p_out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(disjoint_csv)

    # Keep: encoders + RNA-LMs + CircMAC (NoPT/MLM/SSP/Pairing)
    KEEP_LABELS = {
        "LSTM", "Transformer", "Mamba", "Hymba",
        "RNABERT (ft)", "RNAErnie (ft)", "RNAMSM (ft)", "RNA-FM (ft)",
        "CircMAC (NoPT)", "CircMAC (MLM)", "CircMAC (SSP)", "CircMAC (Pairing)",
    }

    rows = []
    for _, r in df.iterrows():
        label = r["label"]
        split = r["split"]
        if label not in KEEP_LABELS:
            continue
        group = "pretraining" if label.startswith("CircMAC") else "encoder"
        rows.append({
            "split":       split,
            "model":       label,
            "group":       group,
            "n_seeds":     r.get("n_seeds", ""),
            "auroc_mean":  r.get("auroc_mean", ""),
            "auroc_std":   r.get("auroc_std",  ""),
            "auprc_mean":  r.get("auprc_mean", ""),
            "auprc_std":   r.get("auprc_std",  ""),
            "f1pos_mean":  r.get("f1_pos_mean", ""),
            "f1pos_std":   r.get("f1_pos_std",  ""),
            "f1mac_mean":  r.get("f1_macro_mean", ""),
            "f1mac_std":   r.get("f1_macro_std",  ""),
        })

    out_df = pd.DataFrame(rows)
    csv_path = p_out / "fig_disjoint_data.csv"
    out_df.to_csv(csv_path, index=False)
    print(f"  [disjoint] {csv_path}  ({len(out_df)} rows)")

    # Also save separate iso / bsj CSVs for convenience
    for split in ["iso", "bsj"]:
        sub = out_df[out_df["split"] == split]
        sub.to_csv(p_out / f"fig_disjoint_{split}_data.csv", index=False)
        print(f"  [disjoint/{split}] {len(sub)} rows")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval",
                        default=str(ROOT / "eval_results/eval_full_summary.csv"),
                        help="Path to eval_full_summary.csv")
    parser.add_argument("--disjoint",
                        default=str(ROOT / "eval_results/disjoint_new_summary.csv"),
                        help="Path to disjoint_new_summary.csv")
    args = parser.parse_args()

    # ── Pair-split figures ────────────────────────────────────────────────────
    csv_path = Path(args.eval)
    if not csv_path.exists():
        print(f"[WARN] not found: {csv_path}")
        print("  Run: python scripts/eval_full.py --group merge")
    else:
        print(f"Loading {csv_path} ...")
        df = load_eval(csv_path)
        print(f"  {len(df)} rows (test split, seeds {SEEDS} only)")
        print()
        print("Updating figure data CSVs (pair split)...")
        for fn, name in [
            (update_fig1, "fig1"),
            (update_fig2, "fig2"),
            (update_fig3, "fig3"),
            (update_fig4, "fig4"),
            (update_fig5, "fig5"),
        ]:
            try:
                fn(df)
            except Exception as e:
                print(f"  [{name}] SKIP — {e}")

    # ── Disjoint split figure ─────────────────────────────────────────────────
    print()
    disj_path = Path(args.disjoint)
    if not disj_path.exists():
        print(f"[WARN] not found: {disj_path}")
        print("  Run: python scripts/eval_disjoint_new.py --device 0")
    else:
        print(f"Loading {disj_path} ...")
        try:
            update_disjoint(disj_path)
        except Exception as e:
            print(f"  [disjoint] SKIP — {e}")

    print()
    print("Done! Now re-run figure scripts:")
    print("  python figures_paper/fig1_rna_lm/fig1_rna_lm.py")
    print("  python figures_paper/fig2_pretraining/fig2_pretraining.py")
    print("  python figures_paper/fig3_encoder/fig3_encoder.py")
    print("  python figures_paper/fig4_ablation_modules/fig4_ablation_modules.py")
    print("  python figures_paper/fig5_ablation_int_head/fig5_ablation_int_head.py")
    print("  python figures_paper/fig_disjoint/fig_disjoint.py  (create if needed)")


if __name__ == "__main__":
    main()
