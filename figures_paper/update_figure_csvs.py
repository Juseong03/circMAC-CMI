#!/usr/bin/env python3
"""
update_figure_csvs.py
eval_results/eval_full_summary.csv (test split) → 각 figure 스크립트가 사용하는 data CSV 업데이트

Usage:
    python figures_paper/update_figure_csvs.py
    python figures_paper/update_figure_csvs.py --eval eval_results/eval_full_summary.csv
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT  = Path(__file__).resolve().parent

def load_eval(csv_path):
    df = pd.read_csv(csv_path)
    # auroc → roc_auc alias
    if "auroc" in df.columns and "roc_auc" not in df.columns:
        df["roc_auc"] = df["auroc"]
    # eval_full 형식: split 컬럼이 있으면 test만 사용
    if "split" in df.columns:
        df = df[df["split"] == "test"].copy()
    return df

def get_seeds(df, exp_tpl, model_name=None):
    """exp_tpl 기준으로 seed별 rows 반환 (숫자 seed만)"""
    mask = df["exp_tpl"] == exp_tpl
    if model_name:
        mask &= df["model_name"] == model_name
    rows = df[mask & df["seed"].apply(lambda x: str(x).isdigit())]
    return rows

# ─────────────────────────────────────────────────────────────────────────────
# Fig 1: RNA-LM comparison
# ─────────────────────────────────────────────────────────────────────────────
def update_fig1(df):
    specs = [
        # (label, model_name, exp_tpl, frozen)
        ("RNABERT",  "rnabert",  "exp1_fair_frozen_rnabert",     True),
        ("RNAErnie", "rnaernie", "exp1_fair_frozen_rnaernie",    True),
        ("RNAMSM",   "rnamsm",   "exp1_fair_frozen_rnamsm",      True),
        ("RNA-FM",   "rnafm",    "exp1_fair_frozen_rnafm",       True),
        ("RNABERT",  "rnabert",  "exp1_fair_trainable_rnabert",  False),
        ("RNAErnie", "rnaernie", "exp1_fair_trainable_rnaernie", False),
        ("RNAMSM",   "rnamsm",   "exp1_fair_trainable_rnamsm",   False),
        ("RNA-FM",   "rnafm",    "exp1_fair_trainable_rnafm",    False),
        ("CircMAC",  "circmac",  "v2_pt_pairing",                False),
    ]
    merged = {}
    for label, model_name, exp_tpl, frozen in specs:
        mode = "frozen" if frozen else "fine-tuned" if label != "CircMAC" else "proposed"
        seeds_df = get_seeds(df, exp_tpl, model_name)
        for _, row in seeds_df.iterrows():
            key = (label, mode, int(row["seed"]))
            merged[key] = {
                "model":    label, "mode": mode, "seed": int(row["seed"]),
                "f1_macro": row.get("f1_macro", ""),
                "roc_auc":  row.get("auroc", ""),
                "auprc":    row.get("auprc", ""),
            }
    out_df = pd.DataFrame(list(merged.values()))
    p = OUT / "fig1_rna_lm" / "fig1_rna_lm_data.csv"
    out_df.to_csv(p, index=False)
    print(f"  [fig1] {p}  ({len(out_df)} rows)")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 2: Pretraining strategy  (NoPT / MLM / NTP / SSP / Pairing)
# ─────────────────────────────────────────────────────────────────────────────
def update_fig2(df):
    specs = [
        ("No PT",   "v2_pt_nopt",    "baseline"),
        ("MLM",     "v2_pt_mlm",     "single"),
        ("NTP",     "v2_pt_ntp",     "single"),
        ("SSP",     "v2_pt_ssp",     "single"),
        ("Pairing", "v2_pt_pairing", "single"),
    ]
    rows = []
    for label, exp_tpl, group in specs:
        seeds_df = get_seeds(df, exp_tpl, "circmac")
        for _, row in seeds_df.iterrows():
            rows.append({
                "model": label, "group": group, "seed": int(row["seed"]),
                "f1_macro": row.get("f1_macro", ""),
                "roc_auc":  row.get("auroc", ""),
                "auprc":    row.get("auprc", ""),
            })
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
        seeds_df = get_seeds(df, exp_tpl, model_name)
        for _, row in seeds_df.iterrows():
            rows.append({
                "model": label, "seed": int(row["seed"]),
                "f1_macro": row.get("f1_macro", ""),
                "roc_auc":  row.get("auroc", ""),
                "auprc":    row.get("auprc", ""),
            })
    out_df = pd.DataFrame(rows)
    p = OUT / "fig3_encoder" / "fig3_encoder_data.csv"
    out_df.to_csv(p, index=False)
    print(f"  [fig3] {p}  ({len(out_df)} rows)")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 4: Ablation modules
# ─────────────────────────────────────────────────────────────────────────────
def update_fig4(df):
    specs = [
        ("CircMAC",    "v2_abl_full",       "full"),
        ("w/o Attn",   "v2_abl_no_attn",    "remove"),
        ("w/o Conv",   "v2_abl_no_conv",    "remove"),
        ("w/o Mamba",  "v2_abl_no_mamba",   "remove"),
        ("Mamba Only", "v2_abl_mamba_only", "single"),
        ("CNN Only",   "v2_abl_cnn_only",   "single"),
        ("Attn Only",  "v2_abl_attn_only",  "single"),
    ]
    rows = []
    for label, exp_tpl, group in specs:
        seeds_df = get_seeds(df, exp_tpl, "circmac")
        for _, row in seeds_df.iterrows():
            rows.append({
                "model": label, "group": group, "seed": int(row["seed"]),
                "f1_macro": row.get("f1_macro", ""),
                "roc_auc":  row.get("auroc", ""),
                "auprc":    row.get("auprc", ""),
            })
    out_df = pd.DataFrame(rows)
    p = OUT / "fig4_ablation_modules" / "fig4_ablation_modules_data.csv"
    out_df.to_csv(p, index=False)
    print(f"  [fig4] {p}  ({len(out_df)} rows)")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 5: Interaction & Site Head ablation
# ─────────────────────────────────────────────────────────────────────────────
def update_fig5(df):
    interaction_specs = [
        ("Cross-Attn",  "v2_int_cross_attn",  True),
        ("Concat",      "v2_int_concat",       False),
        ("Elementwise", "v2_int_elementwise",  False),
    ]
    head_specs = [
        ("Conv1D", "v2_int_cross_attn", True),
        ("Linear", "v2_head_linear",    False),
    ]
    rows = []
    for label, exp_tpl, is_best in interaction_specs + head_specs:
        ablation = "interaction" if (label, exp_tpl, is_best) in [
            (l, e, b) for l, e, b in interaction_specs] else "head"
        seeds_df = get_seeds(df, exp_tpl, "circmac")
        for _, row in seeds_df.iterrows():
            rows.append({
                "model": label, "ablation": ablation,
                "is_best": is_best, "seed": int(row["seed"]),
                "f1_macro": row.get("f1_macro", ""),
                "roc_auc":  row.get("auroc", ""),
                "auprc":    row.get("auprc", ""),
            })
    out_df = pd.DataFrame(rows)
    p = OUT / "fig5_ablation_int_head" / "fig5_ablation_int_head_data.csv"
    out_df.to_csv(p, index=False)
    print(f"  [fig5] {p}  ({len(out_df)} rows)")

# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", default=str(ROOT / "eval_results/eval_full_summary.csv"))
    args = parser.parse_args()

    csv_path = Path(args.eval)
    if not csv_path.exists():
        print(f"[ERROR] not found: {csv_path}")
        print("  Run: python scripts/eval_full.py --group merge")
        return

    print(f"Loading {csv_path} ...")
    df = load_eval(csv_path)
    print(f"  {len(df)} rows (test split only)")
    print()

    print("Updating figure data CSVs...")
    for fn, name in [(update_fig1, "fig1"), (update_fig2, "fig2"),
                     (update_fig3, "fig3"), (update_fig4, "fig4"),
                     (update_fig5, "fig5")]:
        try:
            fn(df)
        except Exception as e:
            print(f"  [{name}] SKIP — {e}")

    print()
    print("Done! Now re-run figure scripts:")
    print("  python figures_paper/fig1_rna_lm/fig1_rna_lm.py")
    print("  python figures_paper/fig2_pretraining/fig2_pretraining.py")
    print("  python figures_paper/fig3_encoder/fig3_encoder.py")
    print("  python figures_paper/fig4_ablation_modules/fig4_ablation_modules.py")
    print("  python figures_paper/fig5_ablation_int_head/fig5_ablation_int_head.py")

if __name__ == "__main__":
    main()
