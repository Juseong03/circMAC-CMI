#!/usr/bin/env python3
"""
fig_multisite_catalog.py — Catalog of ALL 50 multisite cases (5 per row)

Each cell shows:
  - Title: circ{gene} × miRNA (short), L=, n_sites=
  - Row 0: Ground truth
  - Row 1: CircMAC prediction
  - Row 2: Best encoder competitor (Hymba, highest F1 typically)
  - Annotation: CircMAC F1 / AUROC vs best competitor F1

Output: fig_multisite_catalog.{pdf,png}
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT  = Path(__file__).resolve().parent
DATA_CSV = OUT / "data_predictions.csv"

ENCODER_MODELS = [
    ("lstm",        "LSTM",        "pred_lstm"),
    ("transformer", "Transformer", "pred_transformer"),
    ("mamba",       "Mamba",       "pred_mamba"),
    ("hymba",       "Hymba",       "pred_hymba"),
]

LM_MODELS = [
    ("rnabert",  "RNABERT",  "pred_rnabert_ft"),
    ("rnaernie", "RNAErnie", "pred_rnaernie_ft"),
    ("rnamsm",   "RNAMSM",   "pred_rnamsm_ft"),
    ("rnafm",    "RNA-FM",   "pred_rnafm_ft"),
]

MODEL_COLORS = {
    "circmac":     "#FF7F0E",
    "lstm":        "#E377C2",
    "transformer": "#8C564B",
    "mamba":       "#D62728",
    "hymba":       "#BCBD22",
    "rnabert":     "#9467BD",
    "rnaernie":    "#8C8C00",
    "rnamsm":      "#2CA02C",
    "rnafm":       "#17BECF",
}
NAN_COLOR  = "#cccccc"
GT_COLOR   = "#8B0000"
BSJ_COLOR  = "#1F77B4"

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        8,
    "axes.linewidth":   0.6,
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
})


def f1_score_simple(gt, pred_prob, thr=0.5):
    pred = (pred_prob >= thr).astype(int)
    tp = ((gt == 1) & (pred == 1)).sum()
    fp = ((gt == 0) & (pred == 1)).sum()
    fn = ((gt == 1) & (pred == 0)).sum()
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def draw_bsj(ax, L, lw=0.8, alpha=0.5):
    ax.axvline(-0.5,    color=BSJ_COLOR, lw=lw, ls="--", alpha=alpha)
    ax.axvline(L - 0.5, color=BSJ_COLOR, lw=lw, ls="--", alpha=alpha)


def draw_case(fig, gs_cell, df_case, case_idx, gene, mirna_short, L, n_sites):
    """Draw one catalog cell: GT + CircMAC + best encoder + best LM rows."""
    gt = df_case["ground_truth"].values

    # Compute CircMAC metrics
    cmc_col = "pred_circmac_nopt"
    cmc_prob = df_case[cmc_col].fillna(0).values if cmc_col in df_case.columns else np.zeros(L)
    cmc_f1   = f1_score_simple(gt, cmc_prob)

    # Best encoder (highest F1 among ENCODER_MODELS)
    best_enc_f1, best_enc_key, best_enc_col, best_enc_name = 0, "hymba", "pred_hymba", "Hymba"
    for mkey, mname, mcol in ENCODER_MODELS:
        if mcol in df_case.columns:
            p = df_case[mcol].fillna(0).values
            f = f1_score_simple(gt, p)
            if f > best_enc_f1:
                best_enc_f1, best_enc_key, best_enc_col, best_enc_name = f, mkey, mcol, mname

    # Best LM (highest F1 among LM_MODELS)
    best_lm_f1, best_lm_key, best_lm_col, best_lm_name = 0, "rnamsm", "pred_rnamsm_ft", "RNAMSM"
    for mkey, mname, mcol in LM_MODELS:
        if mcol in df_case.columns:
            p = df_case[mcol].fillna(0).values
            f = f1_score_simple(gt, p)
            if f > best_lm_f1:
                best_lm_f1, best_lm_key, best_lm_col, best_lm_name = f, mkey, mcol, mname

    # Inner grid: title + 4 rows (GT, CircMAC, best enc, best LM)
    gs_inner = GridSpecFromSubplotSpec(
        4, 1, subplot_spec=gs_cell,
        height_ratios=[1.4, 1.0, 1.0, 1.0],
        hspace=0.06,
    )

    rows_data = [
        ("GT",       GT_COLOR,                          "ground_truth",  None),
        ("CircMAC",  MODEL_COLORS["circmac"],            cmc_col,        cmc_f1),
        (best_enc_name, MODEL_COLORS[best_enc_key],      best_enc_col,   best_enc_f1),
        (best_lm_name,  MODEL_COLORS.get(best_lm_key, "#888"), best_lm_col, best_lm_f1),
    ]

    for ri, (row_label, color, col, f1_val) in enumerate(rows_data):
        ax = fig.add_subplot(gs_inner[ri])

        if col in df_case.columns:
            raw = df_case[col].values.astype(float)
            masked = np.ma.masked_invalid(raw)
            if col == "ground_truth":
                cmap = mcolors.LinearSegmentedColormap.from_list("gt", ["#f7f7f7", color])
            else:
                cmap = mcolors.LinearSegmentedColormap.from_list(col, ["#f7f7f7", color])
            cmap.set_bad(color=NAN_COLOR)
            ax.imshow(masked[np.newaxis, :], aspect="auto",
                      cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        else:
            ax.set_facecolor("#eeeeee")

        draw_bsj(ax, L)
        ax.set_yticks([])
        ax.set_xticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

        # Row label on left
        label_txt = row_label
        if f1_val is not None:
            label_txt += f"\nF1={f1_val:.2f}"
        ax.set_ylabel(label_txt, rotation=0, ha="right", va="center",
                      fontsize=6.5, labelpad=4,
                      color=color if col != "ground_truth" else "#333333",
                      fontweight="bold" if col in ("pred_circmac", "ground_truth") else "normal")

        if ri == 0:
            n_cl = int((np.diff(np.concatenate([[0], gt, [0]])) == 1).sum())
            ax.set_title(
                f"[{case_idx}] circ{gene} × {mirna_short}\n"
                f"L={L}, {n_sites} sites, {n_cl} clusters",
                fontsize=7.5, fontweight="bold", pad=3, loc="center",
            )

        if ri == 3:
            ax.set_xticks([0, L // 2, L - 1])
            ax.set_xticklabels([0, L // 2, L - 1], fontsize=5.5)
            ax.tick_params(axis="x", length=2)


def main():
    df_all = pd.read_csv(DATA_CSV)
    pairs  = df_all[["isoform_ID", "miRNA_ID", "gene_name", "length"]].drop_duplicates().reset_index(drop=True)
    N = len(pairs)

    NCOLS = 5
    NROWS = int(np.ceil(N / NCOLS))

    cell_w = 4.2
    cell_h = 3.8
    fig_w  = NCOLS * cell_w
    fig_h  = NROWS * cell_h

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = GridSpec(NROWS, NCOLS, figure=fig, hspace=0.55, wspace=0.45)

    for i, row in pairs.iterrows():
        r, c = divmod(i, NCOLS)
        mask = (df_all["isoform_ID"] == row["isoform_ID"]) & (df_all["miRNA_ID"] == row["miRNA_ID"])
        df_case = df_all[mask].sort_values("position").reset_index(drop=True)
        gene        = row["gene_name"]
        mirna_short = row["miRNA_ID"].replace("hsa-", "")
        L           = int(row["length"])
        n_sites     = int(df_case["ground_truth"].sum())
        draw_case(fig, gs[r, c], df_case, i, gene, mirna_short, L, n_sites)

    fig.suptitle(
        "Multi-site binding catalog — all 50 cases\n"
        "(GT / CircMAC / best encoder / best LM fine-tuned)",
        fontsize=11, fontweight="bold", y=1.01,
    )

    for ext in ["pdf", "png"]:
        p = OUT / f"fig_multisite_catalog.{ext}"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved → {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
