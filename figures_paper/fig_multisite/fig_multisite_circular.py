#!/usr/bin/env python3
"""
fig_multisite_circular.py
=========================
Circular (polar) visualization of multi-site binding predictions.

Layout: grid of circles
  rows    = cases  (circFANCA, circKHDRBS1, circJARID2)
  columns = models

Each circle:
  - Outer ring : Ground truth  (red = binding, light gray = no binding)
  - Inner fill : Model predicted probability (radial bars)
  - BSJ mark   : blue tick at top (angle = 0)

Output:
    fig_multisite_circular_encoder.{pdf,png}
    fig_multisite_circular_pretrained.{pdf,png}

Usage:
    python figures_paper/fig_multisite/fig_multisite_circular.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT     = Path(__file__).resolve().parents[2]
OUT      = Path(__file__).resolve().parent
DATA_CSV = OUT / "data_predictions.csv"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size":   9,
    "pdf.fonttype": 42,
    "ps.fonttype":  42,
})

# ── Cases ─────────────────────────────────────────────────────────────────────
CASES = [
    dict(label="circFANCA",
         isoform_prefix="chr16|89782859",
         mirna="hsa-miR-6858-5p"),
    dict(label="circKHDRBS1",
         isoform_prefix="chr1|32036910,32037835,32038552|32037043",
         mirna="hsa-miR-6880-5p"),
    dict(label="circJARID2",
         isoform_prefix="chr6|15410224,15452006,15468542,15487307",
         mirna="hsa-miR-608"),
]

# ── Model groups ──────────────────────────────────────────────────────────────
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

GROUPS = {
    "encoder": [
        ("lstm",        "LSTM",        "pred_lstm"),
        ("transformer", "Transformer", "pred_transformer"),
        ("mamba",       "Mamba",       "pred_mamba"),
        ("hymba",       "Hymba",       "pred_hymba"),
        ("circmac",     "CircMAC",     "pred_circmac_nopt"),
    ],
    "pretrained": [
        ("rnabert",  "RNABERT",  "pred_rnabert_ft"),
        ("rnaernie", "RNAErnie", "pred_rnaernie_ft"),
        ("rnamsm",   "RNAMSM",   "pred_rnamsm_ft"),
        ("rnafm",    "RNA-FM",   "pred_rnafm_ft"),
        ("circmac",  "CircMAC",  "pred_circmac_nopt"),
    ],
}

GT_BIND_COLOR   = "#D62728"
GT_NOBIND_COLOR = "#EEEEEE"
BSJ_COLOR       = "#1F77B4"
GT_H            = 0.12   # GT ring height (fraction of radius)
PRED_MAX        = 0.82   # prediction fills up to this radius


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_case(df_all, case):
    mask = (
        df_all["isoform_ID"].str.startswith(case["isoform_prefix"]) &
        (df_all["miRNA_ID"] == case["mirna"])
    )
    df = df_all[mask].sort_values("position").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No rows for {case['label']} / {case['mirna']}")
    return df


def draw_one(ax, df, mkey, mname, mcol, show_title_case=None, show_model_col=True):
    """Draw one polar circle: GT outer ring + one model's prediction."""
    gt    = df["ground_truth"].values
    L     = len(df)
    theta = np.linspace(0, 2 * np.pi, L, endpoint=False)
    bar_w = 2 * np.pi / L
    color = MODEL_COLORS.get(mkey, "#888888")

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines["polar"].set_visible(False)
    ax.set_ylim(0, 1.10)

    # ── Prediction radial bars ─────────────────────────────────────────────
    if mcol in df.columns and not df[mcol].isna().all():
        pred = df[mcol].fillna(0).values.astype(float)
        ax.bar(theta, pred * PRED_MAX, width=bar_w * 0.95, bottom=0.0,
               color=color, alpha=0.80, edgecolor="none", align="edge")

    # ── GT outer ring ──────────────────────────────────────────────────────
    r_gt = 1.0 - GT_H
    ax.bar(theta, GT_H, width=bar_w, bottom=r_gt,
           color=np.where(gt > 0.5, GT_BIND_COLOR, GT_NOBIND_COLOR),
           alpha=0.88, edgecolor="none", align="edge")

    # outer circle outline
    tc = np.linspace(0, 2 * np.pi, 512)
    ax.plot(tc, np.full_like(tc, 1.0), color="#555555", lw=0.7)
    ax.plot(tc, np.full_like(tc, r_gt), color="#aaaaaa", lw=0.4)

    # BSJ tick at top
    ax.plot([0, 0], [0.97, 1.07], color=BSJ_COLOR, lw=2.0, solid_capstyle="round")

    # ── Titles ─────────────────────────────────────────────────────────────
    if show_title_case:
        mirna_short = show_title_case["mirna"].replace("hsa-", "")
        gt_n = int(gt.sum())
        diff = np.diff(np.concatenate([[0], gt, [0]]))
        n_cl = int((diff == 1).sum())
        ax.set_title(
            f"{show_title_case['label']}\n"
            f"L={L}, {gt_n} sites, {n_cl} cl.",
            fontsize=8.5, fontweight="bold", pad=12, color="#222222",
        )

    if show_model_col:
        ax.text(0.5, -0.08, mname,
                ha="center", va="top", fontsize=9,
                fontweight="bold" if mkey == "circmac" else "normal",
                color=color, transform=ax.transAxes)


# ── Figure builder ────────────────────────────────────────────────────────────
def make_figure(group_key, df_all):
    models  = GROUPS[group_key]
    n_cases = len(CASES)
    n_models= len(models)

    fig, axes = plt.subplots(
        n_cases, n_models,
        figsize=(3.2 * n_models, 3.6 * n_cases),
        subplot_kw={"projection": "polar"},
        gridspec_kw={"hspace": 0.45, "wspace": 0.20},
    )

    for ri, case in enumerate(CASES):
        df = load_case(df_all, case)
        for ci, (mkey, mname, mcol) in enumerate(models):
            ax = axes[ri][ci]
            draw_one(
                ax, df, mkey, mname, mcol,
                show_title_case=case if ci == 0 else None,
                show_model_col=(ri == 0),
            )

    # model name labels on top row
    for ci, (mkey, mname, mcol) in enumerate(models):
        color = MODEL_COLORS.get(mkey, "#888")
        axes[0][ci].set_title(
            mname,
            fontsize=10,
            fontweight="bold" if mkey == "circmac" else "normal",
            color=color,
            pad=14,
        )

    # case labels on left
    for ri, case in enumerate(CASES):
        df   = load_case(df_all, case)
        gt   = df["ground_truth"].values
        gt_n = int(gt.sum())
        diff = np.diff(np.concatenate([[0], gt, [0]]))
        n_cl = int((diff == 1).sum())
        axes[ri][0].set_ylabel(
            f"{case['label']}\nL={len(df)}, {gt_n} sites, {n_cl} cl.",
            fontsize=9, fontweight="bold", labelpad=70, rotation=0,
            va="center", ha="right",
        )

    # Legend
    handles = [
        mpatches.Patch(color=GT_BIND_COLOR,   alpha=0.85, label="GT binding"),
        mpatches.Patch(color=GT_NOBIND_COLOR, ec="#aaaaaa", label="GT non-binding"),
        plt.Line2D([0], [0], color=BSJ_COLOR, lw=2.0, label="BSJ"),
    ] + [
        mpatches.Patch(color=MODEL_COLORS.get(mkey, "#888"), alpha=0.80,
                       label=mname + (" ★" if mkey == "circmac" else ""))
        for mkey, mname, _ in models
    ]
    fig.legend(handles=handles, loc="lower center",
               ncol=min(len(handles), 5), fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.04))

    title = ("Multi-site Binding — Encoder Comparison (Circular)"
             if group_key == "encoder"
             else "Multi-site Binding — RNA-LM Comparison (Circular)")
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)

    for ext in ["pdf", "png"]:
        p = OUT / f"fig_multisite_circular_{group_key}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"Saved → {p}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not DATA_CSV.exists():
        raise FileNotFoundError(
            f"{DATA_CSV} not found.\n"
            "Run: python scripts/extract_multisite_predictions.py --device 0"
        )
    df_all = pd.read_csv(DATA_CSV)
    make_figure("encoder",    df_all)
    make_figure("pretrained", df_all)
    print("Done.")


if __name__ == "__main__":
    main()
