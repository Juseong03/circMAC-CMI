#!/usr/bin/env python3
"""
Fig 9 — Heatmap + Per-case Metrics (AUROC / AUPRC)

Two subplots per figure:
  (a) Nucleotide-level prediction heatmap  (GT row + model rows, BSJ lines)
  (b) Per-case AUROC and AUPRC bar charts  (3 cases × model bars)

Two figures generated (encoder / pretrained).

Output:
  fig9_encoder_heatmap_metrics.{pdf,png}
  fig9_pretrained_heatmap_metrics.{pdf,png}
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score


# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
OUT  = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)


# ── Case definitions ─────────────────────────────────────────────────────────
CASES = [
    dict(
        label   = "circCDYL2\n(chr4)",
        label_m = "circCDYL2",
        csv     = ROOT / "figures_claude/Fig5_Case_CDYL2/data_predictions.csv",
        isoform = "chr4|84678168,84679116|84678259,84679242|-",
        mirna   = "hsa-miR-449a",
    ),
    dict(
        label   = "circMAPK1\n(chr22)",
        label_m = "circMAPK1",
        csv     = ROOT / "figures_claude/Fig6_Case_MAPK1/data_predictions.csv",
        isoform = "chr22|21799012,21805850,21807664|21799128,21806039,21807846|-",
        mirna   = "hsa-miR-12119",
    ),
    dict(
        label   = "circAPP\n(chr21)",
        label_m = "circAPP",
        csv     = ROOT / "figures_claude/Fig7_Case_APP/data_predictions.csv",
        isoform = "chr21|25954590,25955627,25975070,25975954,25982344|25954689,25955755,25975228,25976028,25982477|-",
        mirna   = "hsa-miR-5001-3p",
    ),
]


# ── Model groups ─────────────────────────────────────────────────────────────
MODEL_COLORS = {
    "circmac":     "#FF7F0E",
    "lstm":        "#E377C2",
    "transformer": "#8C564B",
    "mamba":       "#D62728",
    "hymba":       "#BCBD22",
    "rnabert":     "#1F77B4",
    "rnaernie":    "#9467BD",
    "rnamsm":      "#2CA02C",
    "rnafm":       "#17BECF",
}

BSJ_COLOR = "#1F77B4"
GT_COLOR  = "#8B0000"

# (color_key, display_label, pred_col)
GROUPS = {
    "encoder": [
        ("lstm",        "LSTM",           "pred_lstm"),
        ("transformer", "Transformer",    "pred_transformer"),
        ("mamba",       "Mamba",          "pred_mamba"),
        ("hymba",       "Hymba",          "pred_hymba"),
        ("circmac",     "circMAC (ours)", "pred_circmac"),
    ],
    "pretrained": [
        ("rnabert",  "RNABERT\n(fine-tuned)",  "pred_rnabert"),
        ("rnaernie", "RNAErnie\n(fine-tuned)", "pred_rnaernie"),
        ("rnamsm",   "RNAMSM\n(fine-tuned)",   "pred_rnamsm"),
        ("rnafm",    "RNA-FM\n(fine-tuned)",   "pred_rnafm"),
        ("circmac",  "circMAC (ours)",          "pred_circmac"),
    ],
}


# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         9,
    "axes.linewidth":    0.8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
})


# ── Helpers ──────────────────────────────────────────────────────────────────
def load_case(case):
    df_all = pd.read_csv(case["csv"])
    df = df_all[
        (df_all["miRNA_ID"]   == case["mirna"])  &
        (df_all["isoform_ID"] == case["isoform"])
    ].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No rows for {case['label_m']} / {case['mirna']}")
    return df


def compute_metrics(df, models):
    """Return dict {pred_col: {auroc, auprc}}."""
    gt  = df["ground_truth"].values
    out = {}
    for mkey, mname, mcol in models:
        if mcol not in df.columns:
            out[mcol] = dict(auroc=float("nan"), auprc=float("nan"))
            continue
        p = df[mcol].values
        try:
            auroc = roc_auc_score(gt, p)  if gt.sum() > 0 else float("nan")
            auprc = average_precision_score(gt, p) if gt.sum() > 0 else float("nan")
        except Exception:
            auroc, auprc = float("nan"), float("nan")
        out[mcol] = dict(auroc=auroc, auprc=auprc)
    return out


def draw_bsj(ax, L, lw=1.0, alpha=0.55):
    ax.axvline(-0.5,    color=BSJ_COLOR, lw=lw, ls="--", alpha=alpha)
    ax.axvline(L - 0.5, color=BSJ_COLOR, lw=lw, ls="--", alpha=alpha)


def clean_spines(ax):
    for sp in ax.spines.values():
        sp.set_visible(False)


# ── Heatmap section ──────────────────────────────────────────────────────────
def draw_heatmap_section(fig, gs_slot, models, first_case=True):
    """
    Draw (GT + model rows) × n_cases inside gs_slot.
    gs_slot is a GridSpec row of n_cases columns.
    """
    n_models = len(models)
    n_cases  = len(CASES)

    # Inner: n_cases columns
    gs_inner = GridSpecFromSubplotSpec(
        1, n_cases,
        subplot_spec=gs_slot,
        wspace=0.14,
    )

    for ci, case in enumerate(CASES):
        df = load_case(case)
        gt = df["ground_truth"].values
        L  = len(df)

        gs_case = GridSpecFromSubplotSpec(
            n_models + 1, 1,
            subplot_spec=gs_inner[ci],
            height_ratios=[1.35] + [1.0] * n_models,
            hspace=0.08,
        )

        # Ground-truth row
        ax_gt = fig.add_subplot(gs_case[0])
        ax_gt.imshow(gt[np.newaxis, :], aspect="auto", cmap="Reds",
                     vmin=0, vmax=1, interpolation="nearest")
        ax_gt.set_yticks([])
        ax_gt.set_xticks([])
        draw_bsj(ax_gt, L, lw=1.4, alpha=0.75)
        clean_spines(ax_gt)

        if ci == 0:
            ax_gt.set_ylabel("Ground truth", rotation=0, ha="right", va="center",
                              fontsize=8, fontweight="bold", color=GT_COLOR, labelpad=42)

        mirna_short = case["mirna"].replace("hsa-", "")
        ax_gt.set_title(
            f"{case['label']}\n{mirna_short}",
            fontsize=9, fontweight="bold", pad=6,
        )

        # Model rows
        for ri, (mkey, mname, mcol) in enumerate(models):
            ax = fig.add_subplot(gs_case[ri + 1])
            if mcol in df.columns:
                pred  = df[mcol].values
                color = MODEL_COLORS.get(mkey, "#888888")
                cmap  = mcolors.LinearSegmentedColormap.from_list(
                    f"{mkey}_cm", ["#f7f7f7", color])
                ax.imshow(pred[np.newaxis, :], aspect="auto", cmap=cmap,
                          vmin=0, vmax=1, interpolation="nearest")
            else:
                ax.set_facecolor("#eeeeee")

            ax.set_yticks([])
            ax.set_xticks([])
            draw_bsj(ax, L, lw=0.9, alpha=0.5)
            clean_spines(ax)

            if ci == 0:
                color = MODEL_COLORS.get(mkey, "#888888")
                ax.set_ylabel(mname, rotation=0, ha="right", va="center",
                              fontsize=8, fontweight="bold", color=color, labelpad=42)

            # x-axis on bottom row
            if ri == n_models - 1:
                xticks = np.linspace(0, L - 1, 5, dtype=int)
                ax.set_xticks(xticks)
                ax.set_xticklabels([str(x) for x in xticks], fontsize=7)
                ax.set_xlabel("Sequence position", fontsize=7.5, labelpad=3)


# ── Metrics bar chart section ────────────────────────────────────────────────
def draw_metrics_section(fig, gs_slot, models):
    """
    Draw per-case metrics (AUROC, AUPRC) inside gs_slot.
    Layout: n_cases columns × 2 metrics rows (AUROC top, AUPRC bottom).
    """
    n_cases  = len(CASES)
    n_models = len(models)

    gs_inner = GridSpecFromSubplotSpec(
        2, n_cases,
        subplot_spec=gs_slot,
        hspace=0.55,
        wspace=0.14,
    )

    metric_info = [
        ("auroc", "AUROC",  (0.0, 1.05)),
        ("auprc", "AUPRC",  (0.0, 1.05)),
    ]

    bar_positions = np.arange(n_models)
    bar_w = 0.62

    for ci, case in enumerate(CASES):
        df      = load_case(case)
        metrics = compute_metrics(df, models)
        mirna_short = case["mirna"].replace("hsa-", "")

        for mi, (mkey, metric_label, ylim) in enumerate(metric_info):
            ax = fig.add_subplot(gs_inner[mi, ci])

            for bi, (mkey_m, mname, mcol) in enumerate(models):
                val   = metrics[mcol][mkey]
                color = MODEL_COLORS.get(mkey_m, "#888888")
                alpha = 0.92 if mkey_m == "circmac" else 0.78

                if not np.isnan(val):
                    ax.bar(bi, val, width=bar_w, color=color, alpha=alpha,
                           zorder=2, linewidth=0)
                    ax.text(bi, val + 0.02, f"{val:.2f}",
                            ha="center", va="bottom", fontsize=6.5,
                            fontweight="bold", color="#222",
                            bbox=dict(boxstyle="round,pad=0.1",
                                      fc="white", ec="none", alpha=0.85))

            ax.set_xlim(-0.6, n_models - 0.4)
            ax.set_ylim(*ylim)
            ax.set_xticks(bar_positions)
            ax.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
            ax.set_axisbelow(True)
            ax.tick_params(axis="x", length=0)

            # x labels only on bottom metric row
            if mi == 1:
                short_names = [m[1].split("\n")[0] for m in models]
                ax.set_xticklabels(short_names, rotation=30, ha="right", fontsize=7.5)
                # Highlight circMAC label
                for tick, (mkey_m, _, _) in zip(ax.get_xticklabels(), models):
                    if mkey_m == "circmac":
                        tick.set_fontweight("bold")
                        tick.set_color(MODEL_COLORS["circmac"])
            else:
                ax.set_xticklabels([])

            ax.set_ylabel(metric_label, fontsize=8, labelpad=4)

            # Case title on top metric row
            if mi == 0:
                ax.set_title(
                    f"{case['label_m']}\n× {mirna_short}",
                    fontsize=8.5, fontweight="bold", pad=6,
                )


# ── Main ─────────────────────────────────────────────────────────────────────
def make_figure(group_key):
    models   = GROUPS[group_key]
    n_models = len(models)
    n_cases  = len(CASES)

    # Heights:
    #   Heatmap: (1.35 + n_models) * scale_factor inches
    #   Metrics: fixed ~3.8 inches for 2 metric rows
    hm_rows     = n_models + 1
    hm_h        = hm_rows * 0.52 + 0.6
    metrics_h   = 4.2
    fig_w       = 5.8 * n_cases
    fig_h       = hm_h + metrics_h + 0.9

    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = GridSpec(
        2, 1,
        figure=fig,
        height_ratios=[hm_h, metrics_h],
        hspace=0.55,
    )

    draw_heatmap_section(fig, gs[0], models)
    draw_metrics_section(fig, gs[1], models)

    # Section labels (using figure text for consistent vertical placement)
    label_x = 0.01
    fig.text(label_x, 0.995, "(a) Nucleotide-level prediction heatmap",
             ha="left", va="top", fontsize=10, fontweight="bold",
             transform=fig.transFigure)
    fig.text(label_x, 0.48, "(b) Per-case site-level metrics",
             ha="left", va="top", fontsize=10, fontweight="bold",
             transform=fig.transFigure)

    if group_key == "encoder":
        group_label = "General encoder models"
    else:
        group_label = "Pretrained RNA language models (fine-tuned)"

    fig.suptitle(
        f"Case study — {group_label}",
        fontsize=12, fontweight="bold", y=1.012,
    )

    for ext in ["pdf", "png"]:
        p = OUT / f"fig9_{group_key}_heatmap_metrics.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"Saved → {p}")
    plt.close(fig)


def main():
    make_figure("encoder")
    make_figure("pretrained")


if __name__ == "__main__":
    main()
