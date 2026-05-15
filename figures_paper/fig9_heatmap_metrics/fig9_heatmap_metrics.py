#!/usr/bin/env python3
"""
Fig 9 — Heatmap + Per-case Metrics (F1 / Recall / Precision / AUROC)

Two subplots per figure:
  (a) Nucleotide-level prediction heatmap  (GT row + model rows, BSJ lines)
  (b) Per-case bar charts: F1, Recall, Precision, AUROC  (3 cases × model bars)

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
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    recall_score,
    precision_score,
)


# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)


# ── Case definitions ─────────────────────────────────────────────────────────
CASES = [
    dict(
        label="circCDYL2\nx\n",
        label_m="circCDYL2",
        csv=ROOT / "figures_paper/Fig_Case_CDYL2/data_predictions.csv",
        isoform="chr4|84678168,84679116|84678259,84679242|-",
        mirna="hsa-miR-449a",
    ),
    dict(
        label="circMAPK1\nx\n",
        label_m="circMAPK1",
        csv=ROOT / "figures_paper/Fig_Case_MAPK1/data_predictions.csv",
        isoform="chr22|21799012,21805850,21807664|21799128,21806039,21807846|-",
        mirna="hsa-miR-12119",
    ),
    dict(
        label="circAPP\nx\n",
        label_m="circAPP",
        csv=ROOT / "figures_paper/Fig_Case_APP/data_predictions.csv",
        isoform="chr21|25954590,25955627,25975070,25975954,25982344|25954689,25955755,25975228,25976028,25982477|-",
        mirna="hsa-miR-5001-3p",
    ),
]


# ── Model groups ─────────────────────────────────────────────────────────────
MODEL_COLORS = {
    "circmac": "#FF7F0E",
    "lstm": "#E377C2",
    "transformer": "#8C564B",
    "mamba": "#D62728",
    "hymba": "#BCBD22",
    "rnabert": "#1F77B4",
    "rnaernie": "#9467BD",
    "rnamsm": "#2CA02C",
    "rnafm": "#17BECF",
}

BSJ_COLOR = "#1F77B4"

# (color_key, display_label, pred_col)
GROUPS = {
    "encoder": [
        ("lstm", "LSTM", "pred_lstm"),
        ("transformer", "Transformer", "pred_transformer"),
        ("mamba", "Mamba", "pred_mamba"),
        ("hymba", "Hymba", "pred_hymba"),
        ("circmac", "circMAC", "pred_circmac"),
    ],
    "pretrained": [
        ("rnabert", "RNABERT\n(fine-tuned)", "pred_rnabert"),
        ("rnaernie", "RNAErnie\n(fine-tuned)", "pred_rnaernie"),
        ("rnamsm", "RNAMSM\n(fine-tuned)", "pred_rnamsm"),
        ("rnafm", "RNA-FM\n(fine-tuned)", "pred_rnafm"),
        ("circmac", "circMAC", "pred_circmac"),
    ],
}


# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ── Helpers ──────────────────────────────────────────────────────────────────
def load_case(case):
    df_all = pd.read_csv(case["csv"])

    df = df_all[
        (df_all["miRNA_ID"] == case["mirna"]) &
        (df_all["isoform_ID"] == case["isoform"])
    ].reset_index(drop=True)

    if df.empty:
        raise ValueError(f"No rows for {case['label_m']} / {case['mirna']}")

    return df


def compute_metrics(df, models):
    """
    Return dict {pred_col: {f1, recall, precision, auroc}} at threshold=0.5.
    """
    gt = df["ground_truth"].values
    out = {}

    for mkey, mname, mcol in models:
        nan4 = dict(
            f1=float("nan"),
            recall=float("nan"),
            precision=float("nan"),
            auroc=float("nan"),
        )

        if mcol not in df.columns:
            out[mcol] = nan4
            continue

        p = df[mcol].values
        pred = (p >= 0.5).astype(int)

        try:
            f1 = f1_score(gt, pred, zero_division=0)
            rec = recall_score(gt, pred, zero_division=0)
            pre = precision_score(gt, pred, zero_division=0)
            auroc = roc_auc_score(gt, p) if gt.sum() > 0 else float("nan")
        except Exception:
            f1, rec, pre, auroc = (float("nan"),) * 4

        out[mcol] = dict(
            f1=f1,
            recall=rec,
            precision=pre,
            auroc=auroc,
        )

    return out


def draw_bsj(ax, L, lw=1.0, alpha=0.55):
    ax.axvline(-0.5, color=BSJ_COLOR, lw=lw, ls="--", alpha=alpha)
    ax.axvline(L - 0.5, color=BSJ_COLOR, lw=lw, ls="--", alpha=alpha)


def clean_spines(ax):
    for sp in ax.spines.values():
        sp.set_visible(False)


# ── Heatmap section ──────────────────────────────────────────────────────────
def draw_heatmap_section(fig, gs_slot, models):
    """
    Draw (GT + model rows) × n_cases inside gs_slot.
    gs_slot is a GridSpec row of n_cases columns.
    """
    n_models = len(models)
    n_cases = len(CASES)

    gs_inner = GridSpecFromSubplotSpec(
        1,
        n_cases,
        subplot_spec=gs_slot,
        wspace=0.14,
    )

    for ci, case in enumerate(CASES):
        df = load_case(case)
        gt = df["ground_truth"].values
        L = len(df)

        gs_case = GridSpecFromSubplotSpec(
            n_models + 1,
            1,
            subplot_spec=gs_inner[ci],
            height_ratios=[1.35] + [1.0] * n_models,
            hspace=0.08,
        )

        # Ground-truth row
        ax_gt = fig.add_subplot(gs_case[0])
        ax_gt.imshow(
            gt[np.newaxis, :],
            aspect="auto",
            cmap="Reds",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )

        ax_gt.set_yticks([])
        ax_gt.set_xticks([])
        draw_bsj(ax_gt, L, lw=1.4, alpha=0.75)
        clean_spines(ax_gt)

        if ci == 0:
            ax_gt.set_ylabel(
                "Ground truth",
                rotation=0,
                ha="right",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="black",
                labelpad=10,
            )

        mirna_short = case["mirna"].replace("hsa-", "")

        ax_gt.set_title(
            f"{case['label_m']}×{mirna_short}",
            fontsize=10,
            fontweight="bold",
            pad=6,
            color="black",
        )

        # Model rows
        for ri, (mkey, mname, mcol) in enumerate(models):
            ax = fig.add_subplot(gs_case[ri + 1])

            if mcol in df.columns:
                pred = df[mcol].values
                color = MODEL_COLORS.get(mkey, "#888888")

                cmap = mcolors.LinearSegmentedColormap.from_list(
                    f"{mkey}_cm",
                    ["#f7f7f7", color],
                )

                ax.imshow(
                    pred[np.newaxis, :],
                    aspect="auto",
                    cmap=cmap,
                    vmin=0,
                    vmax=1,
                    interpolation="nearest",
                )
            else:
                ax.set_facecolor("#eeeeee")

            ax.set_yticks([])
            ax.set_xticks([])
            draw_bsj(ax, L, lw=0.9, alpha=0.5)
            clean_spines(ax)

            if ci == 0:
                ax.set_ylabel(
                    mname,
                    rotation=0,
                    ha="right",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color="black",
                    labelpad=10,
                )

            # x-axis on bottom row
            if ri == n_models - 1:
                xticks = np.linspace(0, L - 1, 5, dtype=int)
                ax.set_xticks(xticks)
                ax.set_xticklabels(
                    [str(x) for x in xticks],
                    fontsize=7,
                    color="black",
                )
                ax.set_xlabel(
                    "Sequence position",
                    fontsize=7.5,
                    labelpad=3,
                    color="black",
                )
                ax.tick_params(axis="x", colors="black")


# ── Metrics bar chart section ────────────────────────────────────────────────
def draw_metrics_section(fig, gs_slot, models):
    """
    Draw per-case metrics (F1 / Recall / Precision / AUROC) inside gs_slot.
    Layout: 4 metric rows × n_cases columns.
    X-axis labels shown only on the bottom row.
    """
    n_cases = len(CASES)
    n_models = len(models)

    gs_inner = GridSpecFromSubplotSpec(
        4,
        n_cases,
        subplot_spec=gs_slot,
        hspace=0.50,
        wspace=0.14,
    )

    metric_info = [
        ("f1", "F1", (0.0, 1.12)),
        ("recall", "Recall", (0.0, 1.12)),
        ("precision", "Precision", (0.0, 1.12)),
        ("auroc", "AUROC", (0.0, 1.12)),
    ]

    n_metrics = len(metric_info)
    bar_positions = np.arange(n_models)
    bar_w = 0.62

    for ci, case in enumerate(CASES):
        df = load_case(case)
        metrics = compute_metrics(df, models)
        mirna_short = case["mirna"].replace("hsa-", "")

        for mi, (metric_key, metric_label, ylim) in enumerate(metric_info):
            ax = fig.add_subplot(gs_inner[mi, ci])

            for bi, (mkey_m, mname, mcol) in enumerate(models):
                val = metrics[mcol][metric_key]
                color = MODEL_COLORS.get(mkey_m, "#888888")
                alpha = 0.92 if mkey_m == "circmac" else 0.78

                if not np.isnan(val):
                    ax.bar(
                        bi,
                        val,
                        width=bar_w,
                        color=color,
                        alpha=alpha,
                        zorder=2,
                        linewidth=0,
                    )

                    ax.text(
                        bi,
                        val + 0.025,
                        f"{val:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=6,
                        fontweight="bold",
                        color="#222222",
                        bbox=dict(
                            boxstyle="round,pad=0.1",
                            fc="white",
                            ec="none",
                            alpha=0.85,
                        ),
                    )

            ax.set_xlim(-0.6, n_models - 0.4)
            ax.set_ylim(*ylim)
            ax.set_xticks(bar_positions)

            ax.yaxis.grid(
                True,
                linestyle="--",
                alpha=0.35,
                zorder=0,
            )
            ax.set_axisbelow(True)

            ax.tick_params(axis="x", length=0, colors="black")
            ax.tick_params(axis="y", colors="black")

            # x-tick labels only on the bottom metric row
            if mi == n_metrics - 1:
                short_names = [m[1].split("\n")[0] for m in models]

                ax.set_xticklabels(
                    short_names,
                    rotation=30,
                    ha="right",
                    fontsize=7.5,
                    color="black",
                )

                for tick, (mkey_m, _, _) in zip(ax.get_xticklabels(), models):
                    if mkey_m == "circmac":
                        tick.set_fontweight("bold")
                        tick.set_color("black")
            else:
                ax.set_xticklabels([])

            ax.set_ylabel(
                metric_label,
                fontsize=10,
                fontweight='bold',
                labelpad=4,
                color="black",
            )

            # Case title on the top metric row only
            if mi == 0:
                ax.set_title(
                    f"{case['label_m']} × {mirna_short}",
                    fontsize=10,
                    fontweight="bold",
                    pad=6,
                    color="black",
                )


# ── Main ─────────────────────────────────────────────────────────────────────
def make_figure(group_key):
    models = GROUPS[group_key]
    n_models = len(models)
    n_cases = len(CASES)

    # Heights:
    #   Heatmap: larger rows for readability
    #   Metrics: fixed height for 4 metric rows
    hm_rows = n_models + 1
    hm_h = hm_rows * 0.80 + 0.8
    metrics_h = 6.0
    fig_w = 5.8 * n_cases
    fig_h = hm_h + metrics_h + 1.0

    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[hm_h, metrics_h],
        hspace=0.50,
    )

    draw_heatmap_section(fig, gs[0], models)
    draw_metrics_section(fig, gs[1], models)

    # Panel labels — position computed from height ratios
    panel_x = 0.01
    hm_frac = hm_h / fig_h
    b_y = hm_frac - 0.04   # just below heatmap section

    fig.text(
        panel_x+0.05, 0.985,
        "(A)",
        ha="left", va="top",
        fontsize=13, fontweight="bold",
        color="black", transform=fig.transFigure,
    )

    fig.text(
        panel_x+0.05, b_y+0.1,
        "(B)",
        ha="left", va="top",
        fontsize=13, fontweight="bold",
        color="black", transform=fig.transFigure,
    )

    if group_key == "encoder":
        group_label = "General encoder models"
    else:
        group_label = "Pretrained RNA language models (fine-tuned)"

    fig.suptitle(
        f"Case study — {group_label}",
        fontsize=12,
        fontweight="bold",
        y=1.012,
        color="black",
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