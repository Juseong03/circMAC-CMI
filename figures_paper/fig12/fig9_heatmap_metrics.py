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
    roc_curve,
    f1_score,
    recall_score,
    precision_score,
)


# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)


# ── Case definitions ─────────────────────────────────────────────────────────
# Main text cases: 3 cases where circMAC clearly outperforms pretrained RNA LMs
MAIN_CASES = [
    dict(
        label_m="circCDYL2",
        csv=ROOT / "figures_paper/Fig_Case_CDYL2/data_predictions.csv",
        isoform="chr4|84678168,84679116|84678259,84679242|-",
        mirna="hsa-miR-34c-5p",
    ),
    dict(
        label_m="circMAPK1",
        csv=ROOT / "figures_paper/Fig_Case_MAPK1/data_predictions.csv",
        isoform="chr22|21799012,21805850,21807664|21799128,21806039,21807846|-",
        mirna="hsa-miR-12119",
    ),
    dict(
        label_m="circHUWE1",
        csv=ROOT / "figures_paper/Fig_Case_HUWE1/data_predictions.csv",
        isoform="chrX|53645311,53647368,53648212,53648573,53654063|53645463,53647574,53648310,53649000,53654131|-",
        mirna="hsa-miR-29b-3p",
    ),
]

# Supplementary cases: 3 additional cases
SUPP_CASES = [
    dict(
        label_m="circ17q21.31",
        csv=ROOT / "figures_paper/Fig_Case_PT_CHR17/data_predictions.csv",
        isoform="chr17|63193991,63200771|63194139,63200957|+",
        mirna="hsa-miR-4732-5p",
    ),
    dict(
        label_m="circAPP",
        csv=ROOT / "figures_paper/Fig_Case_APP/data_predictions.csv",
        isoform="chr21|25954590,25955627,25975070,25975954,25982344|25954689,25955755,25975228,25976028,25982477|-",
        mirna="hsa-miR-5001-3p",
    ),
    dict(
        label_m="circNFIB",
        csv=ROOT / "figures_paper/Fig_Case_NFIB/data_predictions.csv",
        isoform="chr9|14120440,14125632,14146689,14150145,14155825,14179727|14120624,14125766,14146807,14150265,14155893,14179780|-",
        mirna="hsa-miR-373-3p",
    ),
]


# ── Model groups ─────────────────────────────────────────────────────────────
MODEL_COLORS = {
    "circmac":         "#FF7F0E",
    "lstm":            "#E377C2",
    "transformer":     "#8C564B",
    "mamba":           "#D62728",
    "hymba":           "#BCBD22",
    # Frozen: lighter/desaturated tones
    "rnabert_frozen":  "#A8C8E8",
    "rnaernie_frozen": "#C9B8E8",
    "rnamsm_frozen":   "#A8D8A8",
    "rnafm_frozen":    "#A8E8E8",
    # Fine-tuned: original saturated tones
    "rnabert_ft":      "#1F77B4",
    "rnaernie_ft":     "#9467BD",
    "rnamsm_ft":       "#2CA02C",
    "rnafm_ft":        "#17BECF",
}

BSJ_COLOR = "#1F77B4"

# (color_key, display_label, pred_col)
GROUPS = {
    "encoder": [
        ("lstm",        "LSTM",        "pred_lstm"),
        ("transformer", "Transformer", "pred_transformer"),
        ("mamba",       "Mamba",       "pred_mamba"),
        ("hymba",       "Hymba",       "pred_hymba"),
        ("circmac",     "circMAC",     "pred_circmac"),
    ],
    # Frozen backbone only
    "pretrained_frozen": [
        ("rnabert_frozen",  "RNABERT\n(frozen)",   "pred_rnabert_frozen"),
        ("rnaernie_frozen", "RNAErnie\n(frozen)",  "pred_rnaernie_frozen"),
        ("rnamsm_frozen",   "RNA-MSM\n(frozen)",    "pred_rnamsm_frozen"),
        ("rnafm_frozen",    "RNA-FM\n(frozen)",    "pred_rnafm_frozen"),
        ("circmac",         "circMAC",             "pred_circmac"),
    ],
    # Full fine-tuning
    "pretrained_ft": [
        ("rnabert_ft",  "RNABERT\n(fine-tuned)",  "pred_rnabert_ft"),
        ("rnaernie_ft", "RNAErnie\n(fine-tuned)", "pred_rnaernie_ft"),
        ("rnamsm_ft",   "RNA-MSM\n(fine-tuned)",   "pred_rnamsm_ft"),
        ("rnafm_ft",    "RNA-FM\n(fine-tuned)",   "pred_rnafm_ft"),
        ("circmac",     "circMAC",                "pred_circmac"),
    ],
    # Legacy alias
    "pretrained": [
        ("rnabert_ft",  "RNABERT\n(fine-tuned)",  "pred_rnabert_ft"),
        ("rnaernie_ft", "RNAErnie\n(fine-tuned)", "pred_rnaernie_ft"),
        ("rnamsm_ft",   "RNA-MSM\n(fine-tuned)",   "pred_rnamsm_ft"),
        ("rnafm_ft",    "RNA-FM\n(fine-tuned)",   "pred_rnafm_ft"),
        ("circmac",     "circMAC",                "pred_circmac"),
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


def _cases_for(cases):
    """Helper to iterate over a case list."""
    return cases


def optimal_threshold(gt, prob):
    """F1-optimal threshold from precision-recall curve."""
    from sklearn.metrics import precision_recall_curve
    prec, rec, thresholds = precision_recall_curve(gt, prob)
    # thresholds has len = len(prec) - 1
    f1s = np.where((prec[:-1] + rec[:-1]) > 0,
                   2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1]),
                   0.0)
    idx = np.argmax(f1s)
    return float(thresholds[idx])


def compute_metrics(df, models):
    """
    Return dict {pred_col: {f1, recall, precision, auroc}} using
    the optimal (Youden-index) threshold per model.
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
        if pd.isna(p).all():
            out[mcol] = nan4
            continue

        try:
            auroc = roc_auc_score(gt, p) if gt.sum() > 0 else float("nan")
            thresh = optimal_threshold(gt, p)
            pred = (p >= thresh).astype(int)
            f1  = f1_score(gt, pred, zero_division=0)
            rec = recall_score(gt, pred, zero_division=0)
            pre = precision_score(gt, pred, zero_division=0)
        except Exception:
            f1, rec, pre, auroc = (float("nan"),) * 4

        out[mcol] = dict(f1=f1, recall=rec, precision=pre, auroc=auroc)

    return out


def draw_bsj(ax, L, lw=1.0, alpha=0.55):
    ax.axvline(-0.5, color=BSJ_COLOR, lw=lw, ls="--", alpha=alpha)
    ax.axvline(L - 0.5, color=BSJ_COLOR, lw=lw, ls="--", alpha=alpha)


def clean_spines(ax):
    for sp in ax.spines.values():
        sp.set_visible(False)


# ── Heatmap section ──────────────────────────────────────────────────────────
def draw_heatmap_section(fig, gs_slot, models, cases):
    """
    Draw (GT + model rows) × n_cases inside gs_slot.
    gs_slot is a GridSpec row of n_cases columns.
    """
    n_models = len(models)
    n_cases = len(cases)

    gs_inner = GridSpecFromSubplotSpec(
        1,
        n_cases,
        subplot_spec=gs_slot,
        wspace=0.14,
    )

    for ci, case in enumerate(cases):
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

            if mcol in df.columns and not df[mcol].isna().all():
                pred = df[mcol].fillna(0).values
                color = MODEL_COLORS.get(mkey, "#888888")

                cmap = mcolors.LinearSegmentedColormap.from_list(
                    f"{mkey}_cm",
                    ["#f7f7f7", color],
                )

                # Normalize each model's predictions by its own max for visual clarity
                # (circMAC tends to use a lower probability scale)
                pred_max = pred.max()
                vmax_val = max(pred_max, 0.05)  # at least 0.05 to avoid division issues

                ax.imshow(
                    pred[np.newaxis, :],
                    aspect="auto",
                    cmap=cmap,
                    vmin=0,
                    vmax=vmax_val,
                    interpolation="nearest",
                )
            else:
                ax.set_facecolor("#eeeeee")
                ax.text(0.5, 0.5, "N/A (seq too long)",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=7, color="#888888", style="italic")

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
def draw_metrics_section(fig, gs_slot, models, cases):
    """
    Draw per-case metrics (F1 / Recall / Precision / AUROC) inside gs_slot.
    Layout: 4 metric rows × n_cases columns.
    X-axis labels shown only on the bottom row.
    """
    n_cases = len(cases)
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

    for ci, case in enumerate(cases):
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
def make_figure(group_key, cases, out_stem):
    models = GROUPS[group_key]
    n_models = len(models)
    n_cases = len(cases)

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

    draw_heatmap_section(fig, gs[0], models, cases)
    draw_metrics_section(fig, gs[1], models, cases)

    panel_x = 0.01
    hm_frac = hm_h / fig_h
    b_y = hm_frac - 0.04

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

    # fig.suptitle(
    #     f"Case study — {group_label}",
    #     fontsize=12,
    #     fontweight="bold",
    #     y=1.012,
    #     color="black",
    # )

    for ext in ["pdf", "png", "eps"]:
        p = OUT / f"{out_stem}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"Saved → {p}")

    plt.close(fig)


def main():
    make_figure("encoder",           MAIN_CASES, "fig9_encoder_main")
    make_figure("encoder",           SUPP_CASES, "fig9_encoder_supp")
    make_figure("pretrained_frozen", MAIN_CASES, "fig9_pretrained_frozen_main")
    make_figure("pretrained_frozen", SUPP_CASES, "fig9_pretrained_frozen_supp")
    make_figure("pretrained_ft",     MAIN_CASES, "fig9_pretrained_ft_main")
    make_figure("pretrained_ft",     SUPP_CASES, "fig9_pretrained_ft_supp")


if __name__ == "__main__":
    main()