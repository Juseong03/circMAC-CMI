#!/usr/bin/env python3
"""
fig_multisite.py — Multi-Site Binding Case Study (fig9 style)

Layout (same structure as fig9_heatmap_metrics.py):
  (A) Heatmap: GT row + model rows × 4 cases
  (B) Metrics: F1 / Recall / Precision / AUROC bar charts × 4 cases

Cases: circRPUSD4, circMGA, circDONSON, circFANCA
       (selected for diverse multi-site patterns: 2–3 clusters, varied spread)

Output: fig_multisite.{pdf,png}

Usage:
    python figures_paper/fig_multisite/fig_multisite.py
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
    roc_auc_score, f1_score, recall_score, precision_score,
)

ROOT = Path(__file__).resolve().parents[2]
OUT  = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

DATA_CSV = OUT / "data_predictions.csv"

# ── Cases ─────────────────────────────────────────────────────────────────────
CASES = [
    dict(
        label_m="circFANCA",
        isoform_prefix="chr16|89782859",
        mirna="hsa-miR-6858-5p",
        n_clusters=2,
    ),
    dict(
        label_m="circKHDRBS1",
        isoform_prefix="chr1|32036910,32037835,32038552|32037043",
        mirna="hsa-miR-6880-5p",
        n_clusters=2,
    ),
    dict(
        label_m="circJARID2",
        isoform_prefix="chr6|15410224,15452006,15468542,15487307",
        mirna="hsa-miR-608",
        n_clusters=2,
    ),
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

NAN_COLOR = "#cccccc"   # grey for positions beyond model max_len
BSJ_COLOR = "#1F77B4"

# (color_key, display_label, pred_col)
GROUPS = {
    "encoder": [
        ("lstm",        "LSTM",           "pred_lstm"),
        ("transformer", "Transformer",    "pred_transformer"),
        ("mamba",       "Mamba",          "pred_mamba"),
        ("hymba",       "Hymba",          "pred_hymba"),
        ("circmac",     "CircMAC\n(nopt)","pred_circmac_nopt"),
    ],
    "pretrained": [
        ("rnabert",  "RNABERT",           "pred_rnabert_ft"),
        ("rnaernie", "RNAErnie",          "pred_rnaernie_ft"),
        ("rnamsm",   "RNAMSM",            "pred_rnamsm_ft"),
        ("rnafm",    "RNA-FM",            "pred_rnafm_ft"),
        ("circmac",  "CircMAC\n(nopt)",   "pred_circmac_nopt"),
    ],
}

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        9,
    "axes.linewidth":   0.8,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
})


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_all():
    if not DATA_CSV.exists():
        raise FileNotFoundError(
            f"{DATA_CSV} not found.\n"
            "Run: python scripts/extract_multisite_predictions.py --device 0"
        )
    return pd.read_csv(DATA_CSV)


def load_case(df_all, case):
    mask = (
        df_all["isoform_ID"].str.startswith(case["isoform_prefix"]) &
        (df_all["miRNA_ID"] == case["mirna"])
    )
    df = df_all[mask].sort_values("position").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No rows for {case['label_m']} / {case['mirna']}")
    return df


def compute_metrics(df, models):
    gt = df["ground_truth"].values
    out = {}
    for mkey, mname, mcol in models:
        nan4 = dict(f1=np.nan, recall=np.nan, precision=np.nan, auroc=np.nan)
        if mcol not in df.columns:
            out[mcol] = nan4
            continue
        p    = df[mcol].fillna(0).values
        pred = (p >= 0.5).astype(int)
        try:
            out[mcol] = dict(
                f1        = f1_score(gt, pred, zero_division=0),
                recall    = recall_score(gt, pred, zero_division=0),
                precision = precision_score(gt, pred, zero_division=0),
                auroc     = roc_auc_score(gt, p) if gt.sum() > 0 else np.nan,
            )
        except Exception:
            out[mcol] = nan4
    return out


def draw_bsj(ax, L, lw=1.0, alpha=0.55):
    ax.axvline(-0.5,     color=BSJ_COLOR, lw=lw, ls="--", alpha=alpha)
    ax.axvline(L - 0.5,  color=BSJ_COLOR, lw=lw, ls="--", alpha=alpha)


def clean_spines(ax):
    for sp in ax.spines.values():
        sp.set_visible(False)


# ── (A) Heatmap section ───────────────────────────────────────────────────────
def draw_heatmap_section(fig, gs_slot, df_all, models):
    n_models = len(models)
    n_cases  = len(CASES)

    gs_inner = GridSpecFromSubplotSpec(
        1, n_cases, subplot_spec=gs_slot, wspace=0.14,
    )

    for ci, case in enumerate(CASES):
        df = load_case(df_all, case)
        gt = df["ground_truth"].values
        L  = len(df)

        # count clusters for subtitle
        diff = np.diff(np.concatenate([[0], gt, [0]]))
        n_cl = int((diff == 1).sum())
        n_si = int(gt.sum())

        gs_case = GridSpecFromSubplotSpec(
            n_models + 1, 1,
            subplot_spec=gs_inner[ci],
            height_ratios=[1.35] + [1.0] * n_models,
            hspace=0.08,
        )

        # Ground-truth row
        ax_gt = fig.add_subplot(gs_case[0])
        ax_gt.imshow(gt[np.newaxis, :], aspect="auto",
                     cmap="Reds", vmin=0, vmax=1, interpolation="nearest")
        ax_gt.set_yticks([])
        ax_gt.set_xticks([])
        draw_bsj(ax_gt, L, lw=1.4, alpha=0.75)
        clean_spines(ax_gt)

        if ci == 0:
            ax_gt.set_ylabel("Ground\ntruth", rotation=0, ha="right", va="center",
                             fontsize=8, fontweight="bold", labelpad=10)

        mirna_short = case["mirna"].replace("hsa-", "")
        ax_gt.set_title(
            f"{case['label_m']} × {mirna_short}\n"
            f"L={L},  {n_si} sites,  {n_cl} clusters",
            fontsize=9, fontweight="bold", pad=5,
        )

        # Model rows
        for ri, (mkey, mname, mcol) in enumerate(models):
            ax = fig.add_subplot(gs_case[ri + 1])

            if mcol in df.columns and not df[mcol].isna().all():
                pred_raw = df[mcol].values.astype(float)
                pred_masked = np.ma.masked_invalid(pred_raw)
                color = MODEL_COLORS.get(mkey, "#888888")
                cmap  = mcolors.LinearSegmentedColormap.from_list(
                    f"{mkey}_cm", ["#f7f7f7", color]
                )
                cmap.set_bad(color=NAN_COLOR)
                ax.imshow(pred_masked[np.newaxis, :], aspect="auto",
                          cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
            else:
                ax.set_facecolor("#eeeeee")

            ax.set_yticks([])
            ax.set_xticks([])
            draw_bsj(ax, L, lw=0.9, alpha=0.5)
            clean_spines(ax)

            if ci == 0:
                ax.set_ylabel(mname, rotation=0, ha="right", va="center",
                              fontsize=8, fontweight="bold",
                              color=MODEL_COLORS.get(mkey, "#222222"),
                              labelpad=10)

            if ri == n_models - 1:
                xticks = np.linspace(0, L - 1, 5, dtype=int)
                ax.set_xticks(xticks)
                ax.set_xticklabels([str(x) for x in xticks], fontsize=7)
                ax.set_xlabel("Sequence position", fontsize=7.5, labelpad=3)
                ax.tick_params(axis="x", colors="black")


# ── (B) Metrics section ───────────────────────────────────────────────────────
def draw_metrics_section(fig, gs_slot, df_all, models):
    n_cases  = len(CASES)
    n_models = len(models)

    gs_inner = GridSpecFromSubplotSpec(
        4, n_cases, subplot_spec=gs_slot, hspace=0.50, wspace=0.14,
    )

    metric_info = [
        ("f1",        "F1",        (0.0, 1.12)),
        ("recall",    "Recall",    (0.0, 1.12)),
        ("precision", "Precision", (0.0, 1.12)),
        ("auroc",     "AUROC",     (0.0, 1.12)),
    ]

    bar_positions = np.arange(n_models)
    bar_w = 0.62

    for ci, case in enumerate(CASES):
        df      = load_case(df_all, case)
        metrics = compute_metrics(df, models)
        mirna_short = case["mirna"].replace("hsa-", "")

        for mi, (metric_key, metric_label, ylim) in enumerate(metric_info):
            ax = fig.add_subplot(gs_inner[mi, ci])

            for bi, (mkey, mname, mcol) in enumerate(models):
                val   = metrics[mcol][metric_key]
                color = MODEL_COLORS.get(mkey, "#888888")
                alpha = 0.92 if mkey == "circmac" else 0.78

                if not np.isnan(val):
                    ax.bar(bi, val, width=bar_w, color=color,
                           alpha=alpha, zorder=2, linewidth=0)
                    ax.text(bi, val + 0.025, f"{val:.2f}",
                            ha="center", va="bottom",
                            fontsize=6, fontweight="bold", color="#222222",
                            bbox=dict(boxstyle="round,pad=0.1",
                                      fc="white", ec="none", alpha=0.85))

            ax.set_xlim(-0.6, n_models - 0.4)
            ax.set_ylim(*ylim)
            ax.set_xticks(bar_positions)
            ax.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
            ax.set_axisbelow(True)
            ax.tick_params(axis="x", length=0, colors="black")
            ax.tick_params(axis="y", colors="black")

            if mi == len(metric_info) - 1:
                short_names = [m[1].split("\n")[0] for m in models]
                ax.set_xticklabels(short_names, rotation=30, ha="right",
                                   fontsize=7.5)
                for tick, (mkey, _, _) in zip(ax.get_xticklabels(), models):
                    if mkey == "circmac":
                        tick.set_fontweight("bold")
            else:
                ax.set_xticklabels([])

            ax.set_ylabel(metric_label, fontsize=10, fontweight="bold",
                          labelpad=4)

            if mi == 0:
                ax.set_title(f"{case['label_m']} × {mirna_short}",
                             fontsize=10, fontweight="bold", pad=6)


# ── Main ──────────────────────────────────────────────────────────────────────
def make_figure(group_key, df_all):
    models   = GROUPS[group_key]
    n_models = len(models)
    n_cases  = len(CASES)

    hm_rows   = n_models + 1
    hm_h      = hm_rows * 0.80 + 0.8
    metrics_h = 7.5 if n_models > 6 else 6.0
    fig_w     = 6.2 * n_cases
    fig_h     = hm_h + metrics_h + 1.0

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = GridSpec(2, 1, figure=fig,
                   height_ratios=[hm_h, metrics_h], hspace=0.50)

    draw_heatmap_section(fig, gs[0], df_all, models)
    draw_metrics_section(fig, gs[1], df_all, models)

    panel_x = 0.06
    hm_frac  = hm_h / fig_h
    fig.text(panel_x, 0.985, "(A)", ha="left", va="top",
             fontsize=13, fontweight="bold", transform=fig.transFigure)
    fig.text(panel_x, hm_frac - 0.01, "(B)", ha="left", va="top",
             fontsize=13, fontweight="bold", transform=fig.transFigure)

    if group_key == "encoder":
        title = "Case study — Multi-site binding (CircMAC vs general encoder models)"
    else:
        title = "Case study — Multi-site binding (CircMAC vs RNA language models)"
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.012)

    for ext in ["pdf", "png"]:
        p = OUT / f"fig_multisite_{group_key}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"Saved → {p}")
    plt.close(fig)


def main():
    df_all = load_all()
    make_figure("encoder",    df_all)
    make_figure("pretrained", df_all)
    print("Done.")


if __name__ == "__main__":
    main()
