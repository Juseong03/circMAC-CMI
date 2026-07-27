#!/usr/bin/env python3
"""
fig_multisite.py — Multi-Site Binding Case Study (fig9 style)

Layout (same structure as fig9_heatmap_metrics.py):
  (A) Heatmap: GT row + model rows × 3 cases
  (B) Metrics: F1 / Recall / Precision / AUROC bar charts × 3 cases

Cases: circFANCA, circKHDRBS1, circJARID2
       (selected for diverse multi-site patterns)

Output: fig_multisite_{encoder,pretrained}.{pdf,png,eps}

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
    roc_auc_score,
    f1_score,
    recall_score,
    precision_score,
    precision_recall_curve,
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
        ("circmac",     "circMAC",        "pred_circmac_nopt"),
    ],
    "pretrained": [
        ("rnabert",  "RNABERT",           "pred_rnabert_ft"),
        ("rnaernie", "RNAErnie",          "pred_rnaernie_ft"),
        ("rnamsm",   "RNAMSM",            "pred_rnamsm_ft"),
        ("rnafm",    "RNA-FM",            "pred_rnafm_ft"),
        ("circmac",  "circMAC",   "pred_circmac_nopt"),
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


def optimal_threshold(gt, prob):
    """Return the F1-optimal threshold, matching the Fig. 9 code."""
    prec, rec, thresholds = precision_recall_curve(gt, prob)

    if len(thresholds) == 0:
        return 0.5

    denom = prec[:-1] + rec[:-1]
    f1s = np.divide(
        2 * prec[:-1] * rec[:-1],
        denom,
        out=np.zeros_like(denom, dtype=float),
        where=denom > 0,
    )
    return float(thresholds[int(np.argmax(f1s))])


def compute_metrics(df, models):
    """
    Compute per-case metrics.

    Important
    ---------
    * A model is treated as unavailable when its prediction column is
      missing or contains any NaN values due to an unsupported sequence length.
    * Unavailable model-case combinations remain NaN internally; Panel B
      displays those NaNs as 0.00 only.
    * Valid predictions are NEVER replaced by zero.
    * For valid predictions, use the same F1-optimal threshold as Fig. 9.
    """
    gt = pd.to_numeric(df["ground_truth"], errors="coerce").to_numpy(dtype=float)
    gt_valid = np.isfinite(gt)

    out = {}

    for mkey, mname, mcol in models:
        nan4 = dict(
            f1=np.nan,
            recall=np.nan,
            precision=np.nan,
            auroc=np.nan,
            threshold=np.nan,
            unavailable=True,
        )

        if mcol not in df.columns:
            out[mcol] = nan4
            continue

        p = pd.to_numeric(df[mcol], errors="coerce").to_numpy(dtype=float)

        # Length mismatch / unavailable prediction:
        # do NOT convert these NaNs to prediction probability 0.
        # If any NaN is present, do not compute the metric.
        # This model-case pair will be displayed as 0.00 in Panel B only.
        if np.isnan(p).any():
            out[mcol] = nan4
            continue

        valid = gt_valid & np.isfinite(p)
        if valid.sum() == 0:
            out[mcol] = nan4
            continue

        gt_v = gt[valid].astype(int)
        p_v = p[valid]

        try:
            thresh = optimal_threshold(gt_v, p_v)
            pred = (p_v >= thresh).astype(int)

            auroc = (
                roc_auc_score(gt_v, p_v)
                if np.unique(gt_v).size == 2
                else np.nan
            )

            out[mcol] = dict(
                f1=f1_score(gt_v, pred, zero_division=0),
                recall=recall_score(gt_v, pred, zero_division=0),
                precision=precision_score(gt_v, pred, zero_division=0),
                auroc=auroc,
                threshold=thresh,
                unavailable=False,
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

            pred_series = (
                pd.to_numeric(df[mcol], errors="coerce")
                if mcol in df.columns
                else None
            )

            # IMPORTANT:
            # If any NaN exists, treat the whole model-case combination as unavailable
            # and render the row exactly as "N/A (seq too long)".
            is_unavailable = (
                pred_series is None
                or pred_series.isna().any()
            )

            if not is_unavailable:
                pred = pred_series.to_numpy(dtype=float)

                color = MODEL_COLORS.get(mkey, "#888888")
                cmap = mcolors.LinearSegmentedColormap.from_list(
                    f"{mkey}_{ci}_{ri}_cm",
                    ["#f7f7f7", color],
                )

                # Same visualization rule as Fig. 9:
                # normalize each model by its own maximum for visual clarity.
                pred_max = float(np.max(pred))
                vmax_val = max(pred_max, 0.05)

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
                ax.text(
                    0.5, 0.5,
                    "N/A (seq too long)",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="#888888",
                    style="italic",
                )

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

        print(f"\n[{case['label_m']} × {mirna_short}]")
        for _mkey, _mname, _mcol in models:
            mm = metrics[_mcol]
            if mm["unavailable"]:
                print(f"  {_mname:12s}: N/A (seq too long) -> displayed as 0.00")
            else:
                print(
                    f"  {_mname:12s}: threshold={mm['threshold']:.4f}, "
                    f"F1={mm['f1']:.3f}, Recall={mm['recall']:.3f}, "
                    f"Precision={mm['precision']:.3f}, AUROC={mm['auroc']:.3f}"
                )

        for mi, (metric_key, metric_label, ylim) in enumerate(metric_info):
            ax = fig.add_subplot(gs_inner[mi, ci])

            for bi, (mkey, mname, mcol) in enumerate(models):
                val   = metrics[mcol][metric_key]
                color = MODEL_COLORS.get(mkey, "#888888")
                alpha = 0.78

                if not np.isnan(val):
                    # Valid prediction: plot the actual metric.
                    ax.bar(
                        bi, val,
                        width=bar_w,
                        color=color,
                        alpha=alpha,
                        zorder=2,
                        linewidth=0,
                    )
                    ax.text(
                        bi, min(val + 0.025, 1.075),
                        f"{val:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=6,
                        fontweight="bold",
                        color="#222222",
                    )
                else:
                    # Only genuinely unavailable/NaN model-case combinations
                    # are displayed as zero. Keep NaN internally so that we do
                    # not confuse "not evaluable" with a true metric of zero.
                    ax.bar(
                        bi, 0.0,
                        width=bar_w,
                        color=color,
                        alpha=alpha,
                        zorder=2,
                        linewidth=0,
                    )
                    ax.text(
                        bi, 0.025,
                        "0.00",
                        ha="center",
                        va="bottom",
                        fontsize=6,
                        fontweight="bold",
                        color="#777777",
                    )

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
                # Keep all model labels visually consistent.
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

    for ext in ["pdf", "png", "eps"]:
        p = OUT / f"fig_multisite_{group_key}.{ext}"

        fig.savefig(
            p,
            dpi=200,
            bbox_inches="tight",
            facecolor="white"
        )

        print(f"Saved → {p}")

    plt.close(fig)


def main():
    df_all = load_all()
    make_figure("encoder",    df_all)
    make_figure("pretrained", df_all)
    print("Done.")


if __name__ == "__main__":
    main()