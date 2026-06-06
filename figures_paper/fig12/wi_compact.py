#!/usr/bin/env python3
"""
Fig 7 — w/ miRNA only.

This script saves:
  1) all-case combined figures
  2) separate figures for each case

Combined output:
  fig7_wi_{umap|tsne}_{encoder|pretrained}_minimal_tight_case_aligned.{pdf,png}

Case-wise output:
  fig7_wi_{umap|tsne}_{encoder|pretrained}_{case_key}_case.{pdf,png}
"""

from pathlib import Path
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score


# =============================================================================
# Paths
# =============================================================================
ROOT  = Path(__file__).resolve().parents[2]
CACHE = ROOT / "figures_paper" / "embedding_cache"
OUT   = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Options
# =============================================================================
SAVE_COMBINED_FIGS = True
SAVE_CASE_FIGS     = True


# =============================================================================
# Cases / Models
# =============================================================================
# Main text cases (pretrained comparison)
MAIN_CASES = [
    ("circCDYL2\n×\nmiR-34c-5p",  "cdyl2"),
    ("circMAPK1\n×\nmiR-12119",   "mapk1"),
    ("circHUWE1\n×\nmiR-29b-3p",  "huwe1"),
]

# Supplementary cases
SUPP_CASES = [
    ("circ17q21.31\n×\nmiR-4732-5p", "pt_chr17"),
    ("circAPP\n×\nmiR-5001-3p",      "app"),
    ("circNFIB\n×\nmiR-373-3p",      "nfib"),
]

# Combined for legacy use
CASES = MAIN_CASES

GROUPS = {
    "encoder": [
        ("LSTM",        "LSTM"),
        ("Transformer", "Transformer"),
        ("Mamba",       "Mamba"),
        ("Hymba",       "Hymba"),
        ("CircMAC",     "circMAC"),
    ],
    # Frozen backbone (embedding = frozen model output)
    "pretrained_frozen": [
        ("RNA-MSM (frozen)", "RNAMSM\n(frozen)"),
        ("RNA-FM (frozen)",  "RNA-FM\n(frozen)"),
        ("CircMAC",          "circMAC"),
    ],
    # Fine-tuned (embedding = fine-tuned model output)
    "pretrained_ft": [
        ("RNA-MSM (ft)", "RNAMSM\n(fine-tuned)"),
        ("RNA-FM (ft)",  "RNA-FM\n(fine-tuned)"),
        ("CircMAC",      "circMAC"),
    ],
    # Legacy alias
    "pretrained": [
        ("RNA-MSM (ft)", "RNAMSM\n(fine-tuned)"),
        ("RNA-FM (ft)",  "RNA-FM\n(fine-tuned)"),
        ("CircMAC",      "circMAC"),
    ],
}

PRED_COL = {
    "LSTM":             "pred_lstm",
    "Transformer":      "pred_transformer",
    "Mamba":            "pred_mamba",
    "Hymba":            "pred_hymba",
    "CircMAC":          "pred_circmac",
    # Frozen
    "RNA-MSM (frozen)": "pred_rnamsm_frozen",
    "RNA-FM (frozen)":  "pred_rnafm_frozen",
    # Fine-tuned
    "RNA-MSM (ft)":     "pred_rnamsm_ft",
    "RNA-FM (ft)":      "pred_rnafm_ft",
}

CSV_MAP = {
    "cdyl2":    ROOT / "figures_paper/Fig_Case_CDYL2/data_predictions.csv",
    "huwe1":    ROOT / "figures_paper/Fig_Case_HUWE1/data_predictions.csv",
    "mapk1":    ROOT / "figures_paper/Fig_Case_MAPK1/data_predictions.csv",
    "pt_chr17": ROOT / "figures_paper/Fig_Case_PT_CHR17/data_predictions.csv",
    "app":      ROOT / "figures_paper/Fig_Case_APP/data_predictions.csv",
    "nfib":     ROOT / "figures_paper/Fig_Case_NFIB/data_predictions.csv",
}

ISO_MAP = {
    "cdyl2":    "chr4|84678168,84679116|84678259,84679242|-",
    "huwe1":    "chrX|53645311,53647368,53648212,53648573,53654063|53645463,53647574,53648310,53649000,53654131|-",
    "mapk1":    "chr22|21799012,21805850,21807664|21799128,21806039,21807846|-",
    "pt_chr17": "chr17|63193991,63200771|63194139,63200957|+",
    "app":      "chr21|25954590,25955627,25975070,25975954,25982344|25954689,25955755,25975228,25976028,25982477|-",
    "nfib":     "chr9|14120440,14125632,14146689,14150145,14155825,14179727|14120624,14125766,14146807,14150265,14155893,14179780|-",
}

MIRNA_MAP = {
    "cdyl2":    {"hsa-miR-34c-5p",  "miR-34c-5p"},
    "huwe1":    {"hsa-miR-29b-3p",  "miR-29b-3p"},
    "mapk1":    {"hsa-miR-12119",   "miR-12119"},
    "pt_chr17": {"hsa-miR-4732-5p", "miR-4732-5p"},
    "app":      {"hsa-miR-5001-3p", "miR-5001-3p"},
    "nfib":     {"hsa-miR-373-3p",  "miR-373-3p"},
}


# =============================================================================
# Style
# =============================================================================
BIND_COLOR    = "#E41A1C"
NONBIND_COLOR = "#2166AC"

PRED_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "binding_pred",
    ["#2166AC", "#F7F7F7", "#E41A1C"]
)

LABEL_BIND_SIZE    = 58
LABEL_NONBIND_SIZE = 34
PRED_SIZE          = 38

LABEL_EDGE_COLOR = "white"
PRED_EDGE_COLOR  = "#555555"

LABEL_EDGE_WIDTH = 0.60
PRED_EDGE_WIDTH  = 0.28

PANEL_BORDER_COLOR = "#BDBDBD"
PANEL_BORDER_WIDTH = 0.75

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         9,
    "axes.linewidth":    0.75,
    "axes.spines.top":   True,
    "axes.spines.right": True,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
})


# =============================================================================
# Utilities
# =============================================================================
def load_pickle(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_case_df(case_key):
    df = pd.read_csv(CSV_MAP[case_key])
    df = df[df["miRNA_ID"].isin(MIRNA_MAP[case_key])]
    df = df[df["isoform_ID"] == ISO_MAP[case_key]].reset_index(drop=True)

    if df.empty:
        raise ValueError(f"No rows matched: {case_key}")

    return df


def tight_square_limits(coords, margin=0.08):
    coords = np.asarray(coords)

    x_min, y_min = np.nanmin(coords, axis=0)
    x_max, y_max = np.nanmax(coords, axis=0)

    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)

    x_range = max(x_max - x_min, 1e-6)
    y_range = max(y_max - y_min, 1e-6)

    half = 0.5 * max(x_range, y_range)
    half = half * (1.0 + margin)

    return (cx - half, cx + half), (cy - half, cy + half)


def style_panel(ax, show_border=True):
    ax.set_xticks([])
    ax.set_yticks([])

    for sp in ax.spines.values():
        if show_border:
            sp.set_visible(True)
            sp.set_color(PANEL_BORDER_COLOR)
            sp.set_linewidth(PANEL_BORDER_WIDTH)
        else:
            sp.set_visible(False)


def style_metric_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    for sp in ax.spines.values():
        sp.set_visible(False)


def sil_score(coords, lbls):
    lbls = np.asarray(lbls)

    if len(np.unique(lbls)) < 2:
        return float("nan")

    if (lbls == 1).sum() < 2 or (lbls == 0).sum() < 2:
        return float("nan")

    try:
        return float(silhouette_score(coords, lbls))
    except Exception:
        return float("nan")


def case_title_one_line(case_title):
    return case_title.replace("\n×\n", " × ")


# =============================================================================
# Plotting helpers
# =============================================================================
def plot_label_panel(ax, coords, lbls):
    coords = np.asarray(coords)
    lbls = np.asarray(lbls)

    non_idx = np.where(lbls == 0)[0]
    bind_idx = np.where(lbls == 1)[0]

    ax.scatter(
        coords[non_idx, 0],
        coords[non_idx, 1],
        c=NONBIND_COLOR,
        s=LABEL_NONBIND_SIZE,
        alpha=0.90,
        edgecolors=LABEL_EDGE_COLOR,
        linewidths=LABEL_EDGE_WIDTH,
        zorder=1,
    )

    ax.scatter(
        coords[bind_idx, 0],
        coords[bind_idx, 1],
        c=BIND_COLOR,
        s=LABEL_BIND_SIZE,
        alpha=0.98,
        edgecolors=LABEL_EDGE_COLOR,
        linewidths=LABEL_EDGE_WIDTH,
        zorder=3,
    )


def plot_pred_panel(ax, coords, preds, mappable_store):
    coords = np.asarray(coords)
    preds = np.asarray(preds, dtype=float)

    # Sort ascending so high-prob points are drawn last (on top)
    order = np.argsort(preds)

    # Per-model normalization: use the model's own max so high-prob
    # points always appear red regardless of absolute scale
    vmax = max(float(preds.max()), 0.05)

    ax.scatter(
        coords[order, 0],
        coords[order, 1],
        c=preds[order],
        cmap=PRED_CMAP,
        vmin=0.0,
        vmax=vmax,
        s=PRED_SIZE,
        alpha=0.94,
        edgecolors=PRED_EDGE_COLOR,
        linewidths=PRED_EDGE_WIDTH,
        zorder=2,
    )

    if mappable_store[0] is None:
        sm = ScalarMappable(cmap=PRED_CMAP, norm=Normalize(0, 1))
        sm.set_array([])
        mappable_store[0] = sm


def add_metric_text(ax, sil):
    txt = f"Sil={sil:.3f}" if not np.isnan(sil) else "Sil=n/a"
    ax.text(
        0.5, -0.04, txt,
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=7.5, color="#444444",
    )


def add_case_labels(fig, axes, case_row_sets, cases):
    """
    Add 3-line case labels on the left side for the combined figure.
    """
    for (label_row, pred_row, metric_row), (case_title, _) in zip(case_row_sets, cases):
        pos_top = axes[label_row, 0].get_position()
        pos_bottom = axes[metric_row, 0].get_position()

        y_center = 0.5 * (pos_top.y1 + pos_bottom.y0)

        fig.text(
            0.040,
            y_center,
            case_title,
            ha="center",
            va="center",
            fontsize=9.8,
            fontweight="bold",
            color="black",
            linespacing=1.20,
        )


def make_legend_handles():
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=BIND_COLOR,
            markeredgecolor=LABEL_EDGE_COLOR,
            markeredgewidth=0.6,
            markersize=8,
            label="Binding",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=NONBIND_COLOR,
            markeredgecolor=LABEL_EDGE_COLOR,
            markeredgewidth=0.6,
            markersize=8,
            label="Non-binding",
        ),
    ]


def draw_case_on_axes(
    axes,
    label_row,
    pred_row,
    metric_row,
    models,
    coords_data,
    emb_data,
    case_df,
    case_key,
    mappable_store,
    add_titles=False,
    show_border=True,
):
    """Draw one case (label + pred rows) into axes grid."""
    for ci, (cache_name, display_name) in enumerate(models):
        ax_lbl = axes[label_row, ci]
        ax_pred = axes[pred_row, ci]
        ax_metric = axes[metric_row, ci]

        if cache_name not in coords_data or "coords_wi" not in coords_data[cache_name]:
            ax_lbl.axis("off")
            ax_pred.axis("off")
            ax_metric.axis("off")
            continue

        coords = np.asarray(coords_data[cache_name]["coords_wi"])
        lbls = np.asarray(
            emb_data.get(cache_name, emb_data["CircMAC"])["lbls"]
        )[:len(coords)]
        pred_col = PRED_COL.get(cache_name)
        sil = sil_score(coords, lbls)
        xlim, ylim = tight_square_limits(coords, margin=0.08)

        # Label panel
        plot_label_panel(ax_lbl, coords, lbls)
        ax_lbl.set_xlim(*xlim); ax_lbl.set_ylim(*ylim)
        ax_lbl.set_aspect("equal", adjustable="box")
        style_panel(ax_lbl, show_border=show_border)

        # Prediction panel
        if pred_col is not None and pred_col in case_df.columns:
            preds = case_df[pred_col].values[:len(coords)]
            plot_pred_panel(ax_pred, coords, preds, mappable_store)
        else:
            ax_pred.axis("off")
            ax_metric.axis("off")
            continue

        ax_pred.set_xlim(*xlim); ax_pred.set_ylim(*ylim)
        ax_pred.set_aspect("equal", adjustable="box")
        style_panel(ax_pred, show_border=show_border)
        add_metric_text(ax_pred, sil)
        style_metric_axis(ax_metric)

        if add_titles:
            ax_lbl.set_title(display_name, fontsize=11, fontweight="bold", pad=8)


def draw_case_label_only(
    axes,
    label_row,
    models,
    coords_data,
    emb_data,
    add_titles=False,
    show_border=True,
):
    """Draw one case (label row only + Sil below) into axes grid."""
    for ci, (cache_name, display_name) in enumerate(models):
        ax = axes[label_row, ci]

        if cache_name not in coords_data or "coords_wi" not in coords_data[cache_name]:
            ax.axis("off")
            continue

        coords = np.asarray(coords_data[cache_name]["coords_wi"])
        lbls = np.asarray(
            emb_data.get(cache_name, emb_data["CircMAC"])["lbls"]
        )[:len(coords)]
        sil = sil_score(coords, lbls)
        xlim, ylim = tight_square_limits(coords, margin=0.08)

        plot_label_panel(ax, coords, lbls)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        style_panel(ax, show_border=show_border)
        add_metric_text(ax, sil)

        if add_titles:
            ax.set_title(display_name, fontsize=11, fontweight="bold", pad=8)


# =============================================================================
# Combined figure
# =============================================================================
def make_combined_figure(group_key, red_key, wspace=0.0, show_border=True,
                         stem_suffix="", cases=None, stem_prefix=""):
    if cases is None:
        cases = MAIN_CASES
    models = GROUPS[group_key]
    n_models = len(models)
    n_cases = len(cases)

    # Build height_ratios and case_row_sets dynamically
    height_ratios = []
    case_row_sets = []
    row = 0
    for i in range(n_cases):
        case_row_sets.append((row, row + 1, row + 2))
        gap = 0.14 if i == n_cases - 1 else 0.18
        height_ratios += [1.0, 1.0, gap]
        row += 3
    total_rows = row

    total_h = 13.7 * n_cases / 3.0

    fig, axes = plt.subplots(
        total_rows,
        n_models,
        figsize=(2.55 * n_models, total_h),
        gridspec_kw={
            "height_ratios": height_ratios,
            "hspace": 0.0,
            "wspace": wspace,
        },
        squeeze=False,
    )

    for _, _, metric_row in case_row_sets:
        for c in range(n_models):
            style_metric_axis(axes[metric_row, c])

    mappable_store = [None]

    for case_idx, (case_title, case_key) in enumerate(cases):
        label_row, pred_row, metric_row = case_row_sets[case_idx]

        coords_data = load_pickle(CACHE / f"case_{case_key}_{red_key}_coords.pkl")
        emb_data = load_pickle(CACHE / f"case_{case_key}_embeddings.pkl")
        case_df = load_case_df(case_key)

        draw_case_on_axes(
            axes=axes,
            label_row=label_row,
            pred_row=pred_row,
            metric_row=metric_row,
            models=models,
            coords_data=coords_data,
            emb_data=emb_data,
            case_df=case_df,
            case_key=case_key,
            mappable_store=mappable_store,
            add_titles=(case_idx == 0),
            show_border=show_border,
        )

    fig.canvas.draw()

    # Case labels only. No separator lines.
    add_case_labels(fig, axes, case_row_sets, cases)

    # Legend aligned with colorbar
    handles = make_legend_handles()
    fig.legend(
        handles=handles,
        loc="lower left",
        ncol=2,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.16, 0.006),
        handletextpad=0.4,
        columnspacing=1.0,
    )

    # Colorbar aligned with legend
    if mappable_store[0] is not None:
        cax = fig.add_axes([0.16, 0.052, 0.72, 0.018])
        cb = fig.colorbar(
            mappable_store[0],
            cax=cax,
            orientation="horizontal",
        )
        cb.set_label("Predicted binding probability", fontsize=9, labelpad=2)
        cb.set_ticks([0.0, 0.5, 1.0])
        cb.ax.tick_params(labelsize=8)

    red_label = "UMAP" if red_key == "umap" else "t-SNE"
    group_label = (
        "General encoder models"
        if group_key == "encoder"
        else "Pretrained RNA language models"
    )

    # fig.suptitle(
    #     f"w/ miRNA embeddings  |  {red_label}  |  {group_label}",
    #     fontsize=14,
    #     fontweight="bold",
    #     y=0.992,
    # )

    # fig.subplots_adjust(
    #     left=0.14,
    #     right=0.995,
    #     top=0.93,
    #     bottom=0.115,
    # )

    prefix = f"{stem_prefix}_" if stem_prefix else ""
    stem = f"fig7_wi_{red_key}_{group_key}_{prefix}minimal_tight_case_aligned{stem_suffix}"

    for ext in ("pdf", "png"):
        out_path = OUT / f"{stem}.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved combined -> {out_path}")

    plt.close(fig)


# =============================================================================
# Case-wise figure
# =============================================================================
def make_single_case_figure(group_key, red_key, case_title, case_key,
                            wspace=0.0, show_border=True, stem_suffix=""):
    models = GROUPS[group_key]
    n_models = len(models)

    total_rows = 3
    height_ratios = [1.0, 1.0, 0.18]

    fig, axes = plt.subplots(
        total_rows,
        n_models,
        figsize=(2.65 * n_models, 5.0),
        gridspec_kw={
            "height_ratios": height_ratios,
            "hspace": 0.0,
            "wspace": wspace,
        },
        squeeze=False,
    )

    # Metric row
    for c in range(n_models):
        style_metric_axis(axes[2, c])

    mappable_store = [None]

    coords_data = load_pickle(CACHE / f"case_{case_key}_{red_key}_coords.pkl")
    emb_data = load_pickle(CACHE / f"case_{case_key}_embeddings.pkl")
    case_df = load_case_df(case_key)

    draw_case_on_axes(
        axes=axes,
        label_row=0,
        pred_row=1,
        metric_row=2,
        models=models,
        coords_data=coords_data,
        emb_data=emb_data,
        case_df=case_df,
        case_key=case_key,
        mappable_store=mappable_store,
        add_titles=True,
        show_border=show_border,
    )

    # Legend aligned with colorbar
    handles = make_legend_handles()
    fig.legend(
        handles=handles,
        loc="lower left",
        ncol=2,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.16, 0.030),
        handletextpad=0.4,
        columnspacing=1.0,
    )

    # Colorbar aligned with legend
    if mappable_store[0] is not None:
        cax = fig.add_axes([0.16, 0.120, 0.72, 0.030])
        cb = fig.colorbar(
            mappable_store[0],
            cax=cax,
            orientation="horizontal",
        )
        cb.set_label("Predicted binding probability", fontsize=9, labelpad=2)
        cb.set_ticks([0.0, 0.5, 1.0])
        cb.ax.tick_params(labelsize=8)

    red_label = "UMAP" if red_key == "umap" else "t-SNE"
    group_label = (
        "General encoder models"
        if group_key == "encoder"
        else "Pretrained RNA language models"
    )

    fig.suptitle(
        f"{case_title_one_line(case_title)}  |  w/ miRNA embeddings  |  {red_label}  |  {group_label}",
        fontsize=13,
        fontweight="bold",
        y=0.985,
    )

    fig.subplots_adjust(
        left=0.03,
        right=0.995,
        top=0.86,
        bottom=0.20,
    )

    stem = f"fig7_wi_{red_key}_{group_key}_{case_key}_case{stem_suffix}"

    for ext in ("pdf", "png"):
        out_path = OUT / f"{stem}.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved case -> {out_path}")

    plt.close(fig)


# =============================================================================
# Main
# =============================================================================
STYLES = [
    # (wspace, show_border, stem_suffix)
    (0.04, False, "_noborder_gap"),
    (0.0,  False, "_noborder_nogap"),
]


def run_for_cases(cases, stem_prefix):
    for group_key in ("encoder", "pretrained_frozen", "pretrained_ft"):
        for red_key in ("umap", "tsne"):
            for wspace, show_border, suffix in STYLES:
                if SAVE_COMBINED_FIGS:
                    make_combined_figure(
                        group_key, red_key,
                        wspace=wspace, show_border=show_border,
                        stem_suffix=suffix,
                        cases=cases, stem_prefix=stem_prefix,
                    )
                if SAVE_CASE_FIGS:
                    for case_title, case_key in cases:
                        make_single_case_figure(
                            group_key=group_key,
                            red_key=red_key,
                            case_title=case_title,
                            case_key=case_key,
                            wspace=wspace,
                            show_border=show_border,
                            stem_suffix=suffix,
                        )


def main():
    print("=== Main cases ===")
    run_for_cases(MAIN_CASES, stem_prefix="main")
    print("=== Supplementary cases ===")
    run_for_cases(SUPP_CASES, stem_prefix="supp")


# =============================================================================
# Label-only figure (combined, all 3 cases)
# =============================================================================
def make_label_only_figure(group_key, red_key, wspace=0.04, show_border=False,
                           stem_suffix="", cases=None):
    if cases is None:
        cases = MAIN_CASES
    models = GROUPS[group_key]
    n_models = len(models)
    n_cases = len(cases)

    fig, axes = plt.subplots(
        n_cases, n_models,
        figsize=(2.55 * n_models, 2.8 * n_cases),
        gridspec_kw={"hspace": 0.30, "wspace": wspace},
        squeeze=False,
    )

    for case_idx, (case_title, case_key) in enumerate(cases):
        coords_data = load_pickle(CACHE / f"case_{case_key}_{red_key}_coords.pkl")
        emb_data    = load_pickle(CACHE / f"case_{case_key}_embeddings.pkl")

        draw_case_label_only(
            axes=axes,
            label_row=case_idx,
            models=models,
            coords_data=coords_data,
            emb_data=emb_data,
            add_titles=(case_idx == 0),
            show_border=show_border,
        )

        axes[case_idx, 0].set_ylabel(
            case_title_one_line(case_title).split(" × ")[0],
            fontsize=9, fontweight="bold",
            rotation=0, ha="right", va="center", labelpad=8,
        )

    handles = make_legend_handles()
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 0.01))

    red_label   = "UMAP" if red_key == "umap" else "t-SNE"
    group_label = ("General encoder models" if group_key == "encoder"
                   else "Pretrained RNA language models")
    fig.suptitle(f"w/ miRNA embeddings  |  {red_label}  |  {group_label}",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.subplots_adjust(left=0.16, right=0.995, top=0.92, bottom=0.09)

    stem = f"fig7_wi_{red_key}_{group_key}_label_only{stem_suffix}"
    for ext in ("pdf", "png"):
        out_path = OUT / f"{stem}.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved label-only -> {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
    # Also generate label-only versions for main and supp cases
    for gk in ("encoder", "pretrained_frozen", "pretrained_ft"):
        for rk in ("umap", "tsne"):
            for cases, prefix in [(MAIN_CASES, "main"), (SUPP_CASES, "supp")]:
                make_label_only_figure(gk, rk, wspace=0.04, show_border=False,
                                       stem_suffix=f"_noborder_gap_{prefix}",
                                       cases=cases)