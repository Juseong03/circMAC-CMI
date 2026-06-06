#!/usr/bin/env python3
"""
fig10_encoder_umap_combined.py

Layout per case:
  Rows: (wo_label, wi_label, spacer) × n_cases
  Cols: 5 encoder models
  Sil score shown below each panel

Output:
  fig10_umap_encoder_main.{pdf,png}   — circCDYL2, circMAPK1, circHUWE1
  fig10_umap_encoder_supp.{pdf,png}   — circ17q21.31, circAPP, circNFIB
"""

from pathlib import Path
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
# Cases
# =============================================================================
MAIN_CASES = [
    ("circCDYL2\n×\nmiR-34c-5p",      "cdyl2"),
    ("circMAPK1\n×\nmiR-12119",        "mapk1"),
    ("circHUWE1\n×\nmiR-29b-3p",       "huwe1"),
]

SUPP_CASES = [
    ("circ17q21.31\n×\nmiR-4732-5p",  "pt_chr17"),
    ("circAPP\n×\nmiR-5001-3p",        "app"),
    ("circNFIB\n×\nmiR-373-3p",        "nfib"),
]

CSV_MAP = {
    "cdyl2":    ROOT / "figures_paper/Fig_Case_CDYL2/data_predictions.csv",
    "mapk1":    ROOT / "figures_paper/Fig_Case_MAPK1/data_predictions.csv",
    "huwe1":    ROOT / "figures_paper/Fig_Case_HUWE1/data_predictions.csv",
    "app":      ROOT / "figures_paper/Fig_Case_APP/data_predictions.csv",
    "pt_chr17": ROOT / "figures_paper/Fig_Case_PT_CHR17/data_predictions.csv",
    "nfib":     ROOT / "figures_paper/Fig_Case_NFIB/data_predictions.csv",
}
ISO_MAP = {
    "cdyl2":    "chr4|84678168,84679116|84678259,84679242|-",
    "mapk1":    "chr22|21799012,21805850,21807664|21799128,21806039,21807846|-",
    "huwe1":    "chrX|53645311,53647368,53648212,53648573,53654063|53645463,53647574,53648310,53649000,53654131|-",
    "app":      "chr21|25954590,25955627,25975070,25975954,25982344|25954689,25955755,25975228,25976028,25982477|-",
    "pt_chr17": "chr17|63193991,63200771|63194139,63200957|+",
    "nfib":     "chr9|14120440,14125632,14146689,14150145,14155825,14179727|14120624,14125766,14146807,14150265,14155893,14179780|-",
}
MIRNA_MAP = {
    "cdyl2":    {"hsa-miR-34c-5p"},
    "mapk1":    {"hsa-miR-12119"},
    "huwe1":    {"hsa-miR-29b-3p"},
    "app":      {"hsa-miR-5001-3p"},
    "pt_chr17": {"hsa-miR-4732-5p"},
    "nfib":     {"hsa-miR-373-3p"},
}


# =============================================================================
# Models
# =============================================================================
MODELS = [
    ("LSTM",        "LSTM",        ),
    ("Transformer", "Transformer", ),
    ("Mamba",       "Mamba",       ),
    ("Hymba",       "Hymba",       ),
    ("CircMAC",     "circMAC",     ),
]


# =============================================================================
# Style
# =============================================================================
BIND_COLOR    = "#E41A1C"
NONBIND_COLOR = "#2166AC"

LABEL_BIND_SIZE    = 58
LABEL_NONBIND_SIZE = 34

LABEL_EDGE_COLOR = "white"
LABEL_EDGE_WIDTH = 0.60

COL_WIDTH   = 2.55
WSPACE      = 0.0
SHOW_BORDER = True

CASE_LABEL_X_OFFSET = -0.10
CASE_LABEL_Y_OFFSET =  0.0

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
    cx, cy = 0.5 * (x_min + x_max), 0.5 * (y_min + y_max)
    half = 0.5 * max(x_max - x_min, y_max - y_min, 1e-6) * (1.0 + margin)
    return (cx - half, cx + half), (cy - half, cy + half)


def sil_score(coords, lbls):
    lbls = np.asarray(lbls)
    if len(np.unique(lbls)) < 2 or (lbls == 1).sum() < 2 or (lbls == 0).sum() < 2:
        return float("nan")
    try:
        return float(silhouette_score(coords, lbls))
    except Exception:
        return float("nan")


def style_panel(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(SHOW_BORDER)
        sp.set_linewidth(PANEL_BORDER_WIDTH)


def style_na_panel(ax):
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor("#F5F5F5")
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color("#CCCCCC")
        sp.set_linewidth(0.6)
    ax.text(0.5, 0.5, "N/A\n(seq too long)",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=8.5, color="#AAAAAA", style="italic")


def style_spacer_axis(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def plot_label_panel(ax, coords, lbls):
    coords, lbls = np.asarray(coords), np.asarray(lbls)
    non_idx  = np.where(lbls == 0)[0]
    bind_idx = np.where(lbls == 1)[0]
    ax.scatter(coords[non_idx,  0], coords[non_idx,  1],
               c=NONBIND_COLOR, s=LABEL_NONBIND_SIZE, alpha=0.90,
               edgecolors=LABEL_EDGE_COLOR, linewidths=LABEL_EDGE_WIDTH, zorder=1)
    ax.scatter(coords[bind_idx, 0], coords[bind_idx, 1],
               c=BIND_COLOR,    s=LABEL_BIND_SIZE,    alpha=0.98,
               edgecolors=LABEL_EDGE_COLOR, linewidths=LABEL_EDGE_WIDTH, zorder=3)


def add_sil_text(ax, sil):
    txt = f"Sil={sil:.3f}" if not np.isnan(sil) else "Sil=n/a"
    ax.text(0.5, -0.04, txt, transform=ax.transAxes,
            ha="center", va="top", fontsize=7.5, color="#444444")


def add_case_label(fig, axes, wo_row, wi_row, case_title):
    axes[wo_row, 0].set_ylabel(
        "w/o miRNA", fontsize=8.0, fontweight="bold",
        rotation=0, ha="right", va="center", labelpad=6)
    axes[wi_row, 0].set_ylabel(
        "w/ miRNA", fontsize=8.0, fontweight="bold",
        rotation=0, ha="right", va="center", labelpad=6)

    pos_wo = axes[wo_row, 0].get_position()
    pos_wi = axes[wi_row, 0].get_position()
    y_mid  = 0.5 * (0.5*(pos_wo.y0+pos_wo.y1) + 0.5*(pos_wi.y0+pos_wi.y1))
    y_mid += CASE_LABEL_Y_OFFSET
    x_case = pos_wo.x0 + CASE_LABEL_X_OFFSET
    fig.text(x_case, y_mid, case_title,
             ha="center", va="center", fontsize=9.5, fontweight="bold",
             color="black", linespacing=1.4, transform=fig.transFigure)


# =============================================================================
# Figure builder
# =============================================================================
def make_figure(cases, models, title_suffix, stem):
    n_models = len(models)
    n_cases  = len(cases)

    height_ratios = [1.0, 1.0, 0.22] * n_cases
    height_ratios[-1] = 0.16
    case_row_sets = [(i*3, i*3+1, i*3+2) for i in range(n_cases)]

    fig, axes = plt.subplots(
        n_cases * 3, n_models,
        figsize=(COL_WIDTH * n_models, 4.2 * n_cases),
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.0, "wspace": WSPACE},
        squeeze=False,
    )

    for _, _, spacer_row in case_row_sets:
        for c in range(n_models):
            style_spacer_axis(axes[spacer_row, c])

    for case_idx, (case_title, case_key) in enumerate(cases):
        wo_row, wi_row, spacer_row = case_row_sets[case_idx]

        coords_data = load_pickle(CACHE / f"case_{case_key}_umap_coords.pkl")
        emb_data    = load_pickle(CACHE / f"case_{case_key}_embeddings.pkl")

        for ci, (cache_name, display_name) in enumerate(models):
            ax_wo = axes[wo_row, ci]
            ax_wi = axes[wi_row, ci]

            # Column title always shown
            if case_idx == 0:
                ax_wo.set_title(display_name, fontsize=11, fontweight="bold", pad=8)

            has_wo = (cache_name in coords_data and "coords_wo" in coords_data[cache_name])
            has_wi = (cache_name in coords_data and "coords_wi" in coords_data[cache_name])

            if not has_wo and not has_wi:
                style_na_panel(ax_wo)
                style_na_panel(ax_wi)
                style_spacer_axis(axes[spacer_row, ci])
                continue

            lbls = np.asarray(emb_data.get(cache_name, emb_data["CircMAC"])["lbls"])

            # w/o miRNA label panel
            if has_wo:
                coords_wo = np.asarray(coords_data[cache_name]["coords_wo"])
                n = min(len(coords_wo), len(lbls))
                sil_wo = sil_score(coords_wo[:n], lbls[:n])
                xlim, ylim = tight_square_limits(coords_wo[:n])
                plot_label_panel(ax_wo, coords_wo[:n], lbls[:n])
                ax_wo.set_xlim(*xlim); ax_wo.set_ylim(*ylim)
                ax_wo.set_aspect("equal", adjustable="box")
                style_panel(ax_wo)
                # add_sil_text(ax_wo, sil_wo)
            else:
                style_na_panel(ax_wo)

            # w/ miRNA label panel
            if has_wi:
                coords_wi = np.asarray(coords_data[cache_name]["coords_wi"])
                n = min(len(coords_wi), len(lbls))
                sil_wi = sil_score(coords_wi[:n], lbls[:n])
                xlim, ylim = tight_square_limits(coords_wi[:n])
                plot_label_panel(ax_wi, coords_wi[:n], lbls[:n])
                ax_wi.set_xlim(*xlim); ax_wi.set_ylim(*ylim)
                ax_wi.set_aspect("equal", adjustable="box")
                style_panel(ax_wi)
                # add_sil_text(ax_wi, sil_wi)
            else:
                style_na_panel(ax_wi)

            style_spacer_axis(axes[spacer_row, ci])

    handles = [
        Line2D([0],[0], marker="o", color="none",
               markerfacecolor=BIND_COLOR,    markeredgecolor=LABEL_EDGE_COLOR,
               markeredgewidth=0.6, markersize=8, label="Binding"),
        Line2D([0],[0], marker="o", color="none",
               markerfacecolor=NONBIND_COLOR, markeredgecolor=LABEL_EDGE_COLOR,
               markeredgewidth=0.6, markersize=8, label="Non-binding"),
    ]
    fig.legend(handles=handles, loc="lower left", ncol=2, fontsize=9,
               frameon=False, bbox_to_anchor=(0.16, 0.05),
               handletextpad=0.4, columnspacing=1.0)

    # fig.suptitle(f"UMAP  |  {title_suffix}",
    #              fontsize=14, fontweight="bold", y=0.992)
    fig.subplots_adjust(left=0.22, right=0.995, top=0.93, bottom=0.10)

    fig.canvas.draw()
    for case_idx, (case_title, _) in enumerate(cases):
        wo_row, wi_row, _ = case_row_sets[case_idx]
        add_case_label(fig, axes, wo_row, wi_row, case_title)

    for ext in ("pdf", "png", "eps"):
        out_path = OUT / f"{stem}.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved -> {out_path}")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================
def main():
    make_figure(MAIN_CASES, MODELS,
                "General encoder models  |  Main cases",
                "fig10_umap_encoder_main")
    make_figure(SUPP_CASES, MODELS,
                "General encoder models  |  Supplementary cases",
                "fig10_umap_encoder_supp")


if __name__ == "__main__":
    main()
