#!/usr/bin/env python3
"""
fig11_pretrained_umap_pred_combined

Layout:
  Rows: (wo_pred, wi_pred, spacer) × 3 cases
  Cols: 5 encoder models
  Sil score shown below both rows (outside)
"""

from pathlib import Path
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

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
CASES = [
    ("circCDYL2\n×\nmiR-449a",   "cdyl2"),
    ("circMAPK1\n×\nmiR-12119",  "mapk1"),
    ("circAPP\n×\nmiR-5001-3p",  "app"),
]

CSV_MAP = {
    "cdyl2": ROOT / "figures_paper/Fig_Case_CDYL2/data_predictions.csv",
    "mapk1": ROOT / "figures_paper/Fig_Case_MAPK1/data_predictions.csv",
    "app":   ROOT / "figures_paper/Fig_Case_APP/data_predictions.csv",
}
ISO_MAP = {
    "cdyl2": "chr4|84678168,84679116|84678259,84679242|-",
    "mapk1": "chr22|21799012,21805850,21807664|21799128,21806039,21807846|-",
    "app":   "chr21|25954590,25955627,25975070,25975954,25982344|25954689,25955755,25975228,25976028,25982477|-",
}
MIRNA_MAP = {
    "cdyl2": {"hsa-miR-449a",    "miR-449a"},
    "mapk1": {"hsa-miR-12119",   "miR-12119"},
    "app":   {"hsa-miR-5001-3p", "miR-5001-3p"},
}


# =============================================================================
# Models
# =============================================================================
MODELS = [
    ("RNABert (train)",  "RNABERT\n(fine-tuned)",  "pred_rnabert"),
    ("RNAErnie (train)", "RNAErnie\n(fine-tuned)", "pred_rnaernie"),
    ("RNA-MSM (train)",  "RNAMSM\n(fine-tuned)",   "pred_rnamsm"),
    ("RNA-FM (train)",   "RNA-FM\n(fine-tuned)",   "pred_rnafm"),
    ("CircMAC",          "circMAC",                "pred_circmac"),
]


# =============================================================================
# Style
# =============================================================================
PRED_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "binding_pred", ["#2166AC", "#F7F7F7", "#E41A1C"])

PRED_SIZE       = 38
PRED_EDGE_COLOR = "#555555"
PRED_EDGE_WIDTH = 0.28

COL_WIDTH   = 2.55
WSPACE      = 0.0
SHOW_BORDER = True

CASE_LABEL_X_OFFSET = -0.10
CASE_LABEL_Y_OFFSET =  0.0

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         9,
    "axes.linewidth":    0.75,
    "axes.spines.top":   True,
    "axes.spines.right": True,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
})

PANEL_BORDER_WIDTH = 0.75


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
        if SHOW_BORDER:
            sp.set_linewidth(PANEL_BORDER_WIDTH)


def style_spacer(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def plot_pred_panel(ax, coords, preds, mappable_store):
    coords, preds = np.asarray(coords), np.asarray(preds, dtype=float)
    order = np.argsort(preds)
    ax.scatter(coords[order, 0], coords[order, 1],
               c=preds[order], cmap=PRED_CMAP, vmin=0.0, vmax=1.0,
               s=PRED_SIZE, alpha=0.94,
               edgecolors=PRED_EDGE_COLOR, linewidths=PRED_EDGE_WIDTH, zorder=2)
    if mappable_store[0] is None:
        sm = ScalarMappable(cmap=PRED_CMAP, norm=Normalize(0, 1))
        sm.set_array([])
        mappable_store[0] = sm


def add_sil_text(ax, sil):
    txt = f"Sil={sil:.3f}" if not np.isnan(sil) else "Sil=n/a"
    ax.text(0.5, -0.04, txt, transform=ax.transAxes,
            ha="center", va="top", fontsize=7.5, color="#444444")


def add_case_label(fig, axes, wo_row, wi_row, case_title):
    axes[wo_row, 0].set_ylabel(
        "Prediction\n(w/o miRNA)", fontsize=8.0, fontweight="bold",
        rotation=0, ha="right", va="center", labelpad=6,
    )
    axes[wi_row, 0].set_ylabel(
        "Prediction\n(w/ miRNA)", fontsize=8.0, fontweight="bold",
        rotation=0, ha="right", va="center", labelpad=6,
    )
    pos_wo = axes[wo_row, 0].get_position()
    pos_wi = axes[wi_row, 0].get_position()
    y_mid  = 0.5 * (0.5*(pos_wo.y0+pos_wo.y1) + 0.5*(pos_wi.y0+pos_wi.y1)) + CASE_LABEL_Y_OFFSET
    x_case = pos_wo.x0 + CASE_LABEL_X_OFFSET
    fig.text(x_case, y_mid, case_title,
             ha="center", va="center", fontsize=9.5, fontweight="bold",
             color="black", linespacing=1.4, transform=fig.transFigure)


# =============================================================================
# Main
# =============================================================================
def main():
    n_models = len(MODELS)
    n_cases  = len(CASES)

    total_rows    = n_cases * 3
    height_ratios = [1.0, 1.0, 0.22] * n_cases
    height_ratios[-1] = 0.16

    case_row_sets = [(i*3, i*3+1, i*3+2) for i in range(n_cases)]

    fig, axes = plt.subplots(
        total_rows, n_models,
        figsize=(COL_WIDTH * n_models, 13.7),
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.0, "wspace": WSPACE},
        squeeze=False,
    )

    for _, _, spacer_row in case_row_sets:
        for c in range(n_models):
            style_spacer(axes[spacer_row, c])

    mappable_store = [None]

    for case_idx, (case_title, case_key) in enumerate(CASES):
        wo_row, wi_row, spacer_row = case_row_sets[case_idx]

        coords_data = load_pickle(CACHE / f"case_{case_key}_umap_coords.pkl")
        emb_data    = load_pickle(CACHE / f"case_{case_key}_embeddings.pkl")
        case_df     = load_case_df(case_key)

        for ci, (cache_name, display_name, pred_col) in enumerate(MODELS):
            ax_wo = axes[wo_row, ci]
            ax_wi = axes[wi_row, ci]

            lbls = np.asarray(emb_data.get(cache_name, emb_data["CircMAC"])["lbls"])

            has_wo = cache_name in coords_data and "coords_wo" in coords_data[cache_name]
            has_wi = cache_name in coords_data and "coords_wi" in coords_data[cache_name]

            # w/o miRNA prediction
            if has_wo and pred_col in case_df.columns:
                coords = np.asarray(coords_data[cache_name]["coords_wo"])
                n = min(len(coords), len(lbls), len(case_df))
                xlim, ylim = tight_square_limits(coords[:n])
                plot_pred_panel(ax_wo, coords[:n], case_df[pred_col].values[:n], mappable_store)
                ax_wo.set_xlim(*xlim); ax_wo.set_ylim(*ylim)
                ax_wo.set_aspect("equal", adjustable="box")
                style_panel(ax_wo)
                add_sil_text(ax_wo, sil_score(coords[:n], lbls[:n]))
            else:
                ax_wo.axis("off")

            # w/ miRNA prediction
            if has_wi and pred_col in case_df.columns:
                coords = np.asarray(coords_data[cache_name]["coords_wi"])
                n = min(len(coords), len(lbls), len(case_df))
                xlim, ylim = tight_square_limits(coords[:n])
                plot_pred_panel(ax_wi, coords[:n], case_df[pred_col].values[:n], mappable_store)
                ax_wi.set_xlim(*xlim); ax_wi.set_ylim(*ylim)
                ax_wi.set_aspect("equal", adjustable="box")
                style_panel(ax_wi)
                add_sil_text(ax_wi, sil_score(coords[:n], lbls[:n]))
            else:
                ax_wi.axis("off")

            if case_idx == 0:
                ax_wo.set_title(display_name, fontsize=11, fontweight="bold", pad=8)

    # ── Colorbar ──────────────────────────────────────────────────────────────
    if mappable_store[0] is not None:
        cax = fig.add_axes([0.25, 0.038, 0.68, 0.013])
        cb  = fig.colorbar(mappable_store[0], cax=cax, orientation="horizontal")
        cb.set_label("Predicted binding probability", fontsize=9, labelpad=2)
        cb.set_ticks([0.0, 0.5, 1.0])
        cb.ax.tick_params(labelsize=10)

    fig.suptitle("UMAP  |  Pretrained RNA language models  |  Prediction",
                 fontsize=14, fontweight="bold", y=0.992)
    fig.subplots_adjust(left=0.24, right=0.995, top=0.955, bottom=0.085)

    fig.canvas.draw()
    for case_idx, (case_title, _) in enumerate(CASES):
        wo_row, wi_row, _ = case_row_sets[case_idx]
        add_case_label(fig, axes, wo_row, wi_row, case_title)

    stem = "fig11_pretrained_umap_pred_combined"
    for ext in ("pdf", "png"):
        out_path = OUT / f"{stem}.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved -> {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
