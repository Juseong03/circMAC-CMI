#!/usr/bin/env python3
"""
Standalone script: fig7_wi_umap_pretrained_minimal_tight_case_aligned_noborder_nogap

Layout:
  Rows: (label, pred, metric) × 3 cases = 9 rows total
  Cols: 5 pretrained models
  No borders, no column gap (wspace=0)
  Sil score shown below each prediction panel (outside)
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
# Models  ← 여기서 모델 순서/이름 수정
# =============================================================================
MODELS = [
    ("LSTM",        "LSTM",        "pred_lstm"),
    ("Transformer", "Transformer", "pred_transformer"),
    ("Mamba",       "Mamba",       "pred_mamba"),
    ("Hymba",       "Hymba",       "pred_hymba"),
    ("CircMAC",     "circMAC",     "pred_circmac"),
]


# =============================================================================
# Style  ← 여기서 색상/크기 수정
# =============================================================================
BIND_COLOR    = "#E41A1C"
NONBIND_COLOR = "#2166AC"
PRED_CMAP     = mcolors.LinearSegmentedColormap.from_list(
    "binding_pred", ["#2166AC", "#F7F7F7", "#E41A1C"])

LABEL_BIND_SIZE    = 58
LABEL_NONBIND_SIZE = 34
PRED_SIZE          = 38

LABEL_EDGE_COLOR = "white"
PRED_EDGE_COLOR  = "#555555"
LABEL_EDGE_WIDTH = 0.60
PRED_EDGE_WIDTH  = 0.28

# Figure layout  ← 여기서 패널 크기 수정
COL_WIDTH   = 2.55   # inches per model column
WSPACE      = 0.0    # 0 = no gap between columns
SHOW_BORDER = True   # True = thin grey border around each panel

# Left-side label positions  ← 여기서 위치 수정
CASE_LABEL_X_OFFSET = -0.1  # axes x0 기준 왼쪽 이동량 (음수 = 더 왼쪽)
CASE_LABEL_Y_OFFSET =  0.0   # 두 행 중앙 기준 위아래 이동 (양수 = 위, 음수 = 아래)

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         9,
    "axes.linewidth":    0.75,
    "axes.spines.top":   True,
    "axes.spines.right": True,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
})

PANEL_BORDER_COLOR = "#BDBDBD"
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
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        if SHOW_BORDER:
            sp.set_visible(True)
            # sp.set_color(PANEL_BORDER_COLOR)
            sp.set_linewidth(PANEL_BORDER_WIDTH)
        else:
            sp.set_visible(False)


def style_metric_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
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
    ax.text(0.5, -0.04, txt,
            transform=ax.transAxes, ha="center", va="top",
            fontsize=7.5, color="#444444")


def add_case_label(fig, axes, label_row, pred_row, case_title):
    # Row type labels: close to panels
    axes[label_row, 0].set_ylabel(
        "Labels",
        fontsize=8.5, fontweight="bold",
        rotation=0, ha="right", va="center", labelpad=6,
    )
    axes[pred_row, 0].set_ylabel(
        "Prediction",
        fontsize=8.5,  fontweight="bold",#color="#555555",
        rotation=0, ha="right", va="center", labelpad=6,
    )
    # Case name: average of the two axis centers (= where ylabel text sits)
    pos_lbl  = axes[label_row, 0].get_position()
    pos_pred = axes[pred_row,  0].get_position()
    y_lbl    = 0.5 * (pos_lbl.y0  + pos_lbl.y1)
    y_pred   = 0.5 * (pos_pred.y0 + pos_pred.y1)
    y_mid    = 0.5 * (y_lbl + y_pred) + CASE_LABEL_Y_OFFSET
    x_case   = pos_lbl.x0 + CASE_LABEL_X_OFFSET

    fig.text(
        x_case, y_mid, case_title,
        ha="center", va="center",
        fontsize=9.5, fontweight="bold",
        color="black", linespacing=1.4,
        transform=fig.transFigure,
    )


# =============================================================================
# Main
# =============================================================================
def main():
    n_models = len(MODELS)
    n_cases  = len(CASES)

    # Row layout: (label, pred, metric_spacer) × n_cases
    total_rows    = n_cases * 3
    height_ratios = [1.0, 1.0, 0.22] * n_cases
    height_ratios[-1] = 0.16   # last spacer slightly smaller

    case_row_sets = [(i*3, i*3+1, i*3+2) for i in range(n_cases)]

    fig, axes = plt.subplots(
        total_rows, n_models,
        figsize=(COL_WIDTH * n_models, 13.7),
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.0, "wspace": WSPACE},
        squeeze=False,
    )

    # Hide metric spacer axes
    for _, _, metric_row in case_row_sets:
        for c in range(n_models):
            style_metric_axis(axes[metric_row, c])

    mappable_store = [None]

    for case_idx, (case_title, case_key) in enumerate(CASES):
        label_row, pred_row, metric_row = case_row_sets[case_idx]

        coords_data = load_pickle(CACHE / f"case_{case_key}_umap_coords.pkl")
        emb_data    = load_pickle(CACHE / f"case_{case_key}_embeddings.pkl")
        case_df     = load_case_df(case_key)

        for ci, (cache_name, display_name, pred_col) in enumerate(MODELS):
            ax_lbl    = axes[label_row,  ci]
            ax_pred   = axes[pred_row,   ci]
            ax_metric = axes[metric_row, ci]

            if cache_name not in coords_data or "coords_wi" not in coords_data[cache_name]:
                ax_lbl.axis("off"); ax_pred.axis("off"); ax_metric.axis("off")
                continue

            coords = np.asarray(coords_data[cache_name]["coords_wi"])
            lbls   = np.asarray(emb_data.get(cache_name, emb_data["CircMAC"])["lbls"])[:len(coords)]
            sil    = sil_score(coords, lbls)
            xlim, ylim = tight_square_limits(coords)

            # ── Label panel ──────────────────────────────────────────────────
            plot_label_panel(ax_lbl, coords, lbls)
            ax_lbl.set_xlim(*xlim); ax_lbl.set_ylim(*ylim)
            ax_lbl.set_aspect("equal", adjustable="box")
            style_panel(ax_lbl)

            # ── Pred panel ───────────────────────────────────────────────────
            if pred_col in case_df.columns:
                plot_pred_panel(ax_pred, coords,
                                case_df[pred_col].values[:len(coords)],
                                mappable_store)
            else:
                ax_pred.axis("off"); continue

            ax_pred.set_xlim(*xlim); ax_pred.set_ylim(*ylim)
            ax_pred.set_aspect("equal", adjustable="box")
            style_panel(ax_pred)
            add_sil_text(ax_pred, sil)   # Sil below prediction panel

            style_metric_axis(ax_metric)

            # ── Column title (first case only) ────────────────────────────
            if case_idx == 0:
                ax_lbl.set_title(display_name, fontsize=11, fontweight="bold", pad=8)

    # ── Legend ────────────────────────────────────────────────────────────────
    handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=BIND_COLOR, markeredgecolor=LABEL_EDGE_COLOR,
               markeredgewidth=0.6, markersize=8, label="Binding"),
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=NONBIND_COLOR, markeredgecolor=LABEL_EDGE_COLOR,
               markeredgewidth=0.6, markersize=8, label="Non-binding"),
    ]
    fig.legend(handles=handles, loc="lower left", ncol=2, fontsize=9,
               frameon=False, bbox_to_anchor=(0.16, 0.05),
               handletextpad=0.4, columnspacing=1.0)

    # ── Colorbar ──────────────────────────────────────────────────────────────
    if mappable_store[0] is not None:
        cax = fig.add_axes([0.35, 0.052, 0.6, 0.018])
        cb  = fig.colorbar(mappable_store[0], cax=cax, orientation="horizontal")
        cb.set_label("Predicted binding probability", fontsize=9, labelpad=2)
        cb.set_ticks([0.0, 0.5, 1.0])
        cb.ax.tick_params(labelsize=10)

    # ── Title ─────────────────────────────────────────────────────────────────
    fig.suptitle("w/ miRNA embeddings  |  UMAP  |  General encoder models",
                 fontsize=14, fontweight="bold", y=0.992)
    fig.subplots_adjust(left=0.22, right=0.995, top=0.93, bottom=0.115)

    # ── Case labels (AFTER subplots_adjust so get_position() is accurate) ────
    fig.canvas.draw()
    for case_idx, (case_title, _) in enumerate(CASES):
        label_row, pred_row, _ = case_row_sets[case_idx]
        add_case_label(fig, axes, label_row, pred_row, case_title)

    stem = "fig7_wi_umap_encoder_combined"
    for ext in ("pdf", "png"):
        out_path = OUT / f"{stem}.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved -> {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
