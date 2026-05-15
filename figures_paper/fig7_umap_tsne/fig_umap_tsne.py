#!/usr/bin/env python3
"""
Fig 7 — Per-case UMAP / t-SNE embedding: label + prediction (4 rows × n_models cols)

Rows:
  0  w/o miRNA  · colored by binding label
  1  w/o miRNA  · colored by model prediction
  2  w/ miRNA   · colored by binding label
  3  w/ miRNA   · colored by model prediction

One figure per (case × group × reduction).
Output stem: fig_{umap|tsne}_{encoder|pretrained}_labelpred_{cdyl2|mapk1|app}
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

from sklearn.metrics import silhouette_score

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT  = Path(__file__).resolve().parents[2]
CACHE = ROOT / "figures_paper" / "embedding_cache"
OUT   = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

# ── Cases ─────────────────────────────────────────────────────────────────────
CASES = [
    ("circCDYL2 × miR-449a",    "cdyl2"),
    ("circMAPK1 × miR-12119",   "mapk1"),
    ("circAPP × miR-5001-3p",   "app"),
]

# ── Model groups ──────────────────────────────────────────────────────────────
GROUPS = {
    "encoder": [
        ("LSTM",        "LSTM"),
        ("Transformer", "Transformer"),
        ("Mamba",       "Mamba"),
        ("Hymba",       "Hymba"),
        ("CircMAC",     "circMAC\n(ours)"),
    ],
    "pretrained": [
        ("RNABert (train)",  "RNABERT\n(fine-tuned)"),
        ("RNAErnie (train)", "RNAErnie\n(fine-tuned)"),
        ("RNA-MSM (train)",  "RNAMSM\n(fine-tuned)"),
        ("RNA-FM (train)",   "RNA-FM\n(fine-tuned)"),
        ("CircMAC",          "circMAC\n(ours)"),
    ],
}

PRED_COL = {
    "LSTM":             "pred_lstm",
    "Transformer":      "pred_transformer",
    "Mamba":            "pred_mamba",
    "Hymba":            "pred_hymba",
    "CircMAC":          "pred_circmac",
    "RNABert (train)":  "pred_rnabert",
    "RNAErnie (train)": "pred_rnaernie",
    "RNA-MSM (train)":  "pred_rnamsm",
    "RNA-FM (train)":   "pred_rnafm",
}

# ── Colors ────────────────────────────────────────────────────────────────────
BIND_COLOR    = "#D62728"
NONBIND_COLOR = "#AEC7E8"
PRED_CMAP     = mcolors.LinearSegmentedColormap.from_list(
    "binding_pred", [NONBIND_COLOR, "#FFFFFF", BIND_COLOR])

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         9,
    "axes.linewidth":    0.7,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
})

# Row layout: (coord_key, row_label, kind)
ROW_SPECS = [
    ("coords_wo", "w/o miRNA\nLabel",      "label"),
    ("coords_wo", "w/o miRNA\nPrediction", "pred"),
    ("coords_wi", "w/ miRNA\nLabel",       "label"),
    ("coords_wi", "w/ miRNA\nPrediction",  "pred"),
]
# Rows 0-1 belong to condition "wo", rows 2-3 to "wi"
COND_ROWS = {"wo": [0, 1], "wi": [2, 3]}


# ── Utilities ─────────────────────────────────────────────────────────────────
def load_pickle(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_case_df(case_key):
    import pandas as pd
    csv_map = {
        "cdyl2": ROOT / "figures_paper/Fig_Case_CDYL2/data_predictions.csv",
        "mapk1": ROOT / "figures_paper/Fig_Case_MAPK1/data_predictions.csv",
        "app":   ROOT / "figures_paper/Fig_Case_APP/data_predictions.csv",
    }
    iso_map = {
        "cdyl2": "chr4|84678168,84679116|84678259,84679242|-",
        "mapk1": "chr22|21799012,21805850,21807664|21799128,21806039,21807846|-",
        "app":   "chr21|25954590,25955627,25975070,25975954,25982344|25954689,25955755,25975228,25976028,25982477|-",
    }
    mirna_map = {
        "cdyl2": {"hsa-miR-449a",    "miR-449a"},
        "mapk1": {"hsa-miR-12119",   "miR-12119"},
        "app":   {"hsa-miR-5001-3p", "miR-5001-3p"},
    }
    df = pd.read_csv(csv_map[case_key])
    df = df[df["miRNA_ID"].isin(mirna_map[case_key])]
    df = df[df["isoform_ID"] == iso_map[case_key]].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No rows matched: {case_key}")
    return df


def square_limits(coords_list):
    """Shared square axis limits that encompass all coord arrays."""
    valid = [np.asarray(c) for c in coords_list if c is not None and len(c) > 0]
    if not valid:
        return None, None
    coords = np.vstack(valid)
    x_min, y_min = np.nanmin(coords, axis=0)
    x_max, y_max = np.nanmax(coords, axis=0)
    cx, cy = 0.5 * (x_min + x_max), 0.5 * (y_min + y_max)
    half = 0.5 * max(x_max - x_min, y_max - y_min, 1e-6) * 1.08
    return (cx - half, cx + half), (cy - half, cy + half)


def hide_frame(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def sep_score(coords, lbls):
    b, nb = coords[lbls == 1], coords[lbls == 0]
    if len(b) == 0 or len(nb) == 0:
        return float("nan")
    return float(np.linalg.norm(b.mean(0) - nb.mean(0)))


def sil_score(coords, lbls):
    lbls = np.asarray(lbls)
    if len(np.unique(lbls)) < 2 or (lbls == 1).sum() < 2 or (lbls == 0).sum() < 2:
        return float("nan")
    try:
        return float(silhouette_score(coords, lbls))
    except Exception:
        return float("nan")


def plot_label_panel(ax, coords, lbls):
    order = np.argsort(lbls)
    colors = np.where(lbls[order] == 1, BIND_COLOR, NONBIND_COLOR)
    sizes  = np.where(lbls[order] == 1, 20, 9)
    ax.scatter(coords[order, 0], coords[order, 1],
               c=colors, s=sizes, alpha=0.80, linewidths=0, zorder=2)
    sep = sep_score(coords, lbls)
    sil = sil_score(coords, lbls)
    score_str = (f"Sep={sep:.2f}  Sil={sil:.2f}"
                 if not np.isnan(sep) else "n/a")
    ax.text(0.5, -0.06, score_str,
            transform=ax.transAxes, ha="center", va="top",
            fontsize=6.5, color="#555555")


def plot_pred_panel(ax, coords, preds, mappable_store):
    preds = np.asarray(preds, dtype=float)
    ax.scatter(coords[:, 0], coords[:, 1],
               c=preds, cmap=PRED_CMAP, vmin=0.0, vmax=1.0,
               s=12, alpha=0.88, linewidths=0, zorder=2)
    if mappable_store[0] is None:
        sm = ScalarMappable(cmap=PRED_CMAP, norm=Normalize(0, 1))
        sm.set_array([])
        mappable_store[0] = sm


def add_case_separator(fig, axes, row_above, n_cols):
    """Draw a thin horizontal rule between rows in figure coordinates."""
    # Average the bottom of row_above and top of row_below
    y_top    = axes[row_above, 0].get_position().y0
    y_bottom = axes[row_above + 1, 0].get_position().y1
    y_mid    = 0.5 * (y_top + y_bottom)
    x0 = axes[row_above, 0].get_position().x0
    x1 = axes[row_above, n_cols - 1].get_position().x1
    line = plt.Line2D([x0, x1], [y_mid, y_mid],
                      transform=fig.transFigure,
                      color="#bbbbbb", linewidth=0.8, linestyle="--")
    fig.add_artist(line)


# ── Main builder ──────────────────────────────────────────────────────────────
def make_figure(red_type, group_key, case_key, case_title):
    models    = GROUPS[group_key]
    n_models  = len(models)
    n_rows    = len(ROW_SPECS)

    coords_data = load_pickle(CACHE / f"case_{case_key}_{red_type}_coords.pkl")
    emb_data    = load_pickle(CACHE / f"case_{case_key}_embeddings.pkl")
    case_df     = load_case_df(case_key)

    # ── Per-condition square axis limits ──────────────────────────────────────
    cond_lims = {}
    for cond_key, row_indices in COND_ROWS.items():
        coord_key = ROW_SPECS[row_indices[0]][0]   # e.g. "coords_wo"
        pool = []
        for cache_name, _ in models:
            if cache_name in coords_data and coord_key in coords_data[cache_name]:
                pool.append(coords_data[cache_name][coord_key])
        cond_lims[cond_key] = square_limits(pool)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        n_rows, n_models,
        figsize=(2.7 * n_models, 9.2),
        gridspec_kw={"hspace": 0.36, "wspace": 0.12},
        squeeze=False,
    )

    mappable_store = [None]   # mutable container for colorbar handle

    for ci, (cache_name, display_name) in enumerate(models):
        if cache_name not in coords_data:
            for ri in range(n_rows):
                axes[ri, ci].axis("off")
            if ci == 0:
                for ri in range(n_rows):
                    axes[ri, ci].set_ylabel(ROW_SPECS[ri][1],
                                            fontsize=8.4, fontweight="bold", labelpad=7)
            if ri == 0:
                axes[0, ci].set_title(display_name, fontsize=9, fontweight="bold", pad=5)
            continue

        lbls = np.asarray(emb_data.get(cache_name, emb_data["CircMAC"])["lbls"])
        pred_col = PRED_COL.get(cache_name)
        if pred_col is None or pred_col not in case_df.columns:
            pred_col = None

        for ri, (coord_key, row_label, row_kind) in enumerate(ROW_SPECS):
            ax = axes[ri, ci]

            if coord_key not in coords_data[cache_name]:
                ax.axis("off")
                continue

            coords  = np.asarray(coords_data[cache_name][coord_key])
            n_pts   = len(coords)
            lbls_n  = lbls[:n_pts]

            # Determine which condition this row belongs to
            cond_key = "wo" if ri < 2 else "wi"
            xlim, ylim = cond_lims[cond_key]

            if row_kind == "label":
                plot_label_panel(ax, coords, lbls_n)
            else:
                if pred_col is not None:
                    plot_pred_panel(ax, coords,
                                    case_df[pred_col].values[:n_pts],
                                    mappable_store)
                else:
                    ax.axis("off")

            if xlim is not None:
                ax.set_xlim(*xlim)
                ax.set_ylim(*ylim)
                ax.set_aspect("equal", adjustable="box")

            hide_frame(ax)

            if ri == 0:
                ax.set_title(display_name, fontsize=9, fontweight="bold", pad=5)
            if ci == 0:
                ax.set_ylabel(row_label, fontsize=8.4, fontweight="bold", labelpad=7)

    # ── Legend / colorbar ─────────────────────────────────────────────────────
    handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=BIND_COLOR, markeredgecolor="none",
               markersize=7, label="Binding"),
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=NONBIND_COLOR, markeredgecolor="none",
               markersize=7, label="Non-binding"),
    ]
    fig.legend(handles=handles, loc="lower left", ncol=2, fontsize=8.3,
               frameon=False, bbox_to_anchor=(0.10, 0.0))

    if mappable_store[0] is not None:
        cb = fig.colorbar(
            mappable_store[0], ax=axes,
            orientation="horizontal",
            fraction=0.022, pad=0.07, aspect=44,
        )
        cb.set_label("Predicted binding probability", fontsize=8.2)
        cb.set_ticks([0.0, 0.5, 1.0])

    # ── Separator between w/o and w/ miRNA groups ─────────────────────────────
    fig.canvas.draw()   # needed so get_position() returns real values
    add_case_separator(fig, axes, row_above=1, n_cols=n_models)

    # ── Title ─────────────────────────────────────────────────────────────────
    red_label   = "UMAP" if red_type == "umap" else "t-SNE"
    group_label = ("General encoder models"
                   if group_key == "encoder"
                   else "Pretrained RNA language models")
    fig.suptitle(
        f"{case_title}  |  {red_label}  |  {group_label}",
        fontsize=11, fontweight="bold", y=0.99,
    )

    fig.subplots_adjust(left=0.12, right=0.98, top=0.92, bottom=0.08)
    return fig


def main():
    for group_key in ["encoder", "pretrained"]:
        for red_type in ["umap", "tsne"]:
            for case_title, case_key in CASES:
                fig = make_figure(
                    red_type=red_type,
                    group_key=group_key,
                    case_key=case_key,
                    case_title=case_title,
                )
                stem = f"fig_{red_type}_{group_key}_labelpred_{case_key}"
                for ext in ["pdf", "png"]:
                    fig.savefig(OUT / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
                print(f"Saved → {stem}")
                plt.close(fig)


if __name__ == "__main__":
    main()
