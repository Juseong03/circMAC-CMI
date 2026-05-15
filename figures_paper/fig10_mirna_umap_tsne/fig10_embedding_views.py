#!/usr/bin/env python3
"""
Fig. 10 - Model-wise embedding views split by miRNA conditioning and color mode.

Outputs 16 files:
  {encoder, pretrained} x {umap, tsne} x {wo_label, wi_label, wi_pred, wi_combo}

Files with ``wo_label`` show w/o miRNA embeddings colored by binding labels.
Files with ``wi_label`` show w/ miRNA embeddings colored by binding labels.
Files with ``wi_pred`` show w/ miRNA embeddings colored by model prediction.
Files with ``wi_combo`` show w/ miRNA labels and predictions stacked vertically.
"""

from pathlib import Path
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd


ROOT  = Path(__file__).resolve().parents[2]
CACHE = ROOT / "figures_paper" / "embedding_cache"
OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

CASES = [
    {
        "label": "circCDYL2",
        "mirna": "miR-449a",
        "csv": ROOT / "figures_paper/Fig_Case_CDYL2/data_predictions.csv",
        "isoform": "chr4|84678168,84679116|84678259,84679242|-",
        "emb_key": "cdyl2",
    },
    {
        "label": "circMAPK1",
        "mirna": "miR-12119",
        "csv": ROOT / "figures_paper/Fig_Case_MAPK1/data_predictions.csv",
        "isoform": "chr22|21799012,21805850,21807664|21799128,21806039,21807846|-",
        "emb_key": "mapk1",
    },
    {
        "label": "circAPP",
        "mirna": "miR-5001-3p",
        "csv": ROOT / "figures_paper/Fig_Case_APP/data_predictions.csv",
        "isoform": "chr21|25954590,25955627,25975070,25975954,25982344|25954689,25955755,25975228,25976028,25982477|-",
        "emb_key": "app",
    },
]

GROUPS = {
    "encoder": [
        ("LSTM", "LSTM", "pred_lstm"),
        ("Transformer", "Transformer", "pred_transformer"),
        ("Mamba", "Mamba", "pred_mamba"),
        ("Hymba", "Hymba", "pred_hymba"),
        ("CircMAC", "CircMAC (ours)", "pred_circmac"),
    ],
    "pretrained": [
        ("RNABert (train)",  "RNABERT (fine-tuned)",  "pred_rnabert"),
        ("RNAErnie (train)", "RNAErnie (fine-tuned)",  "pred_rnaernie"),
        ("RNA-MSM (train)",  "RNAMSM (fine-tuned)",    "pred_rnamsm"),
        ("RNA-FM (train)",   "RNA-FM (fine-tuned)",    "pred_rnafm"),
        ("CircMAC",          "circMAC (ours)",          "pred_circmac"),
    ],
}

REDUCTIONS = {
    "umap": "UMAP",
    "tsne": "t-SNE",
}

MODES = {
    "wo_label": {"coord_key": "coords_wo", "kind": "label", "stem": "wo_label", "title": "w/o miRNA, labels"},
    "wi_label": {"coord_key": "coords_wi", "kind": "label", "stem": "wi_label", "title": "w/ miRNA, labels"},
    "wi_pred": {"coord_key": "coords_wi", "kind": "pred", "stem": "wi_pred", "title": "w/ miRNA, predictions"},
    "wi_combo": {"coord_key": "coords_wi", "kind": "combo", "stem": "wi_combo", "title": "w/ miRNA, labels + predictions"},
}

BIND_COLOR = "#D62728"
NONBIND_COLOR = "#AEC7E8"
PRED_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "binding_pred",
    [NONBIND_COLOR, "#FFFFFF", BIND_COLOR],
)

AX_MARGIN_FRAC = 0.05
UMAP_FIG_SCALE = 1.18
UMAP_POINT_SIZE = 8.0
TSNE_POINT_SIZE = 7.0
LABEL_BIND_SIZE = 16
LABEL_NONBIND_SIZE = 7.0
POINT_EDGE = "#111111"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def load_pickle(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing cache file: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_case_df(case):
    if not case["csv"].exists():
        raise FileNotFoundError(f"Missing case csv: {case['csv']}")
    df = pd.read_csv(case["csv"])
    mirna = case["mirna"]
    # Accept both "miR-449a" and "hsa-miR-449a" style IDs
    candidates = {mirna, f"hsa-{mirna}", mirna.replace("miR-", "hsa-miR-")}
    df = df[df["miRNA_ID"].isin(candidates)]
    df = df[df["isoform_ID"] == case["isoform"]].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No rows matched case {case['label']} / {case['mirna']}")
    return df


def draw_label_points(ax, coords, labels):
    labels = np.asarray(labels)
    coords = np.asarray(coords)
    n = min(len(coords), len(labels))
    coords = coords[:n]
    labels = labels[:n]

    nonbind = labels == 0
    bind = labels == 1

    ax.scatter(
        coords[nonbind, 0],
        coords[nonbind, 1],
        s=LABEL_NONBIND_SIZE,
        c=NONBIND_COLOR,
        alpha=0.62,
        edgecolors=POINT_EDGE,
        linewidths=0.18,
        rasterized=True,
    )
    ax.scatter(
        coords[bind, 0],
        coords[bind, 1],
        s=LABEL_BIND_SIZE,
        c=BIND_COLOR,
        alpha=0.92,
        edgecolors=POINT_EDGE,
        linewidths=0.28,
        rasterized=True,
    )


def draw_pred_points(ax, coords, preds, red_key="tsne"):
    preds = np.asarray(preds, dtype=float)
    coords = np.asarray(coords)
    n = min(len(coords), len(preds))
    coords = coords[:n]
    preds = preds[:n]

    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=preds,
        s=UMAP_POINT_SIZE if red_key == "umap" else TSNE_POINT_SIZE,
        cmap=PRED_CMAP,
        norm=Normalize(0.0, 1.0),
        alpha=0.90,
        edgecolors=POINT_EDGE,
        linewidths=0.12,
        rasterized=True,
    )


def compute_square_limits_from_coords(coords):
    coords = np.asarray(coords)
    if coords is None or len(coords) == 0:
        return None, None

    x_min, y_min = np.nanmin(coords, axis=0)
    x_max, y_max = np.nanmax(coords, axis=0)
    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)
    span = max(x_max - x_min, y_max - y_min, 1e-6) * (1.0 + AX_MARGIN_FRAC)
    half = 0.5 * span
    return (x_center - half, x_center + half), (y_center - half, y_center + half)


def get_row_square_limits(coords_list):
    valid = [np.asarray(c) for c in coords_list if c is not None and len(c) > 0]
    if not valid:
        return None, None
    return compute_square_limits_from_coords(np.vstack(valid))


def draw_panel(ax, coords, labels=None, preds=None, kind="label", red_key="tsne"):
    if kind == "label":
        draw_label_points(ax, coords, labels)
    else:
        draw_pred_points(ax, coords, preds, red_key=red_key)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def apply_limits(ax, xlim=None, ylim=None):
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")


def draw_combo_panel(ax_top, ax_bottom, coords, labels, preds, show_subtitles=False):
    draw_label_points(ax_top, coords, labels)
    draw_pred_points(ax_bottom, coords, preds)

    for ax in (ax_top, ax_bottom):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    if show_subtitles:
        ax_top.set_title("Label", fontsize=7.4, pad=2)
        ax_bottom.set_title("Pred", fontsize=7.4, pad=2)


def make_figure(group_key, red_key, mode_key):
    models = GROUPS[group_key]
    mode = MODES[mode_key]
    red_label = REDUCTIONS[red_key]
    stem = f"fig10_{group_key}_{red_key}_{mode['stem']}_model_comparison"

    n_rows = len(CASES)
    n_cols = len(models)
    combo_mode = mode["kind"] == "combo"
    use_row_limits = combo_mode
    fig_scale = UMAP_FIG_SCALE if red_key == "umap" else 1.0
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(
            ((2.2 if combo_mode else 2.25) * n_cols) * fig_scale,
            ((3.2 if combo_mode else 1.95) * n_rows) * fig_scale,
        ),
        squeeze=False,
    )

    colorbar_mappable = None

    for ri, case in enumerate(CASES):
        coords_data = load_pickle(CACHE / f"case_{case['emb_key']}_{red_key}_coords.pkl")
        emb_data = load_pickle(CACHE / f"case_{case['emb_key']}_embeddings.pkl")
        df = load_case_df(case)

        row_coords = []
        for cache_key, _, _ in models:
            if cache_key in coords_data and mode["coord_key"] in coords_data[cache_key]:
                row_coords.append(coords_data[cache_key][mode["coord_key"]])
        row_xlim, row_ylim = get_row_square_limits(row_coords) if use_row_limits else (None, None)

        for ci, (cache_key, display_name, pred_col) in enumerate(models):
            ax = axes[ri, ci]

            if cache_key not in coords_data or mode["coord_key"] not in coords_data[cache_key]:
                ax.axis("off")
                continue

            coords = coords_data[cache_key][mode["coord_key"]]

            if cache_key in emb_data:
                labels = np.asarray(emb_data[cache_key]["lbls"])[: len(coords)]
            else:
                labels = np.asarray(emb_data["CircMAC"]["lbls"])[: len(coords)]

            preds = None
            if mode["kind"] in ("pred", "combo"):
                if pred_col not in df.columns:
                    raise ValueError(f"Missing prediction column: {pred_col}")
                preds = df[pred_col].values[: len(coords)]

            if combo_mode:
                ax.set_visible(False)
                inner = ax.get_subplotspec().subgridspec(2, 1, hspace=0.18)
                ax_top = fig.add_subplot(inner[0, 0])
                ax_bottom = fig.add_subplot(inner[1, 0])
                draw_combo_panel(ax_top, ax_bottom, coords, labels, preds, show_subtitles=(ci == 0))
                apply_limits(ax_top, row_xlim, row_ylim)
                apply_limits(ax_bottom, row_xlim, row_ylim)
                if ri == 0:
                    ax_top.set_title(display_name, fontsize=9.5, fontweight="bold", pad=6)
                if ci == 0:
                    ax_top.set_ylabel(
                        f"{case['label']}\nx\n{case['mirna']}",
                        fontsize=7.8,
                        fontweight="bold",
                        rotation=0,
                        ha="right",
                        va="center",
                        labelpad=28,
                    )
                if colorbar_mappable is None:
                    from matplotlib.cm import ScalarMappable
                    colorbar_mappable = ScalarMappable(norm=Normalize(0.0, 1.0), cmap=PRED_CMAP)
                    colorbar_mappable.set_array([])
                continue

            draw_panel(ax, coords, labels=labels, preds=preds, kind=mode["kind"], red_key=red_key)
            if combo_mode:
                apply_limits(ax, row_xlim, row_ylim)

            if ri == 0:
                ax.set_title(display_name, fontsize=9.5, fontweight="bold", pad=6)
            if ci == 0:
                ax.set_ylabel(
                    f"{case['label']}\nx\n{case['mirna']}",
                    fontsize=8.5,
                    fontweight="bold",
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=36,
                )

            if mode["kind"] == "pred" and colorbar_mappable is None:
                from matplotlib.cm import ScalarMappable
                colorbar_mappable = ScalarMappable(norm=Normalize(0.0, 1.0), cmap=PRED_CMAP)
                colorbar_mappable.set_array([])

    if mode["kind"] == "label":
        handles = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor=BIND_COLOR, markeredgecolor="#111111", markeredgewidth=0.45, markersize=6, label="Binding site"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=NONBIND_COLOR, markeredgecolor="none", alpha=0.7, markersize=6, label="Non-binding site"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.01), fontsize=8.8)
    elif mode["kind"] == "pred":
        cb_ax = fig.add_axes([0.22, 0.035, 0.56, 0.022])
        cb = fig.colorbar(colorbar_mappable, cax=cb_ax, orientation="horizontal")
        cb.set_label("Predicted binding probability", fontsize=8.5)
        cb.set_ticks([0.0, 0.5, 1.0])
    else:
        handles = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor=BIND_COLOR, markeredgecolor="#111111", markeredgewidth=0.45, markersize=6, label="Binding site"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=NONBIND_COLOR, markeredgecolor="none", alpha=0.7, markersize=6, label="Non-binding site"),
        ]
        fig.legend(handles=handles, loc="lower left", ncol=2, frameon=False, bbox_to_anchor=(0.08, 0.005), fontsize=8.3)
        cb_ax = fig.add_axes([0.57, 0.038, 0.33, 0.020])
        cb = fig.colorbar(colorbar_mappable, cax=cb_ax, orientation="horizontal")
        cb.set_label("Predicted binding probability", fontsize=8.2)
        cb.set_ticks([0.0, 0.5, 1.0])

    fig.text(
        0.5,
        0.995,
        f"{group_key.capitalize()} embeddings - {red_label} - {mode['title']}",
        ha="center",
        va="top",
        fontsize=10.4 if red_key == "umap" else 10.0,
        fontweight="bold",
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.99,
        top=0.91,
        bottom=0.11,
        wspace=0.01,
        hspace=0.18,
    )

    for ext in ("pdf", "png"):
        out_path = OUT / f"{stem}.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved -> {out_path}")
    plt.close(fig)


def main():
    for group_key in ("encoder", "pretrained"):
        for red_key in ("umap", "tsne"):
            for mode_key in ("wo_label", "wi_label", "wi_pred", "wi_combo"):
                make_figure(group_key, red_key, mode_key)


if __name__ == "__main__":
    main()
