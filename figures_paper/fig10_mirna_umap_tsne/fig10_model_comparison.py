#!/usr/bin/env python3
"""
Fig. 10 - Model-wise embedding comparison.

Outputs four figure files:
  fig10_encoder_umap_model_comparison.{pdf,png}
  fig10_encoder_tsne_model_comparison.{pdf,png}
  fig10_pretrained_umap_model_comparison.{pdf,png}
  fig10_pretrained_tsne_model_comparison.{pdf,png}

Each file shows three cases, split by w/o vs w/ miRNA conditioning, with
one panel per model.
"""

from pathlib import Path
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT  = Path(__file__).resolve().parents[2]
CACHE = ROOT / "figures_paper" / "embedding_cache"
OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

CASES = [
    ("circCDYL2", "miR-449a", "cdyl2"),
    ("circMAPK1", "miR-12119", "mapk1"),
    ("circAPP", "miR-5001-3p", "app"),
]

GROUPS = {
    "encoder": [
        ("LSTM", "LSTM"),
        ("Transformer", "Transformer"),
        ("Mamba", "Mamba"),
        ("Hymba", "Hymba"),
        ("CircMAC", "CircMAC\n(ours)"),
    ],
    "pretrained": [
        ("RNABert (train)",  "RNABERT\n(fine-tuned)"),
        ("RNAErnie (train)", "RNAErnie\n(fine-tuned)"),
        ("RNA-MSM (train)",  "RNAMSM\n(fine-tuned)"),
        ("RNA-FM (train)",   "RNA-FM\n(fine-tuned)"),
        ("CircMAC",          "circMAC\n(ours)"),
    ],
}

REDUCTIONS = {
    "umap": "UMAP",
    "tsne": "t-SNE",
}

CONDITIONS = [
    ("wo", "w/o miRNA", "coords_wo"),
    ("wi", "w/ miRNA", "coords_wi"),
]

BIND_COLOR = "#D62728"
NONBIND_COLOR = "#AEC7E8"

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


def get_labels(emb_data, model_key, n_points):
    if model_key in emb_data:
        labels = np.asarray(emb_data[model_key]["lbls"])
    else:
        labels = np.asarray(emb_data["CircMAC"]["lbls"])
    return labels[:n_points]


def draw_embedding(ax, coords, labels):
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
        s=8,
        c=NONBIND_COLOR,
        alpha=0.58,
        linewidths=0,
        rasterized=True,
    )
    ax.scatter(
        coords[bind, 0],
        coords[bind, 1],
        s=20,
        c=BIND_COLOR,
        alpha=0.9,
        edgecolors="#111111",
        linewidths=0.25,
        rasterized=True,
    )

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def make_figure(group_key, red_key):
    models    = GROUPS[group_key]
    red_label = REDUCTIONS[red_key]
    stem      = f"fig10_{group_key}_{red_key}_model_comparison"
    n_cond    = len(CONDITIONS)
    n_rows    = len(CASES) * n_cond
    n_cols    = len(models)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.25 * n_cols, 1.65 * n_rows),
        gridspec_kw={"hspace": 0.22, "wspace": 0.07},
        squeeze=False,
    )

    for row_case, (case_name, mirna, case_key) in enumerate(CASES):
        emb_data    = load_pickle(CACHE / f"case_{case_key}_embeddings.pkl")
        coords_data = load_pickle(CACHE / f"case_{case_key}_{red_key}_coords.pkl")

        for cond_idx, (cond_key, cond_label, coord_key) in enumerate(CONDITIONS):
            row = row_case * n_cond + cond_idx
            for col, (model_key, display_name) in enumerate(models):
                ax = axes[row, col]

                if model_key not in coords_data or coord_key not in coords_data[model_key]:
                    ax.axis("off")
                    continue

                coords = coords_data[model_key][coord_key]
                labels = get_labels(emb_data, model_key, len(coords))
                draw_embedding(ax, coords, labels)

                # Column title: top row only
                if row == 0:
                    ax.set_title(display_name, fontsize=9.5, fontweight="bold", pad=6)

                # Row label: first column only; omit red_label (it's in suptitle)
                if col == 0:
                    ax.set_ylabel(
                        f"{case_name} × {mirna}\n{cond_label}",
                        fontsize=8.0,
                        fontweight="bold",
                        rotation=0,
                        ha="right",
                        va="center",
                        labelpad=38,
                    )

    # ── Horizontal separators between case groups ─────────────────────────────
    fig.canvas.draw()
    for case_idx in range(len(CASES) - 1):
        sep_row = (case_idx + 1) * n_cond - 1   # last row of this case group
        y_bot   = axes[sep_row,     0].get_position().y0
        y_top   = axes[sep_row + 1, 0].get_position().y1
        y_mid   = 0.5 * (y_bot + y_top)
        x0      = axes[sep_row, 0].get_position().x0
        x1      = axes[sep_row, n_cols - 1].get_position().x1
        fig.add_artist(plt.Line2D(
            [x0, x1], [y_mid, y_mid],
            transform=fig.transFigure,
            color="#aaaaaa", linewidth=0.9, linestyle="--",
        ))

    # ── Legend ────────────────────────────────────────────────────────────────
    handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=BIND_COLOR, markeredgecolor="#111111",
               markeredgewidth=0.45, markersize=6, label="Binding site"),
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=NONBIND_COLOR, markeredgecolor="none",
               alpha=0.7, markersize=6, label="Non-binding site"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.01), fontsize=8.8)

    group_label = ("General encoder models"
                   if group_key == "encoder"
                   else "Pretrained RNA language models (fine-tuned)")
    fig.suptitle(
        f"{group_label}  |  {red_label}",
        fontsize=12, fontweight="bold", y=0.99,
    )
    fig.subplots_adjust(left=0.16, right=0.99, top=0.94, bottom=0.07,
                        wspace=0.07, hspace=0.22)

    for ext in ("pdf", "png"):
        out_path = OUT / f"{stem}.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved -> {out_path}")
    plt.close(fig)


def main():
    for group_key in ("encoder", "pretrained"):
        for red_key in ("umap", "tsne"):
            make_figure(group_key, red_key)


if __name__ == "__main__":
    main()
