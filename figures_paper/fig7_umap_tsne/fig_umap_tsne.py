#!/usr/bin/env python3
"""
UMAP & t-SNE Embedding Visualization

Generates figures for two model groups and two color modes:
  - encoder_binding
  - encoder_position
  - pretrained_binding
  - pretrained_position

For each setting:
  - one figure per case
  - 2 rows: w/o miRNA, w/ miRNA
  - columns: models

Model order:
  Encoder:
    LSTM → Transformer → Mamba → Hymba → circMAC (ours)

  Pretrained RNA-LMs:
    RNABERT (fine-tuned) → RNAErnie (fine-tuned)
    → RNAMSM (fine-tuned) → RNA-FM (fine-tuned)
    → circMAC (ours)

Output:
  fig_umap_encoder_binding_cdyl2.{pdf,png}
  fig_umap_encoder_position_cdyl2.{pdf,png}
  fig_tsne_encoder_binding_cdyl2.{pdf,png}
  ...
"""

import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score


# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "figures_claude" / "emb_cache"
OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)


# ── Case definitions ─────────────────────────────────────────────────────────
CASES = [
    ("circCDYL2 × miR-449a", "cdyl2"),
    ("circMAPK1 × miR-12119", "mapk1"),
    ("circAPP × miR-5001-3p", "app"),
]


# ── Model groups ─────────────────────────────────────────────────────────────
# First element = cache key
# Second element = display label in figure
#
# Important:
#   Do not change cache keys unless the .pkl files use different names.
#   For pretrained RNA-LMs, cache keys still contain "(train)",
#   but display labels use "(fine-tuned)".
GROUPS = {
    "encoder": [
        ("LSTM",        "LSTM"),
        ("Transformer", "Transformer"),
        ("Mamba",       "Mamba"),
        ("Hymba",       "Hymba"),
        ("CircMAC",     "circMAC\n(ours)"),
    ],
    "pretrained": [
        ("RNABert (train)",    "RNABERT\n(fine-tuned)"),
        ("RNAErnie (train)",  "RNAErnie\n(fine-tuned)"),
        ("RNA-MSM (train)",   "RNAMSM\n(fine-tuned)"),
        ("RNA-FM (train)",    "RNA-FM\n(fine-tuned)"),
        ("CircMAC",           "circMAC\n(ours)"),
    ],
}


# ── Colors ───────────────────────────────────────────────────────────────────
BIND_COLOR = "#D62728"
NONBIND_COLOR = "#AEC7E8"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def sep_score(coords, lbls):
    """
    Euclidean distance between binding and non-binding centroids
    in the 2D projected space.

    This is a descriptive visualization score, not a primary performance metric.
    """
    b = coords[lbls == 1]
    nb = coords[lbls == 0]

    if len(b) == 0 or len(nb) == 0:
        return float("nan")

    return float(np.linalg.norm(b.mean(axis=0) - nb.mean(axis=0)))


def sil_score(coords, lbls):
    """
    Silhouette score [-1, 1] for binding/non-binding separation
    in the projected space.

    Higher values indicate better apparent separation and compactness.
    """
    lbls = np.asarray(lbls)

    if len(np.unique(lbls)) < 2:
        return float("nan")

    if np.sum(lbls == 1) < 2 or np.sum(lbls == 0) < 2:
        return float("nan")

    try:
        return float(silhouette_score(coords, lbls))
    except Exception:
        return float("nan")


def load_pickle(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing cache file: {path}")

    with open(path, "rb") as f:
        return pickle.load(f)


def get_reference_length(emb_data):
    """
    Use CircMAC labels as reference for position colorbar length.
    Fall back to the first available model if CircMAC is not present.
    """
    if "CircMAC" in emb_data:
        return len(emb_data["CircMAC"]["lbls"])

    first_key = next(iter(emb_data.keys()))
    return len(emb_data[first_key]["lbls"])


def hide_axis_frame(ax):
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_binding_panel(ax, coords, lbls):
    """
    Plot 2D embedding colored by binding annotation.
    Binding points are drawn after non-binding points.
    """
    lbls = np.asarray(lbls)

    order = np.argsort(lbls)
    ordered_lbls = lbls[order]

    colors = np.where(ordered_lbls == 1, BIND_COLOR, NONBIND_COLOR)
    sizes = np.where(ordered_lbls == 1, 18, 8)

    ax.scatter(
        coords[order, 0],
        coords[order, 1],
        c=colors,
        s=sizes,
        alpha=0.80,
        linewidths=0,
        zorder=2,
    )

    sep = sep_score(coords, lbls)
    sil = sil_score(coords, lbls)

    sep_str = f"{sep:.3f}" if not np.isnan(sep) else "n/a"
    sil_str = f"{sil:.3f}" if not np.isnan(sil) else "n/a"

    ax.text(
        0.5,
        -0.07,
        f"Sep={sep_str}  Sil={sil_str}",
        ha="center",
        va="top",
        transform=ax.transAxes,
        fontsize=7.0,
        color="#444444",
    )


def plot_position_panel(ax, coords):
    """Plot 2D embedding colored by sequence position."""
    n_pts = len(coords)
    pos = np.arange(n_pts)

    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=pos,
        cmap="viridis",
        s=10,
        alpha=0.80,
        linewidths=0,
    )

    return sc


def make_figure(red_type, group_key, color_mode, case_key, case_title):
    """
    red_type:
      'umap' or 'tsne'

    group_key:
      'encoder' or 'pretrained'

    color_mode:
      'binding' or 'position'

    case_key:
      'cdyl2', 'mapk1', or 'app'
    """
    if red_type not in {"umap", "tsne"}:
        raise ValueError(f"red_type must be 'umap' or 'tsne', got: {red_type}")

    if group_key not in GROUPS:
        raise ValueError(f"Unknown group_key: {group_key}")

    if color_mode not in {"binding", "position"}:
        raise ValueError(f"color_mode must be 'binding' or 'position', got: {color_mode}")

    models = GROUPS[group_key]
    n_models = len(models)

    coords_path = CACHE / f"case_{case_key}_{red_type}_coords.pkl"
    emb_path = CACHE / f"case_{case_key}_embeddings.pkl"

    coords_data = load_pickle(coords_path)
    emb_data = load_pickle(emb_path)

    fig, axes = plt.subplots(
        2,
        n_models,
        figsize=(3.05 * n_models, 5.9),
        gridspec_kw={
            "hspace": 0.38,
            "wspace": 0.22,
        },
        squeeze=False,
    )

    row_labels = ["circRNA only", "circRNA + miRNA"]
    coord_keys = ["coords_wo", "coords_wi"]

    # Reference length for position colorbar
    L_ref = get_reference_length(emb_data)

    # Store one scatter object for position colorbar
    position_scatter = None

    for ci, (cache_name, display_name) in enumerate(models):
        if cache_name not in coords_data:
            print(f"[Warning] Missing coordinates for model: {cache_name}")

            for ri in range(2):
                axes[ri, ci].axis("off")
                axes[ri, ci].set_title(
                    display_name,
                    fontsize=9,
                    fontweight="bold",
                    pad=5,
                )
            continue

        if cache_name in emb_data:
            lbls = np.asarray(emb_data[cache_name]["lbls"])
        else:
            print(
                f"[Warning] Missing labels for {cache_name}; "
                "using CircMAC labels as fallback."
            )
            lbls = np.asarray(emb_data["CircMAC"]["lbls"])

        for ri, (coord_key, row_label) in enumerate(zip(coord_keys, row_labels)):
            ax = axes[ri, ci]

            if coord_key not in coords_data[cache_name]:
                print(f"[Warning] Missing {coord_key} for model: {cache_name}")
                ax.axis("off")
                continue

            coords = np.asarray(coords_data[cache_name][coord_key])
            n_pts = len(coords)

            lbls_n = lbls[:n_pts]

            if len(lbls_n) != n_pts:
                print(
                    f"[Warning] Label length mismatch for {cache_name}: "
                    f"coords={n_pts}, labels={len(lbls)}"
                )

            if color_mode == "binding":
                plot_binding_panel(ax, coords, lbls_n)
            else:
                position_scatter = plot_position_panel(ax, coords)

            hide_axis_frame(ax)

            if ri == 0:
                ax.set_title(
                    display_name,
                    fontsize=9,
                    fontweight="bold",
                    pad=5,
                )

            if ci == 0:
                ax.set_ylabel(
                    row_label,
                    fontsize=9,
                    fontweight="bold",
                    labelpad=5,
                )

    # ── Legend / colorbar ────────────────────────────────────────────────────
    if color_mode == "binding":
        from matplotlib.lines import Line2D

        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=BIND_COLOR,
                markeredgecolor="none",
                markersize=7,
                label="Binding",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=NONBIND_COLOR,
                markeredgecolor="none",
                markersize=7,
                label="Non-binding",
            ),
        ]

        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=2,
            fontsize=8.5,
            frameon=False,
            bbox_to_anchor=(0.5, -0.005),
        )

    else:
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        sm = ScalarMappable(
            cmap="viridis",
            norm=Normalize(vmin=0, vmax=L_ref - 1),
        )
        sm.set_array([])

        cb = fig.colorbar(
            sm,
            ax=axes,
            orientation="horizontal",
            fraction=0.035,
            pad=0.075,
            aspect=42,
        )

        cb.set_label("Sequence position (5' → 3')", fontsize=8.5)
        cb.set_ticks([0, L_ref // 2, L_ref - 1])
        cb.set_ticklabels(["5'", "mid", "3'"])

    red_label = "UMAP" if red_type == "umap" else "t-SNE"

    if group_key == "encoder":
        group_label = "General encoder models"
    else:
        group_label = "Pretrained RNA language models"

    if color_mode == "binding":
        color_label = "colored by binding annotation"
    else:
        color_label = "colored by sequence position"

    fig.suptitle(
        f"{case_title} | {red_label} | {group_label} | {color_label}",
        fontsize=11,
        fontweight="bold",
        y=1.015,
    )

    return fig


def main():
    for group_key in ["encoder", "pretrained"]:
        for color_mode in ["binding", "position"]:
            for red_type in ["umap", "tsne"]:
                for case_title, case_key in CASES:
                    fig = make_figure(
                        red_type=red_type,
                        group_key=group_key,
                        color_mode=color_mode,
                        case_key=case_key,
                        case_title=case_title,
                    )

                    stem = f"fig_{red_type}_{group_key}_{color_mode}_{case_key}"

                    for ext in ["pdf", "png"]:
                        out_path = OUT / f"{stem}.{ext}"
                        fig.savefig(out_path, dpi=300, bbox_inches="tight")

                    print(f"Saved → {stem}")
                    plt.close(fig)


if __name__ == "__main__":
    main()