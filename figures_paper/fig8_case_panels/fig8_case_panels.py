#!/usr/bin/env python3
"""
Fig 8 — Per-case combined panels: Heatmap + UMAP binding

One figure per case (CDYL2, MAPK1, APP):
  Top section  : Heatmap comparison (GT row + model prediction rows, BSJ lines)
  Bottom section: UMAP binding scatter (circRNA only / circRNA+miRNA, Sep+Sil)

Model groups:
  Encoder   : LSTM → Transformer → Mamba → Hymba → circMAC (ours)
  Pretrained: RNABERT → RNAErnie → RNAMSM → RNA-FM → circMAC (ours)

Output:
  fig8_{group}_heatmap_umap_{case_key}.{pdf,png}
  (groups: encoder, pretrained; cases: cdyl2, mapk1, app)
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from sklearn.metrics import silhouette_score


# ── Paths ────────────────────────────────────────────────────────────────────
ROOT  = Path(__file__).resolve().parents[2]
CACHE = ROOT / "figures_claude" / "emb_cache"
OUT   = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)


# ── Case definitions ─────────────────────────────────────────────────────────
CASES = [
    dict(
        key     = "cdyl2",
        label   = "circCDYL2 (chr4)",
        csv     = ROOT / "figures_claude/Fig5_Case_CDYL2/data_predictions.csv",
        isoform = "chr4|84678168,84679116|84678259,84679242|-",
        mirna   = "hsa-miR-449a",
        emb_key = "cdyl2",
    ),
    dict(
        key     = "mapk1",
        label   = "circMAPK1 (chr22)",
        csv     = ROOT / "figures_claude/Fig6_Case_MAPK1/data_predictions.csv",
        isoform = "chr22|21799012,21805850,21807664|21799128,21806039,21807846|-",
        mirna   = "hsa-miR-12119",
        emb_key = "mapk1",
    ),
    dict(
        key     = "app",
        label   = "circAPP (chr21)",
        csv     = ROOT / "figures_claude/Fig7_Case_APP/data_predictions.csv",
        isoform = "chr21|25954590,25955627,25975070,25975954,25982344|25954689,25955755,25975228,25976028,25982477|-",
        mirna   = "hsa-miR-5001-3p",
        emb_key = "app",
    ),
]


# ── Model groups ─────────────────────────────────────────────────────────────
MODEL_COLORS = {
    "circmac":     "#FF7F0E",
    "lstm":        "#E377C2",
    "transformer": "#8C564B",
    "mamba":       "#D62728",
    "hymba":       "#BCBD22",
    "rnabert":     "#1F77B4",
    "rnaernie":    "#9467BD",
    "rnamsm":      "#2CA02C",
    "rnafm":       "#17BECF",
}

BSJ_COLOR    = "#1F77B4"
GT_COLOR     = "#8B0000"
BIND_COLOR   = "#D62728"
NONBIND_COLOR = "#AEC7E8"

# (color_key, display_label, pred_col, umap_cache_key)
GROUPS = {
    "encoder": [
        ("lstm",        "LSTM",           "pred_lstm",        "LSTM"),
        ("transformer", "Transformer",    "pred_transformer", "Transformer"),
        ("mamba",       "Mamba",          "pred_mamba",       "Mamba"),
        ("hymba",       "Hymba",          "pred_hymba",       "Hymba"),
        ("circmac",     "circMAC (ours)", "pred_circmac",     "CircMAC"),
    ],
    "pretrained": [
        ("rnabert",  "RNABERT\n(fine-tuned)",  "pred_rnabert",  "RNABert (train)"),
        ("rnaernie", "RNAErnie\n(fine-tuned)", "pred_rnaernie", "RNAErnie (train)"),
        ("rnamsm",   "RNAMSM\n(fine-tuned)",   "pred_rnamsm",   "RNA-MSM (train)"),
        ("rnafm",    "RNA-FM\n(fine-tuned)",   "pred_rnafm",    "RNA-FM (train)"),
        ("circmac",  "circMAC (ours)",          "pred_circmac",  "CircMAC"),
    ],
}


# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "font.size":    9,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype":  42,
})


# ── Helpers ──────────────────────────────────────────────────────────────────
def load_pickle(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing cache: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def sep_score(coords, lbls):
    b  = coords[lbls == 1]
    nb = coords[lbls == 0]
    if len(b) == 0 or len(nb) == 0:
        return float("nan")
    return float(np.linalg.norm(b.mean(axis=0) - nb.mean(axis=0)))


def sil_score(coords, lbls):
    lbls = np.asarray(lbls)
    if len(np.unique(lbls)) < 2:
        return float("nan")
    if np.sum(lbls == 1) < 2 or np.sum(lbls == 0) < 2:
        return float("nan")
    try:
        return float(silhouette_score(coords, lbls))
    except Exception:
        return float("nan")


def hide_frame(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_bsj(ax, L, lw=1.0, alpha=0.55):
    ax.axvline(-0.5,     color=BSJ_COLOR, lw=lw, ls="--", alpha=alpha)
    ax.axvline(L - 0.5,  color=BSJ_COLOR, lw=lw, ls="--", alpha=alpha)


def clean_spines(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)


# ── Heatmap section ──────────────────────────────────────────────────────────
def draw_heatmap_section(fig, gs_heatmap, df, gt, L, models):
    """
    Draw GT row + model prediction rows inside the heatmap GridSpec slot.
    Row labels shown on every row.
    """
    n_models = len(models)

    gs_inner = GridSpecFromSubplotSpec(
        n_models + 1, 1,
        subplot_spec=gs_heatmap,
        height_ratios=[1.35] + [1.0] * n_models,
        hspace=0.08,
    )

    # Ground truth row
    ax_gt = fig.add_subplot(gs_inner[0])
    ax_gt.imshow(gt[np.newaxis, :], aspect="auto", cmap="Reds",
                 vmin=0, vmax=1, interpolation="nearest")
    ax_gt.set_yticks([])
    ax_gt.set_ylabel("Ground truth", rotation=0, ha="right", va="center",
                     fontsize=8, fontweight="bold", color=GT_COLOR, labelpad=40)
    ax_gt.set_xticks([])
    draw_bsj(ax_gt, L, lw=1.4, alpha=0.75)
    clean_spines(ax_gt)

    bottom_ax = None
    for ri, (mkey, mname, mcol, _) in enumerate(models):
        if mcol not in df.columns:
            continue
        ax = fig.add_subplot(gs_inner[ri + 1])
        pred  = df[mcol].values
        color = MODEL_COLORS.get(mkey, "#888888")
        cmap  = mcolors.LinearSegmentedColormap.from_list(f"{mkey}_cm", ["#f7f7f7", color])
        ax.imshow(pred[np.newaxis, :], aspect="auto", cmap=cmap,
                  vmin=0, vmax=1, interpolation="nearest")
        ax.set_yticks([])
        ax.set_ylabel(mname, rotation=0, ha="right", va="center",
                      fontsize=8, fontweight="bold", color=color, labelpad=40)
        ax.set_xticks([])
        draw_bsj(ax, L, lw=0.9, alpha=0.5)
        clean_spines(ax)
        bottom_ax = ax

    if bottom_ax is not None:
        xticks = np.linspace(0, L - 1, 5, dtype=int)
        bottom_ax.set_xticks(xticks)
        bottom_ax.set_xticklabels([str(x) for x in xticks], fontsize=7.5)
        bottom_ax.set_xlabel("Sequence position", fontsize=8, labelpad=3)


# ── UMAP binding section ─────────────────────────────────────────────────────
def draw_umap_section(fig, gs_umap, coords_data, emb_data, models):
    """
    Draw binding scatter (circRNA only / circRNA + miRNA) × n_models
    inside the UMAP GridSpec slot.
    """
    n_models  = len(models)
    row_labels = ["circRNA only", "circRNA + miRNA"]
    coord_keys = ["coords_wo", "coords_wi"]

    gs_inner = GridSpecFromSubplotSpec(
        2, n_models,
        subplot_spec=gs_umap,
        hspace=0.30,
        wspace=0.18,
    )

    for ci, (mkey, mname, _, cache_name) in enumerate(models):
        if cache_name not in coords_data:
            for ri in range(2):
                ax = fig.add_subplot(gs_inner[ri, ci])
                ax.axis("off")
                if ri == 0:
                    ax.set_title(mname, fontsize=8, fontweight="bold", pad=4)
            continue

        if cache_name in emb_data:
            lbls_full = np.asarray(emb_data[cache_name]["lbls"])
        else:
            lbls_full = np.asarray(emb_data["CircMAC"]["lbls"])

        for ri, (coord_key, row_label) in enumerate(zip(coord_keys, row_labels)):
            ax = fig.add_subplot(gs_inner[ri, ci])

            if coord_key not in coords_data[cache_name]:
                ax.axis("off")
                continue

            coords = np.asarray(coords_data[cache_name][coord_key])
            n_pts  = len(coords)
            lbls   = lbls_full[:n_pts]

            order  = np.argsort(lbls)
            o_lbls = lbls[order]
            colors = np.where(o_lbls == 1, BIND_COLOR, NONBIND_COLOR)
            sizes  = np.where(o_lbls == 1, 18, 8)

            ax.scatter(coords[order, 0], coords[order, 1],
                       c=colors, s=sizes, alpha=0.80, linewidths=0, zorder=2)

            sep = sep_score(coords, lbls)
            sil = sil_score(coords, lbls)
            sep_str = f"{sep:.2f}" if not np.isnan(sep) else "n/a"
            sil_str = f"{sil:.2f}" if not np.isnan(sil) else "n/a"
            ax.text(0.5, -0.10, f"Sep={sep_str}  Sil={sil_str}",
                    ha="center", va="top", transform=ax.transAxes,
                    fontsize=6.5, color="#444444")

            hide_frame(ax)

            if ri == 0:
                ax.set_title(mname, fontsize=8, fontweight="bold", pad=4)
            if ci == 0:
                ax.set_ylabel(row_label, fontsize=8, fontweight="bold", labelpad=5)


# ── Main figure builder ──────────────────────────────────────────────────────
def make_case_figure(case, group_key, red_type="umap"):
    models    = GROUPS[group_key]
    n_models  = len(models)
    case_key  = case["key"]
    emb_key   = case["emb_key"]

    # Load prediction CSV
    df_all = pd.read_csv(case["csv"])
    df = df_all[
        (df_all["miRNA_ID"]  == case["mirna"])  &
        (df_all["isoform_ID"] == case["isoform"])
    ].reset_index(drop=True)

    if df.empty:
        raise ValueError(f"No rows for {case['label']} / {case['mirna']}")

    gt = df["ground_truth"].values
    L  = len(df)

    # Load UMAP/tSNE coords
    coords_data = load_pickle(CACHE / f"case_{emb_key}_{red_type}_coords.pkl")
    emb_data    = load_pickle(CACHE / f"case_{emb_key}_embeddings.pkl")

    # ── Layout ────────────────────────────────────────────────────────────────
    # Top: heatmap  (n_models+1 rows → height proportional)
    # Bottom: embedding (2 rows × n_models)
    #
    # Heatmap height: 1.35 (GT) + n_models * 1.0 → scaled
    n_hm_rows   = n_models + 1
    hm_h_ratio  = 1.35 + n_models * 1.0   # relative height for heatmap
    umap_h_ratio = 2.0 * 1.8              # 2 embedding rows, each ~1.8 in

    fig_w = max(3.2 * n_models, 14.0)
    fig_h = hm_h_ratio * 0.52 + umap_h_ratio + 1.4  # approximate inches

    fig = plt.figure(figsize=(fig_w, fig_h))

    total_h = hm_h_ratio + umap_h_ratio
    gs = GridSpec(
        2, 1,
        figure=fig,
        height_ratios=[hm_h_ratio, umap_h_ratio],
        hspace=0.45,
    )

    # Heatmap section
    draw_heatmap_section(fig, gs[0], df, gt, L, models)

    # UMAP section
    draw_umap_section(fig, gs[1], coords_data, emb_data, models)

    # ── Section labels ────────────────────────────────────────────────────────
    # Heatmap sub-title
    fig.text(0.5, 0.99, "(a) Nucleotide-level prediction heatmap",
             ha="center", va="top", fontsize=10, fontweight="bold",
             transform=fig.transFigure)

    red_label  = "UMAP" if red_type == "umap" else "t-SNE"
    fig.text(0.5, 0.52, f"(b) {red_label} embedding (colored by binding)",
             ha="center", va="top", fontsize=10, fontweight="bold",
             transform=fig.transFigure)

    # ── Legend for UMAP ───────────────────────────────────────────────────────
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=BIND_COLOR, markeredgecolor="none",
               markersize=7, label="Binding"),
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=NONBIND_COLOR, markeredgecolor="none",
               markersize=7, label="Non-binding"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 0.0))

    # ── Super title ───────────────────────────────────────────────────────────
    mirna_short = case["mirna"].replace("hsa-", "")
    if group_key == "encoder":
        group_label = "Encoder models"
    else:
        group_label = "Pretrained RNA-LMs (fine-tuned)"

    fig.suptitle(
        f"{case['label']}  ×  {mirna_short}  |  {group_label}",
        fontsize=12, fontweight="bold", y=1.015,
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    stem = f"fig8_{group_key}_heatmap_{red_type}_{case_key}"
    for ext in ["pdf", "png"]:
        p = OUT / f"{stem}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"Saved → {p}")
    plt.close(fig)


def main():
    for group_key in ["encoder", "pretrained"]:
        for case in CASES:
            for red_type in ["umap", "tsne"]:
                make_case_figure(case, group_key, red_type=red_type)


if __name__ == "__main__":
    main()
