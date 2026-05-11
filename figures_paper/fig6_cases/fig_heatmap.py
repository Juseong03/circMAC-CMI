#!/usr/bin/env python3
"""
Case Study — Heatmap Comparison Figures

Generates two figures:
  1) fig_heatmap_encoder.{pdf,png}
  2) fig_heatmap_pretrained.{pdf,png}

Style:
  - Ground-truth row: Reds
  - Model rows: white → model-specific color
  - BSJ positions: blue dashed lines

Notes:
  - Row labels are shown only in the first case panel to avoid overlap.
  - Model order follows baseline → stronger models → circMAC (ours).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from pathlib import Path


# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


# ── Case definitions ─────────────────────────────────────────────────────────
CASES = [
    dict(
        label="circCDYL2 (chr4)",
        csv=ROOT / "figures_claude/Fig5_Case_CDYL2/data_predictions.csv",
        isoform="chr4|84678168,84679116|84678259,84679242|-",
        mirna="hsa-miR-449a",
    ),
    dict(
        label="circMAPK1 (chr22)",
        csv=ROOT / "figures_claude/Fig6_Case_MAPK1/data_predictions.csv",
        isoform="chr22|21799012,21805850,21807664|21799128,21806039,21807846|-",
        mirna="hsa-miR-12119",
    ),
    dict(
        label="circAPP (chr21)",
        csv=ROOT / "figures_claude/Fig7_Case_APP/data_predictions.csv",
        isoform="chr21|25954590,25955627,25975070,25975954,25982344|25954689,25955755,25975228,25976028,25982477|-",
        mirna="hsa-miR-5001-3p",
    ),
]


# ── Colors ───────────────────────────────────────────────────────────────────
MODEL_COLORS = {
    "circmac": "#FF7F0E",

    "lstm": "#E377C2",
    "transformer": "#8C564B",
    "mamba": "#D62728",
    "hymba": "#BCBD22",

    "rnabert": "#1F77B4",
    "rnaernie": "#9467BD",
    "rnamsm": "#2CA02C",
    "rnafm": "#17BECF",
}

BSJ_COLOR = "#1F77B4"
GT_COLOR = "#8B0000"


# ── Model groups ─────────────────────────────────────────────────────────────
# Encoder order:
#   LSTM → Transformer → Mamba → Hymba → circMAC (ours)
#
# Pretrained RNA-LM order:
#   RNABERT → RNAErnie → RNAMSM → RNA-FM → circMAC (ours)
GROUPS = {
    'encoder': [
        ('lstm',        'LSTM',           'pred_lstm'),
        ('transformer', 'Transformer',    'pred_transformer'),
        ('mamba',       'Mamba',          'pred_mamba'),
        ('hymba',       'Hymba',          'pred_hymba'),
        ('circmac',     'circMAC (ours)', 'pred_circmac'),
    ],
    'pretrained': [
        ('rnabert',     'RNABERT\n(fine-tuned)',  'pred_rnabert'),
        ('rnaernie',    'RNAErnie\n(fine-tuned)', 'pred_rnaernie'),
        ('rnamsm',      'RNAMSM\n(fine-tuned)',   'pred_rnamsm'),
        ('rnafm',       'RNA-FM\n(fine-tuned)',   'pred_rnafm'),
        ('circmac',     'circMAC (ours)',          'pred_circmac'),
    ],
}


# ── Plot style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def set_row_label(ax, label, color, show=True):
    """
    Add row label only when show=True.

    We use ylabel instead of yticklabel because yticklabels easily overlap
    when heatmap rows are narrow.
    """
    ax.set_yticks([])

    if show:
        ax.set_ylabel(
            label,
            rotation=0,
            ha="right",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color=color,
            labelpad=36,
        )
    else:
        ax.set_ylabel("")


def draw_bsj_markers(ax, L, lw=1.0, alpha=0.55):
    """Draw BSJ boundary markers."""
    ax.axvline(-0.5, color=BSJ_COLOR, lw=lw, ls="--", alpha=alpha)
    ax.axvline(L - 0.5, color=BSJ_COLOR, lw=lw, ls="--", alpha=alpha)


def clean_axis(ax):
    """Remove all spines."""
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_case_heatmap(fig, gs_slot, df, gt, L, models, show_row_labels=True):
    """
    Draw one case panel:
      - one ground-truth row
      - multiple model prediction rows
    """
    n_models = len(models)

    gs_inner = GridSpecFromSubplotSpec(
        n_models + 1,
        1,
        subplot_spec=gs_slot,
        height_ratios=[1.35] + [1.0] * n_models,
        hspace=0.08,
    )

    # ── Ground-truth row ─────────────────────────────────────────────────────
    ax_gt = fig.add_subplot(gs_inner[0])

    ax_gt.imshow(
        gt[np.newaxis, :],
        aspect="auto",
        cmap="Reds",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )

    set_row_label(
        ax_gt,
        "Ground truth",
        color=GT_COLOR,
        show=show_row_labels,
    )

    ax_gt.set_xticks([])
    draw_bsj_markers(ax_gt, L, lw=1.4, alpha=0.75)
    clean_axis(ax_gt)

    # ── Model rows ───────────────────────────────────────────────────────────
    bottom_ax = None

    for ri, (mkey, mname, mcol) in enumerate(models):
        if mcol not in df.columns:
            raise ValueError(f"Missing prediction column: {mcol}")

        ax = fig.add_subplot(gs_inner[ri + 1])

        pred = df[mcol].values
        color = MODEL_COLORS.get(mkey, "#888888")

        cmap = mcolors.LinearSegmentedColormap.from_list(
            f"{mkey}_cmap",
            ["#f7f7f7", color],
        )

        ax.imshow(
            pred[np.newaxis, :],
            aspect="auto",
            cmap=cmap,
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )

        set_row_label(
            ax,
            mname,
            color=color,
            show=show_row_labels,
        )

        ax.set_xticks([])
        draw_bsj_markers(ax, L, lw=0.9, alpha=0.5)
        clean_axis(ax)

        bottom_ax = ax

    # Show x-axis only on the bottom row
    if bottom_ax is not None:
        xticks = np.linspace(0, L - 1, 5, dtype=int)
        bottom_ax.set_xticks(xticks)
        bottom_ax.set_xticklabels([str(x) for x in xticks], fontsize=8)
        bottom_ax.set_xlabel("Sequence position", fontsize=8.5, labelpad=4)

    return ax_gt


def load_case_dataframe(case):
    """Load and filter one case dataframe."""
    if not case["csv"].exists():
        raise FileNotFoundError(f"Missing CSV file: {case['csv']}")

    df_all = pd.read_csv(case["csv"])

    required_cols = {"miRNA_ID", "isoform_ID", "ground_truth"}
    missing_cols = required_cols - set(df_all.columns)

    if missing_cols:
        raise ValueError(
            f"{case['csv']} is missing required columns: {missing_cols}"
        )

    df = df_all[
        (df_all["miRNA_ID"] == case["mirna"])
        & (df_all["isoform_ID"] == case["isoform"])
    ].reset_index(drop=True)

    if df.empty:
        raise ValueError(
            f"No matching rows found for case: "
            f"{case['label']} / {case['mirna']}"
        )

    gt = df["ground_truth"].values
    L = len(df)

    return df, gt, L


def make_figure(group_key):
    """Generate one heatmap figure for a model group."""
    if group_key not in GROUPS:
        raise ValueError(f"Unknown group_key: {group_key}")

    models = GROUPS[group_key]
    n_cases = len(CASES)
    n_rows = len(models) + 1

    # Wider panel and larger height reduce label and title collisions.
    fig_w = 6.2 * n_cases
    fig_h = 0.82 * n_rows + 1.35

    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = GridSpec(
        1,
        n_cases,
        figure=fig,
        wspace=0.18,
    )

    for ci, case in enumerate(CASES):
        df, gt, L = load_case_dataframe(case)

        ax_gt = draw_case_heatmap(
            fig=fig,
            gs_slot=gs[ci],
            df=df,
            gt=gt,
            L=L,
            models=models,
            show_row_labels=(ci == 0),  # labels only in the first panel
        )

        mirna_short = case["mirna"].replace("hsa-", "")

        ax_gt.set_title(
            f"{case['label']}\n{mirna_short}",
            fontsize=9.5,
            fontweight="bold",
            pad=7,
        )

    if group_key == "encoder":
        title_suffix = "General encoder models"
    else:
        title_suffix = "Pretrained RNA language models"

    fig.suptitle(
        f"Nucleotide-level prediction heatmap — {title_suffix}",
        fontsize=12,
        fontweight="bold",
        y=1.035,
    )

    for ext in ["pdf", "png"]:
        out_path = OUT / f"fig_heatmap_{group_key}.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved → {out_path}")

    plt.close(fig)


def main():
    make_figure("encoder")
    make_figure("pretrained")


if __name__ == "__main__":
    main()