#!/usr/bin/env python3
"""
fig_multisite.py — Multi-Site Binding Case Study Figure

Shows CircMAC's ability to detect multiple binding sites scattered across circRNA.

Layout: 2 rows × 4 columns
  Row 1: Circular plots (ground truth + CircMAC probability)
  Row 2: Linear heatmap comparison (GT vs CircMAC vs LSTM vs Mamba vs Hymba)

Cases selected (diverse multi-site patterns):
  - RPUSD4  (3 clusters, BSJ-adjacent)
  - MGA     (2 clusters, wide spread)
  - DONSON  (2 clusters, opposite ends)
  - FANCA   (2 clusters, short circRNA)

Output: fig_multisite.{pdf,png}

Usage:
    python figures_paper/fig_multisite/fig_multisite.py
    python figures_paper/fig_multisite/fig_multisite.py --csv path/to/data_predictions.csv
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT  = Path(__file__).resolve().parent

# ── Representative cases ──────────────────────────────────────────────────────
CASES = [
    dict(label='circRPUSD4',  gene='RPUSD4',     isoform_prefix='chr11|126203345',
         mirna='hsa-miR-5008-3p',  chr_label='chr11'),
    dict(label='circMGA',     gene='MGA',         isoform_prefix='chr15|41696075',
         mirna='hsa-miR-3978',     chr_label='chr15'),
    dict(label='circDONSON',  gene='DONSON',      isoform_prefix='chr21|33945',
         mirna='hsa-miR-296-3p',   chr_label='chr21'),
    dict(label='circFANCA',   gene='FANCA',       isoform_prefix='chr16|89782859',
         mirna='hsa-miR-6858-5p',  chr_label='chr16'),
]

# ── Colors ────────────────────────────────────────────────────────────────────
PRED_COLOR   = '#FF7F0E'   # CircMAC
BIND_COLOR   = '#D62728'   # Ground-truth binding
NOBIND_COLOR = '#F0F0F0'
BSJ_COLOR    = '#1F77B4'

MODEL_COLORS = {
    'circmac':     '#FF7F0E',
    'lstm':        '#E377C2',
    'transformer': '#8C564B',
    'mamba':       '#D62728',
    'hymba':       '#BCBD22',
    'rnamsm_ft':   '#2CA02C',
    'rnafm_ft':    '#17BECF',
}

HEATMAP_MODELS = [
    ('ground_truth',  'GT',          '#8B0000'),
    ('pred_circmac',  'CircMAC',     '#FF7F0E'),
    ('pred_mamba',    'Mamba',       '#D62728'),
    ('pred_lstm',     'LSTM',        '#E377C2'),
    ('pred_hymba',    'Hymba',       '#BCBD22'),
    ('pred_transformer', 'Transformer', '#8C564B'),
]

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8})


def get_case_df(df_all: pd.DataFrame, case: dict):
    """Filter df to a single case (isoform + miRNA)."""
    mask = (
        df_all['isoform_ID'].str.startswith(case['isoform_prefix']) &
        (df_all['miRNA_ID'] == case['mirna'])
    )
    df = df_all[mask].sort_values('position').reset_index(drop=True)
    if df.empty:
        # fallback: gene_name match
        mask2 = (
            (df_all.get('gene_name', pd.Series(dtype=str)) == case['gene']) &
            (df_all['miRNA_ID'] == case['mirna'])
        )
        df = df_all[mask2].sort_values('position').reset_index(drop=True)
    return df


def plot_circular(ax, df: pd.DataFrame, case: dict, show_legend: bool = False):
    """Polar plot: GT ring (outer) + CircMAC probability bars (inner)."""
    L     = len(df)
    theta = np.linspace(0, 2 * np.pi, L, endpoint=False)
    width = 2 * np.pi / L * 0.95

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines['polar'].set_visible(False)

    # CircMAC probability bars
    pred_col = 'pred_circmac' if 'pred_circmac' in df.columns else None
    if pred_col:
        pred = df[pred_col].fillna(0).values
        ax.bar(theta, pred * 0.78, width=width, bottom=0.0,
               align='edge', color=PRED_COLOR, alpha=0.75, edgecolor='none',
               label='CircMAC prob.')

    # Ground truth ring
    gt      = df['ground_truth'].values
    gt_mask = gt > 0.5
    ax.bar(theta, np.full(L, 0.17), width=width, bottom=0.83,
           align='edge',
           color=np.where(gt_mask, BIND_COLOR, NOBIND_COLOR),
           alpha=0.85, edgecolor='#CCCCCC', linewidth=0.2,
           label='Ground truth')

    # BSJ markers (dashed lines at θ=0 and θ≈2π)
    for bsj_theta in [0, 2 * np.pi * (L - 1) / L]:
        ax.axvline(bsj_theta + np.pi / 2, color=BSJ_COLOR,
                   linestyle='--', linewidth=0.8, alpha=0.7)

    # Cluster count annotation
    diff = np.diff(np.concatenate([[0], gt, [0]]))
    n_clusters = int((diff == 1).sum())
    n_sites    = int(gt.sum())
    ax.set_title(f'{case["label"]}\n'
                 f'L={L}, {n_sites} sites ({n_clusters} clusters)',
                 fontsize=7.5, pad=4, fontweight='bold')

    if show_legend:
        ax.legend(loc='upper right', bbox_to_anchor=(1.45, 1.1),
                  fontsize=6.5, frameon=True)


def plot_heatmap(ax, df: pd.DataFrame, case: dict, show_ylabel: bool = True):
    """Linear heatmap: rows = models, columns = positions."""
    L = len(df)

    # Build data matrix
    rows_data = []
    row_labels = []
    row_colors = []
    for col, label, color in HEATMAP_MODELS:
        if col not in df.columns:
            continue
        rows_data.append(df[col].fillna(0).values)
        row_labels.append(label)
        row_colors.append(color)

    if not rows_data:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes)
        return

    mat = np.stack(rows_data)  # (n_models, L)

    # Show up to 800 nt (zoom window around binding region)
    gt = df['ground_truth'].values
    if gt.sum() > 0:
        first = max(0, np.where(gt == 1)[0][0] - 30)
        last  = min(L, np.where(gt == 1)[0][-1] + 30)
        # If very spread out, show full sequence
        if last - first < 100 or L <= 300:
            first, last = 0, L
    else:
        first, last = 0, min(L, 600)

    mat_show = mat[:, first:last]
    L_show   = last - first

    im = ax.imshow(mat_show, aspect='auto', cmap='YlOrRd',
                   interpolation='nearest', vmin=0, vmax=1)

    # Y-axis labels with colored dots
    if show_ylabel:
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=7)
        for tick, color in zip(ax.get_yticklabels(), row_colors):
            tick.set_color(color)
            if color == '#8B0000':
                tick.set_fontweight('bold')
    else:
        ax.set_yticks([])

    # X-axis: show position ticks
    n_ticks = min(6, L_show)
    tick_pos = np.linspace(0, L_show - 1, n_ticks, dtype=int)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([str(first + p) for p in tick_pos], fontsize=6.5)
    ax.set_xlabel('Position (nt)', fontsize=7)

    # Highlight binding regions
    for pos in range(L_show):
        if gt[first + pos] == 1:
            ax.axvline(pos, color='#D62728', alpha=0.08, linewidth=0.5)

    # Mark cluster boundaries
    diff = np.diff(np.concatenate([[0], gt[first:last], [0]]))
    for pos in np.where(diff == 1)[0]:
        ax.axvline(pos - 0.5, color='#D62728', linewidth=1.0, alpha=0.6,
                   linestyle='-')
    for pos in np.where(diff == -1)[0]:
        ax.axvline(pos - 0.5, color='#D62728', linewidth=1.0, alpha=0.6,
                   linestyle='-')

    return im


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default=str(OUT / 'data_predictions.csv'))
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f'[ERROR] {csv_path} not found.')
        print('  Run first: python scripts/extract_multisite_predictions.py --device 0')
        return

    df_all = pd.read_csv(csv_path)
    print(f'Loaded {len(df_all)} rows from {csv_path.name}')

    n_cases = len(CASES)

    # ── Layout: 2 rows × n_cases cols ────────────────────────────────────────
    fig = plt.figure(figsize=(3.5 * n_cases, 6.5))
    gs  = gridspec.GridSpec(2, n_cases, figure=fig,
                            hspace=0.45, wspace=0.35,
                            height_ratios=[1.1, 0.9])

    axes_circ = [fig.add_subplot(gs[0, c], projection='polar') for c in range(n_cases)]
    axes_heat = [fig.add_subplot(gs[1, c]) for c in range(n_cases)]

    last_im = None
    for c, case in enumerate(CASES):
        df_case = get_case_df(df_all, case)
        if df_case.empty:
            print(f'  [WARN] No data for {case["label"]} — skipping')
            axes_circ[c].set_visible(False)
            axes_heat[c].set_visible(False)
            continue

        print(f'  [{case["label"]}] {len(df_case)} positions, '
              f'n_sites={int(df_case["ground_truth"].sum())}')

        plot_circular(axes_circ[c], df_case, case, show_legend=(c == n_cases - 1))
        im = plot_heatmap(axes_heat[c], df_case, case, show_ylabel=(c == 0))
        if im is not None:
            last_im = im

    # Colorbar
    if last_im is not None:
        cbar_ax = fig.add_axes([0.92, 0.08, 0.015, 0.30])
        cb = fig.colorbar(last_im, cax=cbar_ax)
        cb.set_label('Probability', fontsize=7)
        cb.ax.tick_params(labelsize=6)

    # Panel labels
    for c, ax in enumerate(axes_circ):
        ax.annotate(chr(ord('A') + c), xy=(-0.08, 1.08),
                    xycoords='axes fraction', fontsize=10, fontweight='bold')
    for c, ax in enumerate(axes_heat):
        ax.annotate(chr(ord('A') + n_cases + c), xy=(-0.08 if c == 0 else -0.05, 1.08),
                    xycoords='axes fraction', fontsize=10, fontweight='bold')

    # Super title
    fig.suptitle('Multi-site binding case study: CircMAC detects dispersed binding sites',
                 fontsize=9, y=1.01, fontweight='bold')

    # Legend for circular plots
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor=BIND_COLOR,   label='Binding site (GT)'),
        Patch(facecolor=PRED_COLOR,   alpha=0.75, label='CircMAC probability'),
        Line2D([0], [0], color=BSJ_COLOR, linestyle='--', label='BSJ boundary'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3,
               fontsize=7, bbox_to_anchor=(0.5, 1.04), frameon=True)

    for fmt in ['pdf', 'png']:
        out_path = OUT / f'fig_multisite.{fmt}'
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f'Saved → {out_path}')

    plt.close(fig)
    print('Done.')


if __name__ == '__main__':
    main()
