#!/usr/bin/env python3
"""
Case Study — Circular Comparison Figures
  Generates two figures (encoder / pretrained), each showing all 3 cases.
  Style: Wedge-based rings — inner=GT (red/light), outer=model prediction (model color)
  BSJ arrow at top, model-colored title.

Output:
  fig_circular_encoder.{pdf,png}
  fig_circular_pretrained.{pdf,png}
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Wedge
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT  = Path(__file__).resolve().parent

# ── Case definitions ───────────────────────────────────────────────────────────
CASES = [
    dict(
        label   = 'circCDYL2\n(chr4)',
        csv     = ROOT / 'figures_claude/Fig5_Case_CDYL2/data_predictions.csv',
        isoform = 'chr4|84678168,84679116|84678259,84679242|-',
        mirna   = 'hsa-miR-449a',
    ),
    dict(
        label   = 'circMAPK1\n(chr22)',
        csv     = ROOT / 'figures_claude/Fig6_Case_MAPK1/data_predictions.csv',
        isoform = 'chr22|21799012,21805850,21807664|21799128,21806039,21807846|-',
        mirna   = 'hsa-miR-12119',
    ),
    dict(
        label   = 'circAPP\n(chr21)',
        csv     = ROOT / 'figures_claude/Fig7_Case_APP/data_predictions.csv',
        isoform = 'chr21|25954590,25955627,25975070,25975954,25982344|25954689,25955755,25975228,25976028,25982477|-',
        mirna   = 'hsa-miR-5001-3p',
    ),
]

# ── Model groups ───────────────────────────────────────────────────────────────
MODEL_COLORS = {
    'circmac':     '#FF7F0E',
    'mamba':       '#D62728',
    'lstm':        '#E377C2',
    'transformer': '#8C564B',
    'hymba':       '#BCBD22',
    'rnabert':     '#1F77B4',
    'rnaernie':    '#9467BD',
    'rnamsm':      '#2CA02C',
    'rnafm':       '#17BECF',
}

GROUPS = {
    'encoder':    [('CircMAC', 'pred_circmac'), ('Mamba', 'pred_mamba'),
                   ('LSTM', 'pred_lstm'), ('Transformer', 'pred_transformer'),
                   ('Hymba', 'pred_hymba')],
    'pretrained': [('CircMAC', 'pred_circmac'),
                   ('RNABert\n(fine-tuned)', 'pred_rnabert'),
                   ('RNAErnie\n(fine-tuned)', 'pred_rnaernie'),
                   ('RNA-MSM\n(fine-tuned)', 'pred_rnamsm'),
                   ('RNA-FM\n(fine-tuned)', 'pred_rnafm')],
}

BIND_COLOR   = '#D62728'
NOBIND_COLOR = '#F0F0F0'
BSJ_COLOR    = '#1F77B4'

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})


def draw_circle(ax, gt, pred, model_key, title=''):
    """Draw a single circular diagram (Wedge style)."""
    L = len(gt)
    color  = MODEL_COLORS.get(model_key, '#888888')
    angles = np.linspace(90, 90 - 360, L, endpoint=False)
    arc_w  = 360 / L

    R_IN, R_OUT     = 0.52, 0.74
    R_P_IN, R_P_OUT = R_OUT + 0.02, R_OUT + 0.12

    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.45)
    ax.set_aspect('equal')
    ax.axis('off')

    # ── GT inner ring ──────────────────────────────────────────────
    for i in range(L):
        fc = BIND_COLOR if gt[i] > 0.5 else NOBIND_COLOR
        a  = 1.0 if gt[i] > 0.5 else 0.30
        w  = Wedge((0, 0), R_OUT, angles[i] - arc_w * 0.45, angles[i] + arc_w * 0.45,
                   width=R_OUT - R_IN, facecolor=fc, edgecolor='white',
                   linewidth=0.08, alpha=a, zorder=2)
        ax.add_patch(w)

    # ── Prediction outer ring ──────────────────────────────────────
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        model_key, ['#f7f7f7', color])
    for i in range(L):
        w = Wedge((0, 0), R_P_OUT, angles[i] - arc_w * 0.45, angles[i] + arc_w * 0.45,
                  width=R_P_OUT - R_P_IN,
                  facecolor=cmap(float(pred[i]) * 0.85 + 0.10),
                  edgecolor='none', alpha=0.92, zorder=3)
        ax.add_patch(w)

    # ── BSJ marker ────────────────────────────────────────────────
    bsj_rad = np.radians(90)
    ax.annotate('',
        xy=(np.cos(bsj_rad) * (R_P_OUT + 0.04), np.sin(bsj_rad) * (R_P_OUT + 0.04)),
        xytext=(np.cos(bsj_rad) * (R_IN - 0.05), np.sin(bsj_rad) * (R_IN - 0.05)),
        arrowprops=dict(arrowstyle='->', color=BSJ_COLOR, lw=1.8))
    ax.text(0, R_P_OUT + 0.22, 'BSJ', ha='center', fontsize=7.5,
            color=BSJ_COLOR, fontweight='bold')

    ax.set_title(title, fontsize=9, fontweight='bold', color=color, pad=14)


def make_figure(group_key):
    models = GROUPS[group_key]
    n_models = len(models)      # 5
    n_cases  = len(CASES)       # 3

    # Layout: (n_models rows) × (n_cases cols) of small circles
    # +  1 legend row at bottom
    fig_w = 3.8 * n_cases
    fig_h = 3.5 * n_models + 0.6
    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = GridSpec(n_models, n_cases, figure=fig,
                  hspace=0.25, wspace=0.08)

    for ci, case in enumerate(CASES):
        df_all = pd.read_csv(case['csv'])
        df = df_all[(df_all['miRNA_ID'] == case['mirna']) &
                    (df_all['isoform_ID'] == case['isoform'])].reset_index(drop=True)
        gt   = df['ground_truth'].values

        for mi, (name, col) in enumerate(models):
            ax = fig.add_subplot(gs[mi, ci])
            pred = df[col].values
            model_key = col.replace('pred_', '').replace('_', '')
            draw_circle(ax, gt, pred, model_key, title=name)

            # Case label (top row only)
            if mi == 0:
                mirna_short = case['mirna'].replace('hsa-', '')
                ax.set_title(f'{name}', fontsize=9, fontweight='bold',
                             color=MODEL_COLORS.get(model_key, '#888'),
                             pad=14)
                # Add case header above first row
                ax.text(0, 1.58, f"{case['label'].replace(chr(10), '  ')}  ×  {mirna_short}",
                        ha='center', va='bottom', transform=ax.transAxes,
                        fontsize=9, fontweight='bold', color='#333333')

    # Shared legend at bottom
    from matplotlib.lines import Line2D
    handles = [
        mpatches.Patch(color=BIND_COLOR, alpha=0.85, label='GT binding (inner ring)'),
        mpatches.Patch(color=NOBIND_COLOR, alpha=0.6, ec='#aaa', label='GT non-binding'),
        Line2D([0], [0], color='#888', lw=5, alpha=0.7, label='Model prediction (outer ring)'),
        Line2D([0], [0], color=BSJ_COLOR, lw=1.5, label='BSJ position'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=4,
               fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 0.0))

    title_suffix = 'Encoder Models' if group_key == 'encoder' else 'Pretrained RNA LMs (fine-tuned)'
    fig.suptitle(f'Circular Binding Landscape — {title_suffix}',
                 fontsize=12, fontweight='bold', y=1.01)

    for ext in ['pdf', 'png']:
        p = OUT / f'fig_circular_{group_key}.{ext}'
        fig.savefig(p, dpi=200, bbox_inches='tight')
        print(f'Saved → {p}')
    plt.close(fig)


def main():
    make_figure('encoder')
    make_figure('pretrained')


if __name__ == '__main__':
    main()
