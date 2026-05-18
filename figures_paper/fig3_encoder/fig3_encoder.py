#!/usr/bin/env python3
"""
Fig 3: Encoder Architecture Comparison
  LSTM / Transformer / Mamba / Hymba / CircMAC (no pretraining)
  Three panels: F1-macro / AUROC / AUPRC

CircMAC x-tick 라벨을 검정색으로 표시하도록 수정.

Output: figures_paper/fig3_encoder/fig3_encoder.{pdf,png}
"""

import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

ROOT  = Path(__file__).resolve().parents[2]
OUT   = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)
DATA_CSV = OUT / 'fig3_encoder_data.csv'

MODELS = [
    ('LSTM',        ),
    ('Transformer', ),
    ('Mamba',       ),
    ('Hymba',       ),
    ('CircMAC',     ),
]

PROPOSED_COLOR = '#E05C2A'
BASELINE_COLOR = '#6B9CC7'

METRICS = [
    ('f1_macro', '(a) F1-macro', (0.57, 0.80)),
    ('roc_auc',  '(b) AUROC',    (0.70, 0.95)),
    ('auprc',    '(c) AUPRC',    (0.15, 0.58)),
]

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size':   10,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# ------------------------------------------------------------------
def load_scores(label):
    """CSV에서 label(model) 기준으로 metric 점수 로드"""
    df = pd.read_csv(DATA_CSV)
    rows = df[df['model'] == label]
    return {m: rows[m].dropna().tolist() for m in ('f1_macro', 'roc_auc', 'auprc')}

# ------------------------------------------------------------------
def plot_panel(ax, data, metric, title, ylim):
    """data: list of (label, color, vals)"""
    y0, y_range = ylim[0], ylim[1]-ylim[0]
    bar_w = 0.55

    for i, (label, color, vals) in enumerate(data):
        if not vals:
            continue
        mean, std = np.mean(vals), np.std(vals)

        ax.bar(i, mean-y0, bottom=y0, width=bar_w,
               color=color, alpha=0.85, linewidth=0, zorder=2)

        if len(vals) > 1:
            ax.errorbar(i, mean, yerr=std, fmt='none', color='#222',
                        capsize=4, capthick=1.4, elinewidth=1.4, zorder=4)

        ax.text(i, mean+std+y_range*0.025, f'{mean:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                color='#222', bbox=dict(boxstyle='round,pad=0.15',
                fc='white', ec='none', alpha=0.85), zorder=6)

    ax.set_xticks(range(len(data)))
    ax.set_xticklabels([d[0] for d in data], rotation=20, ha='right',
                       color='#000000')                    # ← 모두 검정색
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)

    # CircMAC만 굵게 강조 (색상 변경 X)
    for tick in ax.get_xticklabels():
        if tick.get_text() == 'CircMAC':
            tick.set_fontweight('bold')

# ------------------------------------------------------------------
def main():
    all_data = []

    for (label,) in MODELS:
        color  = PROPOSED_COLOR if label == 'CircMAC' else BASELINE_COLOR
        scores = load_scores(label)
        all_data.append((label, color, scores))

    fig, axes = plt.subplots(1,3,figsize=(11,4.2))
    fig.suptitle('Encoder Architecture Comparison',
                 fontsize=12, fontweight='bold', y=1.01)

    for ax,(metric,title,ylim) in zip(axes,METRICS):
        panel_data=[(lbl,col,sc[metric]) for lbl,col,sc in all_data]
        plot_panel(ax, panel_data, metric, title, ylim)

    # 범례(그대로 유지)
    fig.legend(handles=[
        Patch(facecolor=PROPOSED_COLOR, label='CircMAC (ours)'),
        Patch(facecolor=BASELINE_COLOR, label='Baseline encoder')],
        loc='upper center', ncol=2, fontsize=9, frameon=False,
        bbox_to_anchor=(0.5, 0.0))

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)

    out_dir = OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf','png'):
        path = out_dir/f'fig3_encoder.{ext}'
        fig.savefig(path, dpi=200, bbox_inches='tight')
        print('Saved →', path)
    plt.close(fig)

if __name__ == '__main__':
    main()