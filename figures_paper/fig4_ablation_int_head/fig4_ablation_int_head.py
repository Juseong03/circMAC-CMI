#!/usr/bin/env python3
"""
Fig 4: Interaction Mechanism & Site Head Ablation (EXP5 + EXP6)
  Two panels side by side:
    (a) Interaction: cross_attn vs concat vs elementwise
    (b) Head: conv1d vs linear

Output: figures_paper/fig4_ablation_int_head.{pdf,png}
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ROOT  = Path(__file__).resolve().parents[2]
LOGS  = ROOT / 'logs_0510'
OUT   = Path(__file__).resolve().parent
SEEDS = [1, 2, 3]

PROPOSED = '#E05C2A'
BASELINE = '#6B9CC7'

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         10,
    'axes.titlesize':    11,
    'axes.labelsize':    10,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'xtick.labelsize':   10,
    'ytick.labelsize':   9,
})

INTERACTION = [
    ('Cross-Attn',   'v2_int_cross_attn',  True),
    ('Concat',       'v2_int_concat',       False),
    ('Elementwise',  'v2_int_elementwise',  False),
]
HEAD = [
    ('Conv1D',  'v2_head_conv1d',  True),
    ('Linear',  'v2_head_linear',  False),
]


def load_f1(exp_base):
    vals = []
    for s in SEEDS:
        p = LOGS / 'circmac' / f'{exp_base}_s{s}' / str(s) / 'training.json'
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        final = d.get('final', {})
        if final:
            vals.append(list(final.values())[0]['scores']['sites']['f1_macro'])
    return vals


def plot_bars(ax, entries, ylim, title):
    ylo = ylim[0]
    best_mean = None

    for i, (label, exp, is_best) in enumerate(entries):
        vals = load_f1(exp)
        if not vals:
            continue
        mean = np.mean(vals)
        std  = np.std(vals)
        color = PROPOSED if is_best else BASELINE

        if is_best:
            best_mean = mean

        ax.bar(i, mean - ylo, bottom=ylo, width=0.50,
               color=color, alpha=0.85, zorder=2, linewidth=0)
        ax.errorbar(i, mean, yerr=std, fmt='none', color='#222222',
                    capsize=5, capthick=1.4, elinewidth=1.4, zorder=4)
        ax.text(i, mean + std + (ylim[1] - ylim[0]) * 0.03,
                f'{mean:.3f}',
                ha='center', va='bottom', fontsize=9.5, fontweight='bold',
                color='#222222', zorder=6,
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85))

    if best_mean:
        ax.axhline(best_mean, color=PROPOSED, linestyle='--',
                   linewidth=1.2, alpha=0.55, zorder=1)

    ax.set_xticks(range(len(entries)))
    ax.set_xticklabels([e[0] for e in entries])
    ax.set_ylim(*ylim)
    ax.set_ylabel('F1-macro')
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=11, fontweight='bold')

    for tick in ax.get_xticklabels():
        for label, _, is_best in entries:
            if tick.get_text() == label and is_best:
                tick.set_fontweight('bold')
                tick.set_color(PROPOSED)


def main():
    # Save data CSV
    rows = []
    for label, exp, _ in INTERACTION:
        for i, s in enumerate(SEEDS):
            vals = load_f1(exp)
            if i < len(vals):
                rows.append({'group': 'interaction', 'model': label, 'seed': s, 'f1_macro': vals[i]})
    for label, exp, _ in HEAD:
        for i, s in enumerate(SEEDS):
            vals = load_f1(exp)
            if i < len(vals):
                rows.append({'group': 'head', 'model': label, 'seed': s, 'f1_macro': vals[i]})
    df = pd.DataFrame(rows)
    summary = df.groupby(['group','model']).agg(
        f1_mean=('f1_macro','mean'), f1_std=('f1_macro','std')
    ).round(4)
    df.to_csv(OUT / 'fig4_ablation_int_head_data.csv', index=False)
    summary.to_csv(OUT / 'fig4_ablation_int_head_summary.csv')
    print(summary.to_string())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 4.2),
                                    gridspec_kw={'width_ratios': [3, 2]})

    plot_bars(ax1, INTERACTION, ylim=(0.68, 0.80), title='(a) Interaction Mechanism')
    plot_bars(ax2, HEAD,        ylim=(0.72, 0.78), title='(b) Site Prediction Head')

    fig.suptitle('Interaction & Head Ablation', fontsize=12, fontweight='bold', y=1.01)
    fig.tight_layout()

    for ext in ['pdf', 'png']:
        p = OUT / f'fig4_ablation_int_head.{ext}'
        fig.savefig(p, dpi=200, bbox_inches='tight')
        print(f'Saved → {p}')
    plt.close(fig)


if __name__ == '__main__':
    main()
