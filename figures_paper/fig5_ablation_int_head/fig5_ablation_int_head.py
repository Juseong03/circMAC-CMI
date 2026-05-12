#!/usr/bin/env python3
"""
Fig 4: Interaction Mechanism & Site Head Ablation (EXP5 + EXP6)
  Two groups × three metrics = 6 panels (2 rows × 3 cols):
    Row 1: Interaction (cross_attn vs concat vs elementwise)
    Row 2: Head (conv1d vs linear)
  Metrics: F1-macro / AUROC / AUPRC

Output: figures_paper/fig5_ablation_int_head.{pdf,png}
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ROOT  = Path(__file__).resolve().parents[2]
LOGS  = ROOT / 'logs_0512'
OUT   = Path(__file__).resolve().parent
SEEDS = [1, 2, 3]

PROPOSED = '#E05C2A'
BASELINE = '#6B9CC7'

INTERACTION = [
    ('Cross-Attn',  'v2_int_cross_attn',  True),
    ('Concat',      'v2_int_concat',       False),
    ('Elementwise', 'v2_int_elementwise',  False),
]
HEAD = [
    ('Conv1D', 'v2_head_conv1d', True),
    ('Linear', 'v2_head_linear', False),
]

METRICS = [
    ('f1_macro', 'F1-macro', {'interaction': (0.68, 0.80), 'head': (0.72, 0.78)}),
    ('roc_auc',  'AUROC',    {'interaction': (0.86, 0.92), 'head': (0.888, 0.910)}),
    ('auprc',    'AUPRC',    {'interaction': (0.39, 0.55), 'head': (0.490, 0.530)}),
]

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         10,
    'axes.titlesize':    11,
    'axes.labelsize':    10,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'xtick.labelsize':   10,
    'ytick.labelsize':   9,
    'pdf.fonttype':      42,
    'ps.fonttype':       42,
})


def load_scores(exp_base):
    scores = {'f1_macro': [], 'roc_auc': [], 'auprc': []}
    for s in SEEDS:
        p = LOGS / 'circmac' / f'{exp_base}_s{s}' / str(s) / 'training.json'
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        final = d.get('final', {})
        if not final:
            continue
        site_scores = list(final.values())[0]['scores']['sites']
        for metric in scores:
            if metric in site_scores:
                scores[metric].append(site_scores[metric])
    return scores


def plot_panel(ax, entries, group_key, metric, title, ylim):
    ylo     = ylim[0]
    y_range = ylim[1] - ylim[0]

    for i, (label, exp, is_best) in enumerate(entries):
        scores = load_scores(exp)
        vals   = scores[metric]
        if not vals:
            continue
        mean  = np.mean(vals)
        std   = np.std(vals)
        color = PROPOSED if is_best else BASELINE

        ax.bar(i, mean - ylo, bottom=ylo, width=0.50,
               color=color, alpha=0.85, zorder=2, linewidth=0)
        ax.errorbar(i, mean, yerr=std, fmt='none', color='#222222',
                    capsize=5, capthick=1.4, elinewidth=1.4, zorder=4)
        ax.text(i, mean + std + y_range * 0.03,
                f'{mean:.3f}',
                ha='center', va='bottom', fontsize=9.5, fontweight='bold',
                color='#222222', zorder=6,
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85))

    ax.set_xticks(range(len(entries)))
    ax.set_xticklabels([e[0] for e in entries])
    ax.set_ylim(*ylim)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=11, fontweight='bold')

    for tick in ax.get_xticklabels():
        for label, _, is_best in entries:
            if tick.get_text() == label and is_best:
                tick.set_fontweight('bold')
                tick.set_color(PROPOSED)


def main():
    # Save CSV
    rows = []
    for label, exp, _ in INTERACTION:
        scores = load_scores(exp)
        max_len = max(len(scores[m]) for m in scores)
        for i in range(max_len):
            seed = SEEDS[i] if i < len(SEEDS) else i + 1
            rows.append({
                'group': 'interaction', 'model': label, 'seed': seed,
                'f1_macro': scores['f1_macro'][i] if i < len(scores['f1_macro']) else float('nan'),
                'roc_auc':  scores['roc_auc'][i]  if i < len(scores['roc_auc'])  else float('nan'),
                'auprc':    scores['auprc'][i]     if i < len(scores['auprc'])    else float('nan'),
            })
    for label, exp, _ in HEAD:
        scores = load_scores(exp)
        max_len = max(len(scores[m]) for m in scores)
        for i in range(max_len):
            seed = SEEDS[i] if i < len(SEEDS) else i + 1
            rows.append({
                'group': 'head', 'model': label, 'seed': seed,
                'f1_macro': scores['f1_macro'][i] if i < len(scores['f1_macro']) else float('nan'),
                'roc_auc':  scores['roc_auc'][i]  if i < len(scores['roc_auc'])  else float('nan'),
                'auprc':    scores['auprc'][i]     if i < len(scores['auprc'])    else float('nan'),
            })
    df = pd.DataFrame(rows)
    summary = df.groupby(['group', 'model']).agg(
        f1_mean=('f1_macro', 'mean'), f1_std=('f1_macro', 'std'),
        roc_mean=('roc_auc',  'mean'), roc_std=('roc_auc',  'std'),
        auprc_mean=('auprc',  'mean'), auprc_std=('auprc',  'std'),
    ).round(4)
    df.to_csv(OUT / 'fig5_ablation_int_head_data.csv', index=False)
    summary.to_csv(OUT / 'fig5_ablation_int_head_summary.csv')
    print(summary.to_string())

    # 2 rows (interaction, head) × 3 cols (metrics)
    fig, axes = plt.subplots(
        2, 3,
        figsize=(12, 8.0),
        gridspec_kw={'width_ratios': [3, 3, 3], 'hspace': 0.55, 'wspace': 0.30},
    )
    fig.suptitle('Interaction & Head Ablation', fontsize=12, fontweight='bold', y=1.01)

    row_groups = [
        (0, INTERACTION, 'interaction', '(a) Interaction Mechanism'),
        (1, HEAD,        'head',        '(b) Site Prediction Head'),
    ]

    for row, entries, group_key, group_title in row_groups:
        for col, (metric, metric_label, ylims) in enumerate(METRICS):
            ax    = axes[row, col]
            ylim  = ylims[group_key]
            title = f'{group_title}\n{metric_label}' if col == 0 else metric_label
            plot_panel(ax, entries, group_key, metric, title, ylim)

    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor=PROPOSED, label='Proposed (ours)'),
        Patch(facecolor=BASELINE, label='Baseline'),
    ]
    fig.legend(handles=legend_elems, loc='upper center', ncol=2,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.0))

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.08)

    for ext in ['pdf', 'png']:
        p = OUT / f'fig5_ablation_int_head.{ext}'
        fig.savefig(p, dpi=200, bbox_inches='tight')
        print(f'Saved → {p}')
    plt.close(fig)


if __name__ == '__main__':
    main()
