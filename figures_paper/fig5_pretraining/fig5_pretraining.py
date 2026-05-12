#!/usr/bin/env python3
"""
Fig 5: Pretraining Strategy Comparison (EXP2)
  Three panels: F1-macro / AUROC / AUPRC
  Groups: baseline | single-task | combination

Output: figures_paper/fig5_pretraining/fig5_pretraining.{pdf,png}
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

ROOT  = Path(__file__).resolve().parents[2]
LOGS  = ROOT / 'logs_0512'
OUT   = Path(__file__).resolve().parent
SEEDS = [1, 2, 3]

# (display_label, exp_base, group)
# group: 'baseline' | 'single' | 'combo'
MODELS = [
    ('No PT',        'v2_abl_full',        'baseline'),
    ('MLM',          'v2_pt_mlm',          'single'),
    ('SSP',          'v2_pt_ssp',          'single'),
    ('CPCL',         'v2_pt_cpcl',         'single'),
    ('Pairing',      'v2_pt_pairing',      'single'),
    ('MLM+SSP',      'v2_pt_mlm_ssp',      'combo'),
    ('MLM+CPCL',     'v2_pt_mlm_cpcl',     'combo'),
    ('MLM+CPCL+SSP', 'v2_pt_mlm_cpcl_ssp', 'combo'),
]

COLORS = {
    'baseline': '#AAAAAA',
    'single':   '#6B9CC7',
    'combo':    '#7CBB8F',
}

METRICS = [
    ('f1_macro', '(a) F1-macro', (0.67, 0.81)),
    ('roc_auc',  '(b) AUROC',    (0.87, 0.92)),
    ('auprc',    '(c) AUPRC',    (0.40, 0.57)),
]

# separator x position: between single-task and combination groups
SEP_SINGLE_COMBO = 4.5   # between Pairing and MLM+SSP

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         10,
    'axes.titlesize':    11,
    'axes.labelsize':    10,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'xtick.labelsize':   9,
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


def plot_panel(ax, data, metric, title, ylim):
    ylo     = ylim[0]
    y_range = ylim[1] - ylim[0]
    bar_w   = 0.55

    nopt_mean = None

    for i, (label, group, scores) in enumerate(data):
        vals = scores[metric]
        color = COLORS[group]

        if not vals:
            ax.bar(i, ylim[1] - ylo - 0.002, bottom=ylo,
                   width=bar_w, color='none', edgecolor='#bbbbbb',
                   linewidth=1.2, linestyle='--', zorder=2)
            ax.text(i, (ylo + ylim[1]) / 2, 'pending',
                    ha='center', va='center', fontsize=7.5,
                    color='#999999', style='italic', zorder=3)
            continue

        mean = np.mean(vals)
        std  = np.std(vals)

        if label == 'No PT':
            nopt_mean = mean

        ax.bar(i, mean - ylo, bottom=ylo, width=bar_w,
               color=color, alpha=0.85, zorder=2, linewidth=0)
        ax.errorbar(i, mean, yerr=std, fmt='none', color='#222222',
                    capsize=4, capthick=1.4, elinewidth=1.4, zorder=4)
        ax.text(i, mean + std + y_range * 0.025,
                f'{mean:.3f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold',
                color='#222222', zorder=6,
                bbox=dict(boxstyle='round,pad=0.13', fc='white', ec='none', alpha=0.85))

    # No PT baseline reference line
    if nopt_mean is not None:
        ax.axhline(nopt_mean, color=COLORS['baseline'], linestyle=':',
                   linewidth=1.4, alpha=0.8, zorder=1)

    # Separator between single-task and combination
    ax.axvline(SEP_SINGLE_COMBO, color='#cccccc', linewidth=1.0, zorder=1)

    ax.set_xticks(range(len(data)))
    ax.set_xticklabels([d[0] for d in data], rotation=30, ha='right')
    ax.set_ylim(*ylim)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=11, fontweight='bold')

    # Highlight best
    completed = [(d[0], np.mean(d[2][metric])) for d in data if d[2][metric]]
    if completed:
        best_label = max(completed, key=lambda x: x[1])[0]
        for tick in ax.get_xticklabels():
            if tick.get_text() == best_label:
                tick.set_fontweight('bold')
                tick.set_color('#2B7A3A')


def main():
    data = [(label, grp, load_scores(exp)) for label, exp, grp in MODELS]

    # Save CSV
    rows = []
    for (label, exp, grp), (_, _, scores) in zip(MODELS, data):
        max_len = max((len(scores[m]) for m in scores), default=0)
        for i in range(max(max_len, len(SEEDS))):
            seed = SEEDS[i] if i < len(SEEDS) else i + 1
            rows.append({
                'model':    label,
                'group':    grp,
                'seed':     seed,
                'f1_macro': scores['f1_macro'][i] if i < len(scores['f1_macro']) else float('nan'),
                'roc_auc':  scores['roc_auc'][i]  if i < len(scores['roc_auc'])  else float('nan'),
                'auprc':    scores['auprc'][i]     if i < len(scores['auprc'])    else float('nan'),
            })
    df = pd.DataFrame(rows)
    summary = df.groupby('model').agg(
        f1_mean=('f1_macro', 'mean'), f1_std=('f1_macro', 'std'),
        roc_mean=('roc_auc',  'mean'), roc_std=('roc_auc',  'std'),
        auprc_mean=('auprc',  'mean'), auprc_std=('auprc',  'std'),
    ).round(4)
    df.to_csv(OUT / 'fig5_pretraining_data.csv', index=False)
    summary.to_csv(OUT / 'fig5_pretraining_summary.csv')
    print(summary.to_string())

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    fig.suptitle('Pretraining Strategy Comparison', fontsize=12, fontweight='bold', y=1.01)

    for ax, (metric, title, ylim) in zip(axes, METRICS):
        plot_panel(ax, data, metric, title, ylim)

    # Group labels on first panel
    axes[0].text(2.5, METRICS[0][2][1] - 0.003, 'Single-task',
                 ha='center', va='top', fontsize=8.5, color='#555555')
    axes[0].text(6.0, METRICS[0][2][1] - 0.003, 'Combination',
                 ha='center', va='top', fontsize=8.5, color='#555555')

    legend_elems = [
        Patch(facecolor=COLORS['baseline'], label='CircMAC (No PT)'),
        Patch(facecolor=COLORS['single'],   label='Single-task PT'),
        Patch(facecolor=COLORS['combo'],    label='Combination PT'),
    ]
    fig.legend(handles=legend_elems, loc='upper center', ncol=3,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.0))

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)

    for ext in ['pdf', 'png']:
        p = OUT / f'fig5_pretraining.{ext}'
        fig.savefig(p, dpi=200, bbox_inches='tight')
        print(f'Saved → {p}')
    plt.close(fig)


if __name__ == '__main__':
    main()
