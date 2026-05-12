#!/usr/bin/env python3
"""
Fig 3: CircMAC Module Ablation Study (EXP4)
  Three panels: F1-macro / AUROC / AUPRC  mean ± std
  Color: Full model = orange, remove-branch = blue, single-branch = green

Output: figures_paper/fig4_ablation_modules.{pdf,png}
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

# (display_label, exp_suffix, group)
# group: 'full' | 'remove' | 'single'
MODELS = [
    ('Full',        'v2_abl_full',       'full'),
    ('w/o Attn',    'v2_abl_no_attn',    'remove'),
    ('w/o Conv',    'v2_abl_no_conv',    'remove'),
    ('w/o Mamba',   'v2_abl_no_mamba',   'remove'),
    ('Mamba Only',  'v2_abl_mamba_only', 'single'),
    ('CNN Only',    'v2_abl_cnn_only',   'single'),
    ('Attn Only',   'v2_abl_attn_only',  'single'),
]

COLORS = {
    'full':   '#E05C2A',
    'remove': '#4E9AC7',
    'single': '#8DC8A0',
}

METRICS = [
    ('f1_macro', '(a) F1-macro', (0.60, 0.81)),
    ('roc_auc',  '(b) AUROC',    (0.77, 0.92)),
    ('auprc',    '(c) AUPRC',    (0.22, 0.55)),
]

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
    bar_w   = 0.58

    for i, (label, group, scores) in enumerate(data):
        vals = scores[metric]
        if not vals:
            continue
        mean  = np.mean(vals)
        std   = np.std(vals)
        color = COLORS[group]

        ax.bar(i, mean - ylo, bottom=ylo, width=bar_w,
               color=color, alpha=0.85, zorder=2, linewidth=0)
        ax.errorbar(i, mean, yerr=std, fmt='none', color='#222222',
                    capsize=4, capthick=1.4, elinewidth=1.4, zorder=4)
        ax.text(i, mean + std + y_range * 0.025,
                f'{mean:.3f}',
                ha='center', va='bottom', fontsize=8.5, fontweight='bold',
                color='#222222', zorder=6,
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85))

    ax.set_xticks(range(len(data)))
    ax.set_xticklabels([d[0] for d in data], rotation=25, ha='right')
    ax.set_ylim(*ylim)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=11, fontweight='bold')

    for tick in ax.get_xticklabels():
        if tick.get_text() == 'Full':
            tick.set_fontweight('bold')
            tick.set_color(COLORS['full'])


def main():
    data = []
    for label, exp, group in MODELS:
        scores = load_scores(exp)
        data.append((label, group, scores))

    # Save CSV
    rows = []
    for label, group, scores in data:
        max_len = max((len(scores[m]) for m in scores), default=0)
        for i in range(max_len):
            seed = SEEDS[i] if i < len(SEEDS) else i + 1
            rows.append({
                'model':    label,
                'group':    group,
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
    df.to_csv(OUT / 'fig4_ablation_modules_data.csv', index=False)
    summary.to_csv(OUT / 'fig4_ablation_modules_summary.csv')
    print(summary.to_string())

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    fig.suptitle('CircMAC Module Ablation', fontsize=12, fontweight='bold', y=1.01)

    for ax, (metric, title, ylim) in zip(axes, METRICS):
        plot_panel(ax, data, metric, title, ylim)

    legend_elems = [
        Patch(facecolor=COLORS['full'],   label='Full model'),
        Patch(facecolor=COLORS['remove'], label='Remove branch'),
        Patch(facecolor=COLORS['single'], label='Single branch only'),
    ]
    fig.legend(handles=legend_elems, loc='upper center', ncol=3,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.0))

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.20)

    for ext in ['pdf', 'png']:
        p = OUT / f'fig4_ablation_modules.{ext}'
        fig.savefig(p, dpi=200, bbox_inches='tight')
        print(f'Saved → {p}')
    plt.close(fig)


if __name__ == '__main__':
    main()
