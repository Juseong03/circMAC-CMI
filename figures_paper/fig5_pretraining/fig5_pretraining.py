#!/usr/bin/env python3
"""
Fig 5: Pretraining Strategy Comparison (EXP2)
  Single-task strategies + combination strategies
  Missing entries (pairing, mlm_cpcl, mlm_cpcl_ssp) shown as placeholder.

Output: figures_paper/fig5_pretraining.{pdf,png}
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

# (display_label, exp_base, group)
# group: 'baseline' | 'single' | 'combo'
MODELS = [
    ('No PT',        'v2_pt_nopt',        'baseline'),
    ('MLM',          'v2_pt_mlm',         'single'),
    ('NTP',          'v2_pt_ntp',         'single'),
    ('SSP',          'v2_pt_ssp',         'single'),
    ('CPCL',         'v2_pt_cpcl',        'single'),
    ('Pairing',      'v2_pt_pairing',     'single'),
    ('All',          'v2_pt_all',         'combo'),
    ('MLM+NTP',      'v2_pt_mlm_ntp',     'combo'),
    ('MLM+SSP',      'v2_pt_mlm_ssp',     'combo'),
    ('MLM+CPCL',     'v2_pt_mlm_cpcl',    'combo'),
    ('MLM+CPCL+SSP', 'v2_pt_mlm_cpcl_ssp','combo'),
]

COLORS = {
    'baseline': '#AAAAAA',
    'single':   '#6B9CC7',
    'combo':    '#7CBB8F',
}

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         10,
    'axes.titlesize':    11,
    'axes.labelsize':    10,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
})


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


def main():
    data = [(label, exp, grp, load_f1(exp)) for label, exp, grp in MODELS]

    # Save data CSV
    rows = []
    for label, exp, grp, vals in data:
        for i, s in enumerate(SEEDS):
            if i < len(vals):
                rows.append({'model': label, 'group': grp, 'seed': s, 'f1_macro': vals[i]})
            else:
                rows.append({'model': label, 'group': grp, 'seed': s, 'f1_macro': None})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'fig5_pretraining_data.csv', index=False)
    summary = df.dropna().groupby('model').agg(
        f1_mean=('f1_macro','mean'), f1_std=('f1_macro','std')
    ).round(4)
    summary.to_csv(OUT / 'fig5_pretraining_summary.csv')
    print(summary.to_string())

    ylim = (0.67, 0.81)
    ylo  = ylim[0]

    fig, ax = plt.subplots(figsize=(11, 4.4))

    nopt_mean = None

    for i, (label, exp, grp, vals) in enumerate(data):
        color = COLORS[grp]

        if not vals:
            # placeholder — hatched empty bar
            ax.bar(i, ylim[1] - ylo - 0.002, bottom=ylo,
                   width=0.55, color='none', edgecolor='#bbbbbb',
                   linewidth=1.2, linestyle='--', zorder=2)
            ax.text(i, (ylo + ylim[1]) / 2, 'pending',
                    ha='center', va='center', fontsize=8,
                    color='#999999', style='italic', zorder=3)
            continue

        mean = np.mean(vals)
        std  = np.std(vals)

        if label == 'No PT':
            nopt_mean = mean

        ax.bar(i, mean - ylo, bottom=ylo, width=0.55,
               color=color, alpha=0.85, zorder=2, linewidth=0)
        ax.errorbar(i, mean, yerr=std, fmt='none', color='#222222',
                    capsize=4, capthick=1.4, elinewidth=1.4, zorder=4)
        ax.text(i, mean + std + (ylim[1] - ylim[0]) * 0.025,
                f'{mean:.3f}',
                ha='center', va='bottom', fontsize=8.5, fontweight='bold',
                color='#222222', zorder=6,
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85))

    # No PT baseline reference line
    if nopt_mean:
        ax.axhline(nopt_mean, color=COLORS['baseline'], linestyle=':',
                   linewidth=1.4, alpha=0.8, zorder=1,
                   label=f'No PT baseline ({nopt_mean:.3f})')

    # Separator line between single and combo
    ax.axvline(5.5, color='#cccccc', linewidth=1.0, linestyle='-', zorder=1)
    ax.text(2.5, ylim[1] - 0.003, 'Single-task',
            ha='center', va='top', fontsize=9, color='#555555')
    ax.text(8.5, ylim[1] - 0.003, 'Combination',
            ha='center', va='top', fontsize=9, color='#555555')

    ax.set_xticks(range(len(data)))
    ax.set_xticklabels([d[0] for d in data], rotation=25, ha='right')
    ax.set_ylabel('F1-macro')
    ax.set_ylim(*ylim)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title('Pretraining Strategy Comparison', fontsize=12, fontweight='bold')

    # Highlight best
    best_label = max([(d[0], np.mean(d[3])) for d in data if d[3]], key=lambda x: x[1])[0]
    for tick in ax.get_xticklabels():
        if tick.get_text() == best_label:
            tick.set_fontweight('bold')
            tick.set_color('#2B7A3A')

    # Legend
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor=COLORS['baseline'], label='No pretraining'),
        Patch(facecolor=COLORS['single'],   label='Single-task PT'),
        Patch(facecolor=COLORS['combo'],    label='Combination PT'),
    ]
    ax.legend(handles=legend_elems, loc='lower right', fontsize=9,
              framealpha=0.9, edgecolor='#cccccc')

    fig.tight_layout()
    for ext in ['pdf', 'png']:
        p = OUT / f'fig5_pretraining.{ext}'
        fig.savefig(p, dpi=200, bbox_inches='tight')
        print(f'Saved → {p}')
    plt.close(fig)


if __name__ == '__main__':
    main()
