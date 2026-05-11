#!/usr/bin/env python3
"""
Fig 3: CircMAC Module Ablation Study (EXP4)
  Single panel: F1-macro mean ± std, sorted by performance
  Color: Full model = orange, remove-branch = red family, single-branch = blue family

Output: figures_paper/fig3_ablation_modules.{pdf,png}
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

# (display_label, exp_suffix, group)
# group: 'full' | 'remove' | 'single'
MODELS = [
    ('Full',         'v2_abl_full',         'full'),
    ('No Circ Bias', 'v2_abl_no_circ_bias', 'remove'),
    ('w/o Attn',     'v2_abl_no_attn',      'remove'),
    ('w/o Conv',     'v2_abl_no_conv',       'remove'),
    ('w/o Mamba',    'v2_abl_no_mamba',     'remove'),
    ('Mamba Only',   'v2_abl_mamba_only',   'single'),
    ('CNN Only',     'v2_abl_cnn_only',     'single'),
    ('Attn Only',    'v2_abl_attn_only',    'single'),
]

COLORS = {
    'full':   '#E05C2A',
    'remove': '#4E9AC7',
    'single': '#8DC8A0',
}

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         10,
    'axes.titlesize':    11,
    'axes.labelsize':    10,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'xtick.labelsize':   9.5,
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
    data = []
    for label, exp, group in MODELS:
        vals = load_f1(exp)
        data.append((label, group, vals))

    # Save data CSV
    rows = []
    for label, group, vals in data:
        for i, s in enumerate(SEEDS):
            if i < len(vals):
                rows.append({'model': label, 'group': group, 'seed': s, 'f1_macro': vals[i]})
    df = pd.DataFrame(rows)
    summary = df.groupby('model').agg(
        f1_mean=('f1_macro','mean'), f1_std=('f1_macro','std')
    ).round(4)
    df.to_csv(OUT / 'fig3_ablation_modules_data.csv', index=False)
    summary.to_csv(OUT / 'fig3_ablation_modules_summary.csv')
    print(summary.to_string())

    ylim = (0.60, 0.81)
    ylo  = ylim[0]

    fig, ax = plt.subplots(figsize=(9, 4.2))

    bar_w = 0.60
    full_mean = None

    for i, (label, group, vals) in enumerate(data):
        if not vals:
            continue
        mean = np.mean(vals)
        std  = np.std(vals)
        color = COLORS[group]

        if group == 'full':
            full_mean = mean

        ax.bar(i, mean - ylo, bottom=ylo, width=bar_w,
               color=color, alpha=0.85, zorder=2, linewidth=0)
        ax.errorbar(i, mean, yerr=std, fmt='none', color='#222222',
                    capsize=4, capthick=1.4, elinewidth=1.4, zorder=4)
        ax.text(i, mean + std + (ylim[1] - ylim[0]) * 0.025,
                f'{mean:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                color='#222222', zorder=6,
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85))

    # Dashed reference line at Full model mean
    if full_mean:
        ax.axhline(full_mean, color=COLORS['full'], linestyle='--',
                   linewidth=1.2, alpha=0.6, zorder=1)

    ax.set_xticks(range(len(data)))
    ax.set_xticklabels([d[0] for d in data], rotation=20, ha='right')
    ax.set_ylabel('F1-macro')
    ax.set_ylim(*ylim)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title('CircMAC Module Ablation', fontsize=12, fontweight='bold')

    # Bold + color for Full
    for tick in ax.get_xticklabels():
        if tick.get_text() == 'Full':
            tick.set_fontweight('bold')
            tick.set_color(COLORS['full'])

    # Legend
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor=COLORS['full'],   label='Full model'),
        Patch(facecolor=COLORS['remove'], label='Remove branch'),
        Patch(facecolor=COLORS['single'], label='Single branch only'),
    ]
    ax.legend(handles=legend_elems, loc='lower right', fontsize=9,
              framealpha=0.9, edgecolor='#cccccc')

    fig.tight_layout()
    for ext in ['pdf', 'png']:
        p = OUT / f'fig3_ablation_modules.{ext}'
        fig.savefig(p, dpi=200, bbox_inches='tight')
        print(f'Saved → {p}')
    plt.close(fig)


if __name__ == '__main__':
    main()
