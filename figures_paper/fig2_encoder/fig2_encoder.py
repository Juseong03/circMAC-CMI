#!/usr/bin/env python3
"""
Fig 2 (Ablation): Encoder Architecture Comparison
  LSTM / Transformer / Mamba / Hymba / CircMAC (no pretraining)
  Three panels: F1-macro / AUROC / AUPRC

Note: only seed 1 is available for each model (no error bars).

Output: figures_paper/fig2_encoder/fig2_encoder.{pdf,png}
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
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [1, 2, 3]

MODELS = [
    ('LSTM',        'lstm',        'v2_enc_lstm'),
    ('Transformer', 'transformer', 'v2_enc_transformer'),
    ('Mamba',       'mamba',       'v2_enc_mamba'),
    ('Hymba',       'hymba',       'v2_enc_hymba'),
    ('CircMAC',     'circmac',     'v2_enc_circmac'),
]

PROPOSED_COLOR = '#E05C2A'
BASELINE_COLOR = '#6B9CC7'

METRICS = [
    ('f1_macro', '(a) F1-macro', (0.55, 0.83)),
    ('roc_auc',  '(b) AUROC',    (0.70, 0.97)),
    ('auprc',    '(c) AUPRC',    (0.15, 0.65)),
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


def load_scores(model_dir, exp_base):
    scores = {'f1_macro': [], 'roc_auc': [], 'auprc': []}
    for s in SEEDS:
        p = LOGS / model_dir / f'{exp_base}_s{s}' / str(s) / 'training.json'
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        final = d.get('final', {})
        if not final:
            continue
        site = list(final.values())[0]['scores']['sites']
        for m in scores:
            if m in site:
                scores[m].append(site[m])
    return scores


def plot_panel(ax, data, metric, title, ylim):
    """data: list of (label, color, vals)"""
    ylo     = ylim[0]
    y_range = ylim[1] - ylim[0]
    bar_w   = 0.55

    for i, (label, color, vals) in enumerate(data):
        if not vals:
            continue
        mean = np.mean(vals)
        std  = np.std(vals)

        ax.bar(i, mean - ylo, bottom=ylo, width=bar_w,
               color=color, alpha=0.85, zorder=2, linewidth=0)

        if len(vals) > 1:
            ax.errorbar(i, mean, yerr=std, fmt='none', color='#222222',
                        capsize=4, capthick=1.4, elinewidth=1.4, zorder=4)

        ax.text(i, mean + std + y_range * 0.025,
                f'{mean:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                color='#222222', zorder=6,
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85))

    ax.set_xticks(range(len(data)))
    ax.set_xticklabels([d[0] for d in data], rotation=20, ha='right')
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    for tick in ax.get_xticklabels():
        if tick.get_text() == 'CircMAC':
            tick.set_fontweight('bold')
            tick.set_color(PROPOSED_COLOR)


def main():
    all_data = []
    rows = []

    for label, model_dir, exp_base in MODELS:
        color  = PROPOSED_COLOR if label == 'CircMAC' else BASELINE_COLOR
        scores = load_scores(model_dir, exp_base)
        all_data.append((label, color, scores))

        max_len = max((len(scores[m]) for m in scores), default=0)
        for i in range(max_len):
            seed = SEEDS[i] if i < len(SEEDS) else i + 1
            rows.append({
                'model':    label,
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
    df.to_csv(OUT / 'fig2_encoder_data.csv', index=False)
    summary.to_csv(OUT / 'fig2_encoder_summary.csv')
    print(summary.to_string())

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2))
    fig.suptitle('Encoder Architecture Comparison', fontsize=12, fontweight='bold', y=1.01)

    for ax, (metric, title, ylim) in zip(axes, METRICS):
        panel_data = [(label, color, scores[metric]) for label, color, scores in all_data]
        plot_panel(ax, panel_data, metric, title, ylim)

    legend_elems = [
        Patch(facecolor=PROPOSED_COLOR, label='CircMAC (ours)'),
        Patch(facecolor=BASELINE_COLOR, label='Baseline encoder'),
    ]
    fig.legend(handles=legend_elems, loc='upper center', ncol=2,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.0))

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)

    for ext in ['pdf', 'png']:
        p = OUT / f'fig2_encoder.{ext}'
        fig.savefig(p, dpi=200, bbox_inches='tight')
        print(f'Saved → {p}')
    plt.close(fig)


if __name__ == '__main__':
    main()
