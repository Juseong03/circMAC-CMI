#!/usr/bin/env python3
"""
Fig 1: Encoder Architecture Comparison (EXP3)
  - Panel A: F1-macro (mean ± std, individual dots)
  - Panel B: AUROC    (mean ± std, individual dots)
  - Panel C: AUPRC   (mean ± std, individual dots)

Output (this folder):
  fig1_encoder.pdf / fig1_encoder.png
  fig1_encoder_data.csv
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

MODELS = [
    ('CircMAC',     'circmac',     'v2_enc_circmac'),
    ('Transformer', 'transformer', 'v2_enc_transformer'),
    ('Mamba',       'mamba',       'v2_enc_mamba'),
    ('LSTM',        'lstm',        'v2_enc_lstm'),
    ('Hymba',       'hymba',       'v2_enc_hymba'),
]

PROPOSED = '#E05C2A'
BASELINE = '#6B9CC7'
DOT_ALPHA = 0.75
BAR_ALPHA = 0.82

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
        sc = list(final.values())[0]['scores']['sites']
        for k in scores:
            scores[k].append(sc[k])
    return scores


def plot_panel(ax, data, metric, ylabel, ylim):
    ylo = ylim[0]
    for i, (label, color, vals) in enumerate(data):
        if not vals:
            continue
        mean = np.mean(vals)
        std  = np.std(vals)
        ax.bar(i, mean - ylo, bottom=ylo, width=0.55, color=color,
               alpha=BAR_ALPHA, zorder=2, linewidth=0)
        ax.errorbar(i, mean, yerr=std, fmt='none', color='#222222',
                    capsize=4, capthick=1.4, elinewidth=1.4, zorder=4)
        jitter = np.random.default_rng(42).uniform(-0.07, 0.07, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   color='white', edgecolors=color, s=32, linewidths=1.5,
                   zorder=5, alpha=DOT_ALPHA)
        ax.text(i, mean + std + (ylim[1] - ylim[0]) * 0.025,
                f'{mean:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                color='#222222', zorder=6,
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85))
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels([d[0] for d in data], rotation=20, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
    ax.set_axisbelow(True)


def main():
    all_data = []
    rows = []
    for label, model_dir, exp_base in MODELS:
        color = PROPOSED if label == 'CircMAC' else BASELINE
        sc = load_scores(model_dir, exp_base)
        all_data.append((label, color, sc))
        for i, s in enumerate(SEEDS):
            if i < len(sc['f1_macro']):
                rows.append({'model': label, 'seed': s,
                             'f1_macro': sc['f1_macro'][i],
                             'roc_auc':  sc['roc_auc'][i],
                             'auprc':    sc['auprc'][i]})

    # Save data CSV
    df = pd.DataFrame(rows)
    # Add mean/std summary
    summary = df.groupby('model').agg(
        f1_mean=('f1_macro','mean'), f1_std=('f1_macro','std'),
        roc_mean=('roc_auc','mean'),  roc_std=('roc_auc','std'),
        auprc_mean=('auprc','mean'),  auprc_std=('auprc','std'),
    ).round(4)
    df.to_csv(OUT / 'fig1_encoder_data.csv', index=False)
    summary.to_csv(OUT / 'fig1_encoder_summary.csv')
    print(summary.to_string())

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.0))
    fig.suptitle('Encoder Architecture Comparison', fontsize=12, fontweight='bold', y=1.01)

    metrics = [
        ('f1_macro', 'F1-macro', (0.55, 0.85)),
        ('roc_auc',  'AUROC',    (0.70, 0.97)),
        ('auprc',    'AUPRC',    (0.15, 0.65)),
    ]
    for ax, (metric, ylabel, ylim) in zip(axes, metrics):
        panel_data = [(label, color, sc[metric]) for label, color, sc in all_data]
        plot_panel(ax, panel_data, metric, ylabel, ylim)

    for ax in axes:
        for tick in ax.get_xticklabels():
            if tick.get_text() == 'CircMAC':
                tick.set_fontweight('bold')
                tick.set_color(PROPOSED)

    fig.tight_layout()
    for ext in ['pdf', 'png']:
        p = OUT / f'fig1_encoder.{ext}'
        fig.savefig(p, dpi=200, bbox_inches='tight')
        print(f'Saved → {p}')
    plt.close(fig)


if __name__ == '__main__':
    main()
