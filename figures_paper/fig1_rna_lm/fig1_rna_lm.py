#!/usr/bin/env python3
"""
Fig 1 (Main Result): RNA Language Model Comparison
  CircMAC+Pairing (proposed) vs. frozen / fine-tuned RNA-LMs

Layout per metric panel:
  x=0..3  : Frozen RNA-LMs  (RNABERT, RNAErnie, RNAMSM, RNA-FM)  — hatched
  x=5..8  : Fine-tuned RNA-LMs  — solid
  x=10    : CircMAC+Pairing (ours)  — orange, highlighted
  (gaps at x=4 and x=9 act as visual separators)

Note: RNA-LM experiments have only seed 1 (no error bars).
      CircMAC+Pairing uses seeds 1,2,3 (mean ± std shown).

Output: figures_paper/fig1_rna_lm/fig1_rna_lm.{pdf,png}
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

ROOT  = Path(__file__).resolve().parents[2]
LOGS  = ROOT / 'logs_0512'
OUT   = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

SEEDS_LM   = [1, 2, 3]
SEEDS_OURS = [1, 2, 3]

RNA_LMS = [
    ('RNABERT',  'rnabert',  'exp1_fair_frozen_rnabert',     True),
    ('RNAErnie', 'rnaernie', 'exp1_fair_frozen_rnaernie',    True),
    ('RNAMSM',   'rnamsm',   'exp1_fair_frozen_rnamsm',      True),
    ('RNA-FM',   'rnafm',    'exp1_fair_frozen_rnafm',       True),
    ('RNABERT',  'rnabert',  'exp1_fair_trainable_rnabert',  False),
    ('RNAErnie', 'rnaernie', 'exp1_fair_trainable_rnaernie', False),
    ('RNAMSM',   'rnamsm',   'exp1_fair_trainable_rnamsm',   False),
    ('RNA-FM',   'rnafm',    'exp1_fair_trainable_rnafm',    False),
]

LM_COLORS = {
    'RNABERT':  '#4878CF',
    'RNAErnie': '#9467BD',
    'RNAMSM':   '#2CA02C',
    'RNA-FM':   '#17BECF',
}
PROPOSED_COLOR = '#E05C2A'
FROZEN_HATCH   = '//'

METRICS = [
    ('f1_macro', '(a) F1-macro', (0.55, 0.83)),
    ('roc_auc',  '(b) AUROC',    (0.65, 0.97)),
    ('auprc',    '(c) AUPRC',    (0.15, 0.65)),
]

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         10,
    'axes.titlesize':    11,
    'axes.labelsize':    10,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'xtick.labelsize':   8.5,
    'ytick.labelsize':   9,
    'pdf.fonttype':      42,
    'ps.fonttype':       42,
})

# x positions: frozen(0-3) | gap | trainable(5-8) | gap | ours(10)
X_FROZEN    = [0, 1, 2, 3]
X_TRAINABLE = [5, 6, 7, 8]
X_OURS      = 10


def load_lm_scores(model_dir, exp_base, seeds):
    scores = {'f1_macro': [], 'roc_auc': [], 'auprc': []}
    for s in seeds:
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


def load_ours_scores():
    scores = {'f1_macro': [], 'roc_auc': [], 'auprc': []}
    for s in SEEDS_OURS:
        p = LOGS / 'circmac' / f'v2_pt_pairing_s{s}' / str(s) / 'training.json'
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


def plot_panel(ax, entries, metric, title, ylim):
    """
    entries: list of (x_pos, label, color, hatch, vals)
    """
    ylo     = ylim[0]
    y_range = ylim[1] - ylim[0]
    bar_w   = 0.62

    for x, label, color, hatch, vals in entries:
        if not vals:
            continue
        mean = np.mean(vals)
        std  = np.std(vals)

        ax.bar(x, mean - ylo, bottom=ylo, width=bar_w,
               color=color, alpha=0.85, zorder=2, linewidth=0,
               hatch=hatch, edgecolor='white' if hatch else color)

        if len(vals) > 1:
            ax.errorbar(x, mean, yerr=std, fmt='none', color='#222222',
                        capsize=3.5, capthick=1.3, elinewidth=1.3, zorder=4)

        ax.text(x, mean + std + y_range * 0.022,
                f'{mean:.3f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold',
                color='#222222', zorder=6,
                bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=0.85))

    # Group separators
    ax.axvline(4.0, color='#cccccc', lw=1.0, zorder=1)
    ax.axvline(9.0, color='#cccccc', lw=1.0, zorder=1)

    all_x = [e[0] for e in entries]
    ax.set_xlim(min(all_x) - 0.6, max(all_x) + 0.6)
    ax.set_ylim(*ylim)
    ax.set_xticks([e[0] for e in entries])
    ax.set_xticklabels([e[1] for e in entries], rotation=35, ha='right')

    # Bold + color for proposed
    for tick, (_, label, color, _, _) in zip(ax.get_xticklabels(), entries):
        if label == 'Ours':
            tick.set_fontweight('bold')
            tick.set_color(PROPOSED_COLOR)

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
    ax.set_axisbelow(True)


def main():
    # Load scores
    lm_data = {}
    for label, model_dir, exp_base, is_frozen in RNA_LMS:
        key = (label, is_frozen)
        lm_data[key] = load_lm_scores(model_dir, exp_base, SEEDS_LM)

    ours_scores = load_ours_scores()

    # Build CSV
    rows = []
    for label, model_dir, exp_base, is_frozen in RNA_LMS:
        mode = 'frozen' if is_frozen else 'fine-tuned'
        sc = lm_data[(label, is_frozen)]
        for i, s in enumerate(SEEDS_LM):
            rows.append({
                'model': label, 'mode': mode, 'seed': s,
                'f1_macro': sc['f1_macro'][i] if i < len(sc['f1_macro']) else float('nan'),
                'roc_auc':  sc['roc_auc'][i]  if i < len(sc['roc_auc'])  else float('nan'),
                'auprc':    sc['auprc'][i]     if i < len(sc['auprc'])    else float('nan'),
            })
    for i, s in enumerate(SEEDS_OURS):
        rows.append({
            'model': 'CircMAC+Pairing', 'mode': 'proposed', 'seed': s,
            'f1_macro': ours_scores['f1_macro'][i] if i < len(ours_scores['f1_macro']) else float('nan'),
            'roc_auc':  ours_scores['roc_auc'][i]  if i < len(ours_scores['roc_auc'])  else float('nan'),
            'auprc':    ours_scores['auprc'][i]     if i < len(ours_scores['auprc'])    else float('nan'),
        })
    df = pd.DataFrame(rows)
    summary = df.groupby(['model', 'mode']).agg(
        f1_mean=('f1_macro', 'mean'), f1_std=('f1_macro', 'std'),
        roc_mean=('roc_auc',  'mean'), roc_std=('roc_auc',  'std'),
        auprc_mean=('auprc',  'mean'), auprc_std=('auprc',  'std'),
    ).round(4)
    df.to_csv(OUT / 'fig1_rna_lm_data.csv', index=False)
    summary.to_csv(OUT / 'fig1_rna_lm_summary.csv')
    print(summary.to_string())

    # Build entries per panel
    lm_labels = ['RNABERT', 'RNAErnie', 'RNAMSM', 'RNA-FM']

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    fig.suptitle('RNA Language Model Comparison', fontsize=12, fontweight='bold', y=1.01)

    for ax, (metric, title, ylim) in zip(axes, METRICS):
        entries = []
        # Frozen group
        for x, label in zip(X_FROZEN, lm_labels):
            vals = lm_data[(label, True)][metric]
            entries.append((x, label, LM_COLORS[label], FROZEN_HATCH, vals))
        # Trainable group
        for x, label in zip(X_TRAINABLE, lm_labels):
            vals = lm_data[(label, False)][metric]
            entries.append((x, label, LM_COLORS[label], None, vals))
        # Proposed
        entries.append((X_OURS, 'Ours', PROPOSED_COLOR, None, ours_scores[metric]))

        plot_panel(ax, entries, metric, title, ylim)

    # Group header annotations (on all panels)
    for ax, (_, _, ylim) in zip(axes, METRICS):
        y_top = ylim[1]
        y_range = ylim[1] - ylim[0]
        ax.text(1.5, y_top - y_range * 0.01, 'Frozen',
                ha='center', va='top', fontsize=8.5, color='#555555')
        ax.text(6.5, y_top - y_range * 0.01, 'Fine-tuned',
                ha='center', va='top', fontsize=8.5, color='#555555')

    # Legend
    legend_elems = [
        mpatches.Patch(facecolor='#888888', hatch=FROZEN_HATCH, edgecolor='white',
                       label='Frozen miRNA encoder'),
        mpatches.Patch(facecolor='#888888', label='Fine-tuned miRNA encoder'),
        mpatches.Patch(facecolor=PROPOSED_COLOR, label='CircMAC+Pairing (Ours)'),
    ]
    fig.legend(handles=legend_elems, loc='upper center', ncol=3,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.0))

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)

    for ext in ['pdf', 'png']:
        p = OUT / f'fig1_rna_lm.{ext}'
        fig.savefig(p, dpi=200, bbox_inches='tight')
        print(f'Saved → {p}')
    plt.close(fig)


if __name__ == '__main__':
    main()
