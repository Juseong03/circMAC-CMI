#!/usr/bin/env python3
"""
Fig 2: Pretraining Strategy Comparison (EXP2)

세 지표(F1-macro / AUROC / AUPRC)에 대해
CircMAC 백본 + 각 사전학습 과제의 성능을 비교한다.

x-tick 라벨 :  CircMAC, +MLM, +SSP, +BPP
막대 색상   :  baseline  → 회색,   pretraining → 파란색
Pairing     :  BPP(Base-Pairing Prediction)로 표기
"""

import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ROOT  = Path(__file__).resolve().parents[2]
OUT   = Path(__file__).resolve().parent
DATA_CSV = OUT / 'fig2_pretraining_data.csv'

# (internal_label, group)
MODELS = [
    ('No PT', 'baseline'),
    ('MLM',   'single'),
    ('SSP',   'single'),
    ('BPP',   'single'),   # Pairing → BPP
]

# 색상 팔레트 (원래대로)
COLORS = {'baseline': '#AAAAAA',   # light gray
          'single'  : '#6B9CC7'}  # blue

METRICS = [
    ('f1_macro', '(a) F1-macro', (0.72, 0.80)),
    ('roc_auc',  '(b) AUROC',    (0.88, 0.92)),
    ('auprc',    '(c) AUPRC',    (0.47, 0.57)),
]

plt.rcParams.update({
    'font.family':'DejaVu Sans',
    'font.size'  :10,
    'axes.spines.top'  :False,
    'axes.spines.right':False,
    'xtick.labelsize'  :9,
    'pdf.fonttype':42, 'ps.fonttype':42,
})

# ------------------------------------------------------------------
def load_scores(label):
    """CSV에서 label(model) 기준으로 metric 점수 로드"""
    df = pd.read_csv(DATA_CSV)
    rows = df[df['model'] == label]
    out = {m: rows[m].dropna().tolist() for m in ('f1_macro', 'roc_auc', 'auprc')}
    return out

# ------------------------------------------------------------------
def bar_panel(ax, entries, metric, title, ylim):
    """단일 지표 패널 그리기"""
    y0, y_range = ylim[0], ylim[1] - ylim[0]
    bar_w = 0.55
    nopt_mean = None

    for i, (label, grp, scores) in enumerate(entries):
        vals = scores[metric]

        if not vals:          # 결과가 없는 경우(pending)
            ax.bar(i, ylim[1]-y0-0.002, bottom=y0, width=bar_w,
                   color='none', edgecolor='#BBBBBB',
                   linewidth=1.2, linestyle='--')
            ax.text(i, (y0+ylim[1])/2, 'pending', ha='center', va='center',
                    fontsize=7.5, color='#999999', style='italic')
            continue

        mean = np.mean(vals)
        std  = np.std(vals)

        if label == 'No PT':
            nopt_mean = mean

        ax.bar(i, mean-y0, bottom=y0, width=bar_w,
               color=COLORS[grp], alpha=0.9, linewidth=0)
        if len(vals) > 1:
            ax.errorbar(i, mean, yerr=std, fmt='none',
                        color='#222222', capsize=4,
                        capthick=1.3, elinewidth=1.3)
        ax.text(i, mean+std+y_range*0.025, f'{mean:.3f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold',
                color='#222222',
                bbox=dict(boxstyle='round,pad=0.12',
                          fc='white', ec='none', alpha=0.85))

    # No-PT 기준선
    if nopt_mean is not None:
        ax.axhline(nopt_mean, color=COLORS['baseline'],
                   linestyle=':', linewidth=1.3, alpha=0.8)

    # x-tick 설정
    tick_labels = ['CircMAC', '+MLM', '+SSP', '+BPP']
    ax.set_xticks(range(len(entries)))
    ax.set_xticklabels(tick_labels, rotation=0,
                       ha='center', color='#000000')

    ax.set_ylim(*ylim)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=11, fontweight='bold')

# ------------------------------------------------------------------
def main():
    # 데이터 로드
    data = [(lbl, grp, load_scores(lbl))
            for lbl, grp in MODELS]

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(9, 4.4))
    fig.suptitle('Pre-training Strategy Comparison',
                 fontsize=12, fontweight='bold', y=1.01)

    for ax, (metric, title, ylim) in zip(axes, METRICS):
        bar_panel(ax, data, metric, title, ylim)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)

    # 저장
    out_dir = OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        path = out_dir / f'fig2_pretraining.{ext}'
        fig.savefig(path, dpi=200, bbox_inches='tight')
        print('Saved →', path)
    plt.close(fig)

if __name__ == '__main__':
    main()