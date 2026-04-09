"""
BSJ Analysis — CSV에서 시각화
================================
서버에서 bsj_analysis_results.csv를 가져온 후 여기서 실행.

사용법:
  python docs/paper_cmi/plot_bsj_from_csv.py \
    --csv docs/paper_cmi/bsj_analysis_results.csv \
    --out docs/paper_cmi/

  # 여러 서버 결과 합치기:
  python docs/paper_cmi/plot_bsj_from_csv.py \
    --csv docs/paper_cmi/results_server1/bsj_analysis_results.csv \
          docs/paper_cmi/results_server2/bsj_analysis_results.csv \
    --out docs/paper_cmi/
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── 모델 표시 이름 / 순서 ──────────────────────────────────────
MODEL_LABELS = {
    'circmac':     'CircMAC (ours)',
    'mamba':       'Mamba',
    'hymba':       'HyMBA',
    'lstm':        'LSTM',
    'transformer': 'Transformer',
    'rnabert':     'RNA-BERT',
    'rnaernie':    'RNA-ERNIE',
    'rnafm':       'RNA-FM',
    'rnamsm':      'RNA-MSM',
}

MODEL_ORDER = ['circmac', 'mamba', 'hymba', 'lstm', 'transformer',
               'rnabert', 'rnaernie', 'rnafm', 'rnamsm']

COLOR_PROXIMAL = '#E74C3C'
COLOR_DISTAL   = '#3498DB'
COLOR_OVERALL  = '#95A5A6'

ARCH_COLORS = {
    'circmac':     '#E74C3C',
    'mamba':       '#E67E22',
    'hymba':       '#F39C12',
    'lstm':        '#27AE60',
    'transformer': '#2ECC71',
    'rnabert':     '#8E44AD',
    'rnaernie':    '#9B59B6',
    'rnafm':       '#6C3483',
    'rnamsm':      '#A569BD',
}


def load_and_merge(csv_paths: list) -> tuple:
    """
    여러 CSV 파일 로드.
    - 중복 모델 없음(서버별 병합): 그냥 concat 후 반환 (std=None)
    - 중복 모델 있음(seed별 집계): model 기준 mean±std 계산 후 반환
    Returns: (df_mean, df_std or None)
    """
    dfs = []
    for p in csv_paths:
        df = pd.read_csv(p)
        df.columns = df.columns.str.strip()
        dfs.append(df)
    merged = pd.concat(dfs, ignore_index=True)

    # 중복 모델이 있으면 seed 집계 모드
    if merged['model'].duplicated().any():
        numeric_cols = ['overall', 'proximal', 'distal']
        df_mean = merged.groupby('model')[numeric_cols].mean().reset_index()
        df_std  = merged.groupby('model')[numeric_cols].std().reset_index()
        # prox_n, dist_n은 대표값(첫 행)
        for col in ['label', 'prox_n', 'dist_n']:
            if col in merged.columns:
                df_mean[col] = merged.groupby('model')[col].first().values
        n_seeds = merged.groupby('model').size().iloc[0]
        print(f'[Aggregate mode] {n_seeds} seeds → mean ± std per model')
        return df_mean, df_std
    else:
        # 서버별 병합 — 중복 없음
        return merged, None


def sort_by_model_order(df: pd.DataFrame) -> pd.DataFrame:
    order_map = {m: i for i, m in enumerate(MODEL_ORDER)}
    df = df.copy()
    df['_order'] = df['model'].map(order_map).fillna(99)
    return df.sort_values('_order').drop(columns='_order').reset_index(drop=True)


def plot_proximal_vs_distal_bar(df: pd.DataFrame, out_dir: Path, df_std: pd.DataFrame = None):
    """Figure A: proximal vs distal F1 grouped bar chart (with optional error bars)."""
    df = sort_by_model_order(df)
    if df_std is not None:
        df_std = df_std.set_index('model').reindex(df['model']).reset_index()
    n = len(df)
    x = np.arange(n)
    width = 0.28

    fig, ax = plt.subplots(figsize=(max(8, n * 1.1), 5))

    err_p = df_std['proximal'].values if df_std is not None else None
    err_d = df_std['distal'].values   if df_std is not None else None
    err_o = df_std['overall'].values  if df_std is not None else None

    ax.bar(x - width, df['proximal'], width, yerr=err_p, capsize=3,
           color=COLOR_PROXIMAL, alpha=0.85, label='BSJ-Proximal',
           error_kw=dict(elinewidth=1, ecolor='#666'))
    ax.bar(x,         df['distal'],   width, yerr=err_d, capsize=3,
           color=COLOR_DISTAL,   alpha=0.85, label='BSJ-Distal',
           error_kw=dict(elinewidth=1, ecolor='#666'))
    ax.bar(x + width, df['overall'],  width, yerr=err_o, capsize=3,
           color=COLOR_OVERALL,  alpha=0.85, label='Overall',
           error_kw=dict(elinewidth=1, ecolor='#666'))

    # delta annotation (proximal - distal)
    for i, (_, row) in enumerate(df.iterrows()):
        delta = row['proximal'] - row['distal']
        sign = '+' if delta >= 0 else ''
        color = COLOR_PROXIMAL if delta >= 0 else COLOR_DISTAL
        ax.text(x[i] - width/2, max(row['proximal'], row['distal']) + 0.005,
                f'{sign}{delta*100:.1f}pp', ha='center', va='bottom',
                fontsize=7, color=color, fontweight='bold')

    labels = [MODEL_LABELS.get(m, m) for m in df['model']]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('F1 Macro', fontsize=11)
    title = 'BSJ-Proximal vs BSJ-Distal F1 by Model'
    if df_std is not None:
        title += ' (mean ± std, 3 seeds)'
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(0, df[['proximal','distal','overall']].max().max() * 1.15)
    ax.grid(axis='y', alpha=0.3)
    ax.spines[['top','right']].set_visible(False)

    plt.tight_layout()
    out = out_dir / 'bsj_comparison_bar.pdf'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.savefig(str(out).replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


def plot_delta_heatmap(df: pd.DataFrame, out_dir: Path, df_std: pd.DataFrame = None):
    """Figure B: (proximal - distal) delta 막대그래프 (with optional error bars)."""
    df = sort_by_model_order(df)
    df = df.copy()
    df['delta'] = (df['proximal'] - df['distal']) * 100  # in pp

    # delta std: sqrt(std_prox^2 + std_dist^2) * 100
    err_delta = None
    if df_std is not None:
        df_std = df_std.set_index('model').reindex(df['model']).reset_index()
        err_delta = np.sqrt(df_std['proximal'].values**2 + df_std['distal'].values**2) * 100

    fig, ax = plt.subplots(figsize=(max(7, len(df) * 0.9), 4))

    colors = [COLOR_PROXIMAL if d >= 0 else COLOR_DISTAL for d in df['delta']]
    bars = ax.bar(range(len(df)), df['delta'], color=colors, alpha=0.85, edgecolor='white',
                  yerr=err_delta, capsize=4, error_kw=dict(elinewidth=1, ecolor='#444'))

    ax.axhline(0, color='black', linewidth=0.8)
    for i, (bar, d) in enumerate(zip(bars, df['delta'])):
        va = 'bottom' if d >= 0 else 'top'
        offset = 0.05 if d >= 0 else -0.05
        ax.text(bar.get_x() + bar.get_width()/2, d + offset,
                f'{d:+.1f}', ha='center', va=va, fontsize=8, fontweight='bold')

    labels = [MODEL_LABELS.get(m, m) for m in df['model']]
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Proximal − Distal F1 (pp)', fontsize=11)
    title = 'BSJ-Proximal Advantage (positive = proximal better)'
    if df_std is not None:
        title += '\n(mean ± std, 3 seeds)'
    ax.set_title(title, fontsize=12, fontweight='bold')

    prox_patch = mpatches.Patch(color=COLOR_PROXIMAL, alpha=0.85, label='Proximal advantage')
    dist_patch = mpatches.Patch(color=COLOR_DISTAL,   alpha=0.85, label='Distal advantage')
    ax.legend(handles=[prox_patch, dist_patch], fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.spines[['top','right']].set_visible(False)

    plt.tight_layout()
    out = out_dir / 'bsj_delta.pdf'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.savefig(str(out).replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


def plot_scatter(df: pd.DataFrame, out_dir: Path):
    """Figure C: proximal F1 vs distal F1 scatter (y=x line = no bias)."""
    df = sort_by_model_order(df)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    for _, row in df.iterrows():
        m = row['model']
        c = ARCH_COLORS.get(m, '#888')
        ax.scatter(row['distal'], row['proximal'], color=c, s=120, zorder=5, edgecolors='white', linewidths=0.8)
        ax.annotate(MODEL_LABELS.get(m, m),
                    (row['distal'], row['proximal']),
                    textcoords='offset points', xytext=(6, 4),
                    fontsize=8, color=c)

    # y=x diagonal (no bias line)
    lim_min = min(df['distal'].min(), df['proximal'].min()) - 0.02
    lim_max = max(df['distal'].max(), df['proximal'].max()) + 0.02
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', alpha=0.4, linewidth=1, label='No difference')
    ax.fill_between([lim_min, lim_max], [lim_min, lim_max], [lim_max, lim_max],
                    alpha=0.04, color=COLOR_PROXIMAL)
    ax.fill_between([lim_min, lim_max], [lim_min, lim_min], [lim_min, lim_max],
                    alpha=0.04, color=COLOR_DISTAL)

    ax.set_xlabel('BSJ-Distal F1', fontsize=11)
    ax.set_ylabel('BSJ-Proximal F1', fontsize=11)
    ax.set_title('BSJ-Proximal vs BSJ-Distal\n(above diagonal = proximal advantage)', fontsize=11)
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.spines[['top','right']].set_visible(False)

    plt.tight_layout()
    out = out_dir / 'bsj_scatter.pdf'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.savefig(str(out).replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


def print_table(df: pd.DataFrame, df_std: pd.DataFrame = None):
    df = sort_by_model_order(df)
    df = df.copy()
    df['delta_pp'] = (df['proximal'] - df['distal']) * 100
    df['label'] = df['model'].map(MODEL_LABELS)

    if df_std is not None:
        std_idx = df_std.set_index('model')
        print('\n' + '='*85)
        print(f"{'Model':<16} {'Overall':>14} {'Proximal':>16} {'Distal':>14} {'Δ(pp)':>8}")
        print('-'*85)
        for _, row in df.iterrows():
            m = row['model']
            s = std_idx.loc[m] if m in std_idx.index else None
            sign = '↑' if row['delta_pp'] >= 0 else '↓'
            ov  = f"{row['overall']:.4f}±{s['overall']:.4f}"  if s is not None else f"{row['overall']:.4f}"
            pr  = f"{row['proximal']:.4f}±{s['proximal']:.4f}" if s is not None else f"{row['proximal']:.4f}"
            di  = f"{row['distal']:.4f}±{s['distal']:.4f}"    if s is not None else f"{row['distal']:.4f}"
            print(f"{row['label']:<16} {ov:>14} {pr:>16} {di:>14} {row['delta_pp']:>+7.2f}pp {sign}")
        print('='*85)
    else:
        print('\n' + '='*70)
        print(f"{'Model':<16} {'Overall':>8} {'Proximal':>10} {'Distal':>8} {'Δ(pp)':>8}  {'n_prox':>7} {'n_dist':>7}")
        print('-'*70)
        for _, row in df.iterrows():
            sign = '↑' if row['delta_pp'] >= 0 else '↓'
            print(f"{row['label']:<16} {row['overall']:>8.4f} {row['proximal']:>10.4f} "
                  f"{row['distal']:>8.4f} {row['delta_pp']:>+7.2f}pp {sign}  "
                  f"{int(row.get('prox_n', 0)):>6}  {int(row.get('dist_n', 0)):>6}")
        print('='*70)


def main():
    parser = argparse.ArgumentParser(description='Plot BSJ analysis results from CSV')
    parser.add_argument('--csv', nargs='+', required=True,
                        help='CSV file(s): 서버별 병합 또는 seed별 집계 자동 판단')
    parser.add_argument('--out', '--out_dir', default='docs/paper_cmi/',
                        help='Output directory for figures')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, df_std = load_and_merge(args.csv)
    print(f'\nLoaded {len(df)} models from {len(args.csv)} CSV file(s)')
    print_table(df, df_std)

    plot_proximal_vs_distal_bar(df, out_dir, df_std)
    plot_delta_heatmap(df, out_dir, df_std)
    plot_scatter(df, out_dir)

    print(f'\nAll figures saved to: {out_dir}')


if __name__ == '__main__':
    main()
