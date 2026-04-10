"""
CSV로 저장된 raw prediction 값을 이용한 시각화

사용법:
  python plot_from_csv.py --csv binding_visualization_*.csv
  python plot_from_csv.py --csv foo.csv --isoform "chr4|5565258" --top_mirna 10
  python plot_from_csv.py --csv foo.csv --list_isoforms
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


BSJ_COLOR  = '#F39C12'
BIND_COLOR = '#E74C3C'

# ── 유틸 ──────────────────────────────────────────────────────────────────────
def sanitize(s):
    return re.sub(r'[|,\s/\\:*?"<>]', '_', s)


def load_csv(path):
    return pd.read_csv(path)


def list_isoforms(df):
    print(f"{'isoform_ID':<70} {'L':>5} {'n_mirna':>8} {'bsj_prox':>9}")
    print('-' * 96)
    results = []
    for iso, grp in df.groupby('isoform_ID'):
        L = grp['position'].max() + 1
        n_mirna = grp['miRNA_ID'].nunique()
        w = 30
        bsj = grp[(grp['position'] < w) | (grp['position'] >= L - w)]
        bsj_prox = int(bsj['ground_truth'].sum())
        results.append((iso, L, n_mirna, bsj_prox))
    results.sort(key=lambda x: (-x[2], -x[3]))
    for iso, L, n_mirna, bsj_prox in results:
        print(f"{iso[:68]:<70} {L:>5} {n_mirna:>8} {bsj_prox:>9}")


def get_isoform_data(df, isoform_substr):
    mask = df['isoform_ID'].str.contains(isoform_substr, regex=False)
    matched = df[mask]['isoform_ID'].unique()
    if len(matched) == 0:
        raise ValueError(f"No isoform matching '{isoform_substr}'")
    if len(matched) > 1:
        print(f"Warning: {len(matched)} isoforms matched, using first: {matched[0][:60]}")
    return df[df['isoform_ID'] == matched[0]], matched[0]


def pick_top_mirnas(sub, top_n, mode='bsj_prox'):
    """BSJ 근처 binding 기준 상위 miRNA 선택"""
    L = sub['position'].max() + 1
    w = 30
    scores = []
    for mirna, grp in sub.groupby('miRNA_ID'):
        grp = grp.sort_values('position')
        gt = grp['ground_truth'].values
        pred = grp['pred_circmac'].values
        if mode == 'bsj_prox':
            score = gt[:w].sum() + gt[max(0, L-w):].sum()
        else:
            score = gt.sum()
        scores.append((mirna, score, gt, pred))
    scores.sort(key=lambda x: -x[1])
    return scores[:top_n]


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1: Multi-miRNA Heatmap (GT + Pred)
# ══════════════════════════════════════════════════════════════════════════════
def plot_heatmap(sub, iso_full, top_n=12, out_dir=None):
    L = sub['position'].max() + 1
    top = pick_top_mirnas(sub, top_n, mode='bsj_prox')
    mirna_names = [t[0] for t in top]
    gt_mat   = np.array([t[2][:L] for t in top], dtype=float)
    pred_mat = np.array([t[3][:L] for t in top], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(min(L / 8, 20), top_n * 0.55 + 3),
                             sharex=True)
    fig.patch.set_facecolor('white')

    for ax, mat, title, cmap in [
        (axes[0], gt_mat,   'Ground Truth Binding Sites', 'Reds'),
        (axes[1], pred_mat, 'CircMAC Prediction',         'RdYlGn'),
    ]:
        im = ax.imshow(mat, aspect='auto', cmap=cmap,
                       vmin=0, vmax=1, interpolation='nearest')
        ax.set_yticks(range(len(mirna_names)))
        ax.set_yticklabels(mirna_names, fontsize=8)
        ax.set_title(title, fontsize=11, fontweight='bold', pad=4)
        plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01)

        # BSJ 마커
        for x, lbl in [(0, "5'BSJ"), (L - 1, "3'BSJ")]:
            ax.axvline(x, color=BSJ_COLOR, lw=2, linestyle='--', alpha=0.8)

    axes[1].set_xlabel('Position', fontsize=10)
    axes[1].text(0, -1.2, "5' BSJ", ha='center', fontsize=8,
                 color=BSJ_COLOR, fontweight='bold',
                 transform=axes[1].get_xaxis_transform())
    axes[1].text(L - 1, -1.2, "3' BSJ", ha='center', fontsize=8,
                 color=BSJ_COLOR, fontweight='bold',
                 transform=axes[1].get_xaxis_transform())

    iso_short = iso_full if len(iso_full) <= 50 else iso_full[:47] + '...'
    fig.suptitle(f'Multi-miRNA Binding Heatmap\n{iso_short}  (L={L}, top {top_n} by BSJ-proximity)',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()

    stem = f'csv_heatmap_{sanitize(iso_full[:40])}'
    out = Path(out_dir or Path(__file__).parent)
    plt.savefig(out / f'{stem}.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(out / f'{stem}.png', bbox_inches='tight', dpi=150)
    print(f'Saved: {stem}.pdf/png')
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2: 개별 miRNA 선형 뷰 (GT vs Pred 오버레이)
# ══════════════════════════════════════════════════════════════════════════════
def plot_linear_overlay(sub, iso_full, top_n=6, out_dir=None):
    L = sub['position'].max() + 1
    top = pick_top_mirnas(sub, top_n, mode='bsj_prox')

    ncols = 2
    nrows = (len(top) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(14, nrows * 2.2 + 1.5), sharex=False)
    fig.patch.set_facecolor('white')
    axes = axes.flatten()

    x = np.arange(L)
    for idx, (mirna, score, gt, pred) in enumerate(top):
        ax = axes[idx]
        ax.fill_between(x, gt, alpha=0.35, color=BIND_COLOR, label='GT')
        ax.plot(x, pred, color='#2980B9', lw=1.2, alpha=0.9, label='CircMAC')
        ax.axvline(0,     color=BSJ_COLOR, lw=1.8, linestyle='--', alpha=0.8)
        ax.axvline(L - 1, color=BSJ_COLOR, lw=1.8, linestyle='--', alpha=0.8)
        ax.set_ylim(-0.05, 1.15)
        ax.set_title(f'{mirna}  (bsj_prox={score:.0f})', fontsize=8.5,
                     fontweight='bold', pad=3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.legend(fontsize=7.5, loc='upper right')

    # 빈 axes 숨기기
    for ax in axes[len(top):]:
        ax.set_visible(False)

    iso_short = iso_full if len(iso_full) <= 50 else iso_full[:47] + '...'
    fig.suptitle(f'CircMAC vs Ground Truth  |  {iso_short}',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()

    stem = f'csv_linear_{sanitize(iso_full[:40])}'
    out = Path(out_dir or Path(__file__).parent)
    plt.savefig(out / f'{stem}.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(out / f'{stem}.png', bbox_inches='tight', dpi=150)
    print(f'Saved: {stem}.pdf/png')
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',  required=True, help='Path to binding_visualization_*.csv')
    parser.add_argument('--isoform', type=str, default=None,
                        help='isoform_ID substring to visualize')
    parser.add_argument('--top_mirna', type=int, default=12,
                        help='Number of top miRNAs to show (default: 12)')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Output directory (default: same as script)')
    parser.add_argument('--list_isoforms', action='store_true',
                        help='List all isoforms in CSV and exit')
    args = parser.parse_args()

    df = load_csv(args.csv)

    if args.list_isoforms:
        list_isoforms(df)
        exit(0)

    if args.isoform is None:
        # 자동 선택: BSJ prox 가장 많고 miRNA 수 많은 것
        best = None
        best_score = -1
        for iso, grp in df.groupby('isoform_ID'):
            L = grp['position'].max() + 1
            w = 30
            bsj = grp[(grp['position'] < w) | (grp['position'] >= L - w)]
            bsj_prox = int(bsj['ground_truth'].sum())
            n_mirna = grp['miRNA_ID'].nunique()
            score = n_mirna * 10 + bsj_prox
            if score > best_score:
                best_score = score
                best = iso
        print(f'Auto-selected isoform: {best[:60]}')
        sub, iso_full = df[df['isoform_ID'] == best], best
    else:
        sub, iso_full = get_isoform_data(df, args.isoform)

    plot_heatmap(sub, iso_full, top_n=args.top_mirna, out_dir=args.out_dir)
    plot_linear_overlay(sub, iso_full, top_n=min(args.top_mirna, 6), out_dir=args.out_dir)
