"""
CSV raw prediction 값을 이용한 다양한 시각화

생성 그림:
  1. heatmap       - GT + 모델별 예측 히트맵 (isoform × miRNA)
  2. overlay       - 다중 모델 예측 오버레이 (상위 N miRNA)
  3. per_model     - 모델별 개별 저장 (circular + linear)
  4. bsj_zoom      - BSJ 근처 확대 뷰
  5. model_summary - 모델간 BSJ-proximal 예측 성능 비교 bar chart

사용법:
  # 모든 그림 생성
  python plot_from_csv.py --csv foo.csv --isoform "chr4|5565258"

  # 특정 그림만
  python plot_from_csv.py --csv foo.csv --isoform "chr4|5565258" --plots heatmap overlay

  # isoform 목록 확인
  python plot_from_csv.py --csv foo.csv --list_isoforms
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Wedge
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd


# ── 색상 ──────────────────────────────────────────────────────────────────────
BSJ_COLOR  = '#F39C12'
BIND_COLOR = '#E74C3C'
NOBIND_COLOR = '#BDC3C7'

MODEL_COLORS = {
    'circmac':     '#E67E22',
    'mamba':       '#2980B9',
    'transformer': '#27AE60',
    'lstm':        '#8E44AD',
    'hymba':       '#16A085',
    'rnabert':     '#C0392B',
    'rnaernie':    '#E74C3C',
    'rnafm':       '#F39C12',
    'rnamsm':      '#95A5A6',
}
FALLBACK_COLORS = ['#E67E22', '#2980B9', '#27AE60', '#8E44AD',
                   '#16A085', '#C0392B', '#F39C12', '#95A5A6']


# ── 유틸 ──────────────────────────────────────────────────────────────────────
def sanitize(s):
    return re.sub(r'[|,\s/\\:*?"<>]', '_', s)


def get_model_cols(df):
    """CSV에서 pred_* 컬럼 자동 감지 → {'circmac': 'pred_circmac', ...}"""
    cols = [c for c in df.columns if c.startswith('pred_')]
    return {c[5:]: c for c in cols}  # 'pred_circmac' → 'circmac'


def get_model_color(label, idx=0):
    return MODEL_COLORS.get(label, FALLBACK_COLORS[idx % len(FALLBACK_COLORS)])


def load_isoform(df, isoform_substr):
    mask = df['isoform_ID'].str.contains(isoform_substr, regex=False)
    matched = df[mask]['isoform_ID'].unique()
    if len(matched) == 0:
        raise ValueError(f"No isoform matching '{isoform_substr}'")
    if len(matched) > 1:
        print(f"Warning: {len(matched)} matched, using first.")
    iso = matched[0]
    return df[df['isoform_ID'] == iso], iso


def pick_top_mirnas(sub, model_cols, top_n, mode='bsj_prox'):
    L = sub['position'].max() + 1
    w = 50
    rows = []
    for mirna, grp in sub.groupby('miRNA_ID'):
        grp = grp.sort_values('position')
        gt   = grp['ground_truth'].values
        preds = {m: grp[col].values for m, col in model_cols.items()}
        bsj  = gt[:w].sum() + gt[max(0, L - w):].sum()
        rows.append((mirna, bsj, gt, preds, L))
    rows.sort(key=lambda x: -x[1])
    return rows[:top_n]


def list_isoforms(df, model_cols):
    print(f"{'isoform_ID':<65} {'L':>5} {'n_mirna':>8} {'bsj_prox':>9} {'models'}")
    print('-' * 100)
    results = []
    for iso, grp in df.groupby('isoform_ID'):
        L = grp['position'].max() + 1
        n_mirna = grp['miRNA_ID'].nunique()
        w = 50
        bsj = int(grp[(grp['position'] < w) | (grp['position'] >= L - w)]['ground_truth'].sum())
        results.append((iso, L, n_mirna, bsj))
    results.sort(key=lambda x: (-x[2], -x[3]))
    mnames = list(model_cols.keys())
    for iso, L, n_mirna, bsj in results:
        print(f"{iso[:63]:<65} {L:>5} {n_mirna:>8} {bsj:>9}  {mnames}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. Heatmap: GT + 모델별 히트맵 (miRNA × position)
# ══════════════════════════════════════════════════════════════════════════════
def plot_heatmap(sub, iso_full, model_cols, top_n=12, out_dir=None):
    top = pick_top_mirnas(sub, model_cols, top_n)
    if not top:
        print("No data for heatmap"); return
    L          = top[0][4]
    mirna_names = [t[0] for t in top]
    gt_mat     = np.array([t[2][:L] for t in top], dtype=float)
    n_models   = len(model_cols)

    fig, axes = plt.subplots(1 + n_models, 1,
                             figsize=(min(L / 6, 22), top_n * 0.45 * (1 + n_models) + 2),
                             sharex=True)
    if n_models == 0:
        axes = [axes]
    fig.patch.set_facecolor('white')

    # GT
    im = axes[0].imshow(gt_mat, aspect='auto', cmap='Reds',
                        vmin=0, vmax=1, interpolation='nearest')
    axes[0].set_yticks(range(len(mirna_names)))
    axes[0].set_yticklabels(mirna_names, fontsize=7.5)
    axes[0].set_title('Ground Truth', fontsize=10, fontweight='bold', pad=3)
    plt.colorbar(im, ax=axes[0], fraction=0.012, pad=0.01)
    for x in [0, L - 1]:
        axes[0].axvline(x, color=BSJ_COLOR, lw=2, linestyle='--', alpha=0.8)

    # 모델별
    for idx, (m_label, m_col) in enumerate(model_cols.items()):
        pred_mat = np.array([t[3][m_label][:L] for t in top], dtype=float)
        ax = axes[1 + idx]
        color = get_model_color(m_label, idx)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            m_label, ['#f7f7f7', color])
        im = ax.imshow(pred_mat, aspect='auto', cmap=cmap,
                       vmin=0, vmax=1, interpolation='nearest')
        ax.set_yticks(range(len(mirna_names)))
        ax.set_yticklabels(mirna_names, fontsize=7.5)
        ax.set_title(f'{m_label}  (prediction)', fontsize=10,
                     fontweight='bold', pad=3, color=color)
        plt.colorbar(im, ax=ax, fraction=0.012, pad=0.01)
        for x in [0, L - 1]:
            ax.axvline(x, color=BSJ_COLOR, lw=2, linestyle='--', alpha=0.8)

    axes[-1].set_xlabel('Position', fontsize=10)
    iso_short = iso_full[:55] + '...' if len(iso_full) > 55 else iso_full
    fig.suptitle(f'Binding Site Heatmap  |  {iso_short}\n(top {top_n} miRNAs by BSJ-proximity)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()

    stem = f'viz_heatmap_{sanitize(iso_full[:40])}'
    _save(fig, out_dir, stem)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Overlay: 다중 모델 예측 오버레이 (상위 N miRNA)
# ══════════════════════════════════════════════════════════════════════════════
def plot_overlay(sub, iso_full, model_cols, top_n=6, out_dir=None):
    top = pick_top_mirnas(sub, model_cols, top_n)
    if not top:
        print("No data for overlay"); return
    L = top[0][4]

    ncols = 2
    nrows = (len(top) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(14, nrows * 2.5 + 1.5))
    fig.patch.set_facecolor('white')
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    x = np.arange(L)

    for idx, (mirna, bsj_score, gt, preds, _) in enumerate(top):
        ax = axes_flat[idx]
        ax.fill_between(x, gt, alpha=0.25, color=BIND_COLOR,
                        label='Ground Truth', zorder=1)
        ax.step(x, gt, color=BIND_COLOR, lw=0.8, alpha=0.5, zorder=2)
        for m_idx, (m_label, m_col) in enumerate(model_cols.items()):
            color = get_model_color(m_label, m_idx)
            ax.plot(x, preds[m_label][:L], color=color, lw=1.5,
                    alpha=0.85, label=m_label, zorder=3 + m_idx)
        ax.axvline(0,     color=BSJ_COLOR, lw=1.5, linestyle='--', alpha=0.7)
        ax.axvline(L - 1, color=BSJ_COLOR, lw=1.5, linestyle='--', alpha=0.7)
        ax.set_ylim(-0.05, 1.15)
        ax.set_title(f'{mirna}', fontsize=8.5, fontweight='bold', pad=3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.legend(fontsize=7, loc='upper right', framealpha=0.8)

    for ax in axes_flat[len(top):]:
        ax.set_visible(False)

    iso_short = iso_full[:55] + '...' if len(iso_full) > 55 else iso_full
    fig.suptitle(f'Multi-model Prediction Overlay  |  {iso_short}',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    _save(fig, out_dir, f'viz_overlay_{sanitize(iso_full[:40])}')


# ══════════════════════════════════════════════════════════════════════════════
# 3. Per-model circular diagram
# ══════════════════════════════════════════════════════════════════════════════
def _draw_circular(ax, gt, pred, L, model_label, color, title=''):
    angles  = np.linspace(90, 90 - 360, L, endpoint=False)
    arc_w   = 360 / L
    R_IN, R_OUT = 0.55, 0.78
    R_P_IN, R_P_OUT = R_OUT + 0.02, R_OUT + 0.08

    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal'); ax.axis('off')

    for i in range(L):
        ang = angles[i]
        fc  = BIND_COLOR if gt[i] > 0.5 else NOBIND_COLOR
        w   = Wedge((0, 0), R_OUT, ang - arc_w * 0.45, ang + arc_w * 0.45,
                    width=R_OUT - R_IN, facecolor=fc, edgecolor='white',
                    linewidth=0.15, alpha=1.0 if gt[i] > 0.5 else 0.4, zorder=2)
        ax.add_patch(w)

    if pred is not None:
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            'pred', ['#f7f7f7', color])
        for i in range(L):
            ang = angles[i]
            w = Wedge((0, 0), R_P_OUT, ang - arc_w * 0.45, ang + arc_w * 0.45,
                      width=R_P_OUT - R_P_IN,
                      facecolor=cmap(float(pred[i]) * 0.85 + 0.1),
                      edgecolor='none', alpha=0.9, zorder=3)
            ax.add_patch(w)

    # BSJ 마커
    bsj_rad = np.radians(90)
    ax.annotate('', xy=(np.cos(bsj_rad) * (R_P_OUT + 0.02),
                        np.sin(bsj_rad) * (R_P_OUT + 0.02)),
                xytext=(np.cos(bsj_rad) * (R_IN - 0.03),
                        np.sin(bsj_rad) * (R_IN - 0.03)),
                arrowprops=dict(arrowstyle='->', color=BSJ_COLOR, lw=2.0))
    ax.text(0, R_P_OUT + 0.15, 'BSJ', ha='center', fontsize=9,
            color=BSJ_COLOR, fontweight='bold')

    n_bind = int((np.array(gt) > 0.5).sum())
    ax.text(0,  0.08, f'{n_bind}/{L}', ha='center', fontsize=12,
            fontweight='bold', color='#2C3E50')
    ax.text(0, -0.10, 'binding\nsites', ha='center', fontsize=8.5,
            color='#777')

    if title:
        ax.set_title(title, fontsize=9, fontweight='bold', pad=5,
                     color=color if pred is not None else '#2C3E50')


def plot_per_model(sub, iso_full, model_cols, top_n=4, out_dir=None):
    """각 모델별 개별 PDF 저장 (circular + linear overlay)"""
    top = pick_top_mirnas(sub, model_cols, min(top_n, 4))
    if not top:
        print("No data for per_model"); return
    L = top[0][4]
    x = np.arange(L)

    for m_idx, (m_label, m_col) in enumerate(model_cols.items()):
        color = get_model_color(m_label, m_idx)
        ncases = len(top)
        fig = plt.figure(figsize=(5 * ncases, 11))
        fig.patch.set_facecolor('white')
        gs = GridSpec(2, ncases, figure=fig, hspace=0.45, wspace=0.25,
                      top=0.90, bottom=0.07)

        for col_i, (mirna, bsj_score, gt, preds, _) in enumerate(top):
            pred = preds[m_label][:L]

            # Row 0: circular
            ax_circ = fig.add_subplot(gs[0, col_i])
            _draw_circular(ax_circ, gt[:L], pred, L,
                           m_label, color,
                           title=f'{mirna}\n(BSJ prox={bsj_score:.0f})')

            # Row 1: linear overlay
            ax_lin = fig.add_subplot(gs[1, col_i])
            ax_lin.fill_between(x, gt[:L], alpha=0.25, color=BIND_COLOR,
                                label='GT', zorder=1)
            ax_lin.plot(x, pred, color=color, lw=1.5, label=m_label, zorder=2)
            ax_lin.axvline(0,     color=BSJ_COLOR, lw=1.5, linestyle='--', alpha=0.7)
            ax_lin.axvline(L - 1, color=BSJ_COLOR, lw=1.5, linestyle='--', alpha=0.7)
            ax_lin.set_ylim(-0.05, 1.15)
            ax_lin.spines['top'].set_visible(False)
            ax_lin.spines['right'].set_visible(False)
            ax_lin.tick_params(labelsize=7)
            if col_i == 0:
                ax_lin.legend(fontsize=8)

        iso_short = iso_full[:50] + '...' if len(iso_full) > 50 else iso_full
        fig.suptitle(f'[{m_label}]  {iso_short}',
                     fontsize=11, fontweight='bold', color=color)
        _save(fig, out_dir, f'viz_model_{m_label}_{sanitize(iso_full[:35])}')


# ══════════════════════════════════════════════════════════════════════════════
# 4. BSJ Zoom: BSJ 근처 50nt 확대 뷰
# ══════════════════════════════════════════════════════════════════════════════
def plot_bsj_zoom(sub, iso_full, model_cols, top_n=6, zoom_w=50, out_dir=None):
    top = pick_top_mirnas(sub, model_cols, top_n)
    if not top:
        print("No data for bsj_zoom"); return
    L = top[0][4]
    w = min(zoom_w, L // 2)

    ncols = 2
    nrows = (len(top) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols * 2,
                             figsize=(16, nrows * 2.8 + 1.5))
    fig.patch.set_facecolor('white')
    axes_flat = axes.flatten()

    for idx, (mirna, bsj_score, gt, preds, _) in enumerate(top):
        for side, (start, end, bsj_label) in enumerate([
            (0, w, "5' BSJ region"),
            (max(0, L - w), L, "3' BSJ region")
        ]):
            ax = axes_flat[idx * 2 + side]
            x  = np.arange(start, end)
            ax.fill_between(x, gt[start:end], alpha=0.25, color=BIND_COLOR,
                            label='GT', zorder=1)
            ax.step(x, gt[start:end], color=BIND_COLOR, lw=0.8,
                    alpha=0.5, zorder=2)
            for m_i, (m_label, _) in enumerate(model_cols.items()):
                color = get_model_color(m_label, m_i)
                ax.plot(x, preds[m_label][start:end], color=color,
                        lw=1.5, label=m_label, zorder=3 + m_i)
            bsj_x = 0 if side == 0 else L - 1
            ax.axvline(bsj_x, color=BSJ_COLOR, lw=2, linestyle='--', alpha=0.8)
            ax.set_ylim(-0.05, 1.15)
            ax.set_title(f'{mirna}\n{bsj_label}', fontsize=8, fontweight='bold')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(labelsize=7)
            if idx == 0 and side == 0:
                ax.legend(fontsize=7, loc='upper right')

    for ax in axes_flat[len(top) * 2:]:
        ax.set_visible(False)

    iso_short = iso_full[:55] + '...' if len(iso_full) > 55 else iso_full
    fig.suptitle(f'BSJ Region Zoom (±{w}nt)  |  {iso_short}',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    _save(fig, out_dir, f'viz_bsj_zoom_{sanitize(iso_full[:40])}')


# ══════════════════════════════════════════════════════════════════════════════
# 5. Model Summary: BSJ-proximal 예측 성능 비교 bar chart
# ══════════════════════════════════════════════════════════════════════════════
def plot_model_summary(sub, iso_full, model_cols, out_dir=None):
    if len(model_cols) < 2:
        print("model_summary requires >= 2 models"); return
    L = sub['position'].max() + 1
    w = 50

    bsj_mask = (sub['position'] < w) | (sub['position'] >= L - w)
    mid_mask  = ~bsj_mask

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor('white')

    for ax, mask, region in [(axes[0], bsj_mask, f'BSJ-proximal (±{w}nt)'),
                              (axes[1], mid_mask,  'Middle region')]:
        region_df = sub[mask]
        gt_bind   = region_df['ground_truth'].values
        labels, means, colors = [], [], []
        for m_i, (m_label, m_col) in enumerate(model_cols.items()):
            pred = region_df[m_col].values
            # binding site에서의 평균 예측값
            bind_idx = gt_bind > 0.5
            score = pred[bind_idx].mean() if bind_idx.any() else 0.0
            labels.append(m_label)
            means.append(score)
            colors.append(get_model_color(m_label, m_i))

        bars = ax.bar(range(len(labels)), means, color=colors,
                      edgecolor='white', linewidth=1.0, width=0.6)
        for bar, v in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f'{v:.3f}', ha='center', fontsize=9.5, fontweight='bold')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel('Mean pred. prob. at GT binding sites', fontsize=9)
        ax.set_title(region, fontsize=11, fontweight='bold')
        ax.set_ylim(0, max(means) * 1.3 if means else 1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', linestyle='--', alpha=0.3)

    iso_short = iso_full[:55] + '...' if len(iso_full) > 55 else iso_full
    fig.suptitle(f'Model Comparison: Prediction Score  |  {iso_short}',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    _save(fig, out_dir, f'viz_model_summary_{sanitize(iso_full[:40])}')


# ── 저장 유틸 ──────────────────────────────────────────────────────────────────
def _save(fig, out_dir, stem):
    out = Path(out_dir) if out_dir else Path(__file__).parent
    fig.savefig(out / f'{stem}.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(out / f'{stem}.png', bbox_inches='tight', dpi=150)
    print(f'Saved: {stem}.pdf/png')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    parser.add_argument('--csv',  required=True)
    parser.add_argument('--isoform', type=str, default=None)
    parser.add_argument('--top_mirna', type=int, default=12)
    parser.add_argument('--zoom_w', type=int, default=50,
                        help='BSJ zoom window size (default: 50)')
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--list_isoforms', action='store_true')
    parser.add_argument('--plots', nargs='+',
                        choices=['heatmap', 'overlay', 'per_model',
                                 'bsj_zoom', 'model_summary', 'all'],
                        default=['all'],
                        help='Which plots to generate (default: all)')
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    model_cols = get_model_cols(df)
    print(f'Models in CSV: {list(model_cols.keys())}')

    if args.list_isoforms:
        list_isoforms(df, model_cols)
        exit(0)

    # isoform 선택
    if args.isoform is None:
        best, best_score = None, -1
        for iso, grp in df.groupby('isoform_ID'):
            L = grp['position'].max() + 1
            w = 50
            bsj = int(grp[(grp['position'] < w) |
                          (grp['position'] >= L - w)]['ground_truth'].sum())
            score = grp['miRNA_ID'].nunique() * 10 + bsj
            if score > best_score:
                best_score, best = score, iso
        print(f'Auto-selected: {best[:60]}')
        sub, iso_full = df[df['isoform_ID'] == best], best
    else:
        sub, iso_full = load_isoform(df, args.isoform)

    L = sub['position'].max() + 1
    print(f'isoform: {iso_full[:60]}  L={L}  miRNAs={sub["miRNA_ID"].nunique()}')

    do_all   = 'all' in args.plots
    do_plot  = lambda name: do_all or name in args.plots

    if do_plot('heatmap'):
        plot_heatmap(sub, iso_full, model_cols,
                     top_n=args.top_mirna, out_dir=args.out_dir)
    if do_plot('overlay'):
        plot_overlay(sub, iso_full, model_cols,
                     top_n=min(args.top_mirna, 6), out_dir=args.out_dir)
    if do_plot('per_model'):
        plot_per_model(sub, iso_full, model_cols,
                       top_n=4, out_dir=args.out_dir)
    if do_plot('bsj_zoom'):
        plot_bsj_zoom(sub, iso_full, model_cols,
                      top_n=min(args.top_mirna, 6),
                      zoom_w=args.zoom_w, out_dir=args.out_dir)
    if do_plot('model_summary'):
        plot_model_summary(sub, iso_full, model_cols, out_dir=args.out_dir)
