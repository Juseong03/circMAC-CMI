"""
CSV raw prediction 값을 이용한 다양한 시각화

생성 그림:
  1. heatmap        - GT + 모델별 예측 히트맵 (isoform × miRNA)
  2. overlay        - 다중 모델 예측 오버레이 (상위 N miRNA)
  3. per_model      - 모델별 개별 저장 (circular + linear)
  4. bsj_zoom       - BSJ 근처 확대 뷰
  5. region_overlap - 모델간 region-overlap 기반 성능 비교
                      (Site Recall, Site Precision, Site F1, Mean IoU)
                      BSJ-proximal / Middle 구분

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


# ══════════════════════════════════════════════════════════════════════════════
# Position-level metrics 유틸
# ══════════════════════════════════════════════════════════════════════════════

def _fill_gaps(binary_arr, gap):
    """예측 binary에서 ≤gap 크기의 빈틈을 채워 연속 region으로 합침."""
    if gap <= 0:
        return binary_arr.copy()
    arr = np.asarray(binary_arr, dtype=int).copy()
    i = 0
    while i < len(arr):
        if arr[i] == 0:
            j = i
            while j < len(arr) and arr[j] == 0:
                j += 1
            # 빈틈이 양쪽 모두 1로 둘러싸여 있고 크기 ≤ gap이면 채움
            if (j - i) <= gap and i > 0 and j < len(arr):
                arr[i:j] = 1
            i = j
        else:
            i += 1
    return arr


def _dilate(binary_arr, tol):
    """binary mask를 ±tol 만큼 확장 (각 양성 위치 주변 tol bp 허용)."""
    if tol <= 0:
        return np.asarray(binary_arr, dtype=int)
    arr = np.asarray(binary_arr, dtype=int)
    from scipy.ndimage import binary_dilation
    struct = np.ones(2 * tol + 1, dtype=bool)
    return binary_dilation(arr.astype(bool), structure=struct).astype(int)


def _find_optimal_threshold(gt_arr, pred_prob):
    """F1을 최대화하는 threshold 탐색 (precision_recall_curve 기반)."""
    from sklearn.metrics import precision_recall_curve
    gt = np.asarray(gt_arr, dtype=int)
    if gt.sum() == 0 or (1 - gt).sum() == 0:
        return 0.5
    prec, rec, thrs = precision_recall_curve(gt, pred_prob)
    f1s = 2 * prec * rec / (prec + rec + 1e-8)
    best_idx = int(np.argmax(f1s))
    return float(thrs[best_idx]) if best_idx < len(thrs) else 0.5


def _compute_position_metrics(gt_arr, pred_prob, threshold=0.5, tol=0, gap=0):
    """
    Position-level metrics with tolerance & gap-filling.

    tol  : GT를 ±tol bp 확장 → 약간 어긋난 예측도 TP로 허용
    gap  : 예측 binary에서 ≤gap 크기 빈틈을 채워 연속 region 형성

    Recall    = GT 위치 중 ±tol 범위 안에 예측이 있는 비율
    Precision = 예측(gap-filled) 위치 중 ±tol 범위 안에 GT가 있는 비율
    F1        = 2*P*R / (P+R)
    """
    gt   = np.asarray(gt_arr, dtype=int)
    pred = _fill_gaps((np.asarray(pred_prob) >= threshold).astype(int), gap)

    # GT dilated → 예측이 근처에 있으면 TP
    gt_dilated   = _dilate(gt, tol)
    # Pred dilated → GT가 근처에 있으면 detected
    pred_dilated = _dilate(pred, tol)

    # Recall: 원래 GT 위치가 pred_dilated 안에 들어오는 비율
    n_gt = int(gt.sum())
    tp_r = int(((gt == 1) & (pred_dilated == 1)).sum())
    recall = tp_r / n_gt if n_gt > 0 else 0.0

    # Precision: pred(gap-filled) 위치가 gt_dilated 안에 들어오는 비율
    n_pred = int(pred.sum())
    tp_p   = int(((pred == 1) & (gt_dilated == 1)).sum())
    precision = tp_p / n_pred if n_pred > 0 else np.nan

    if np.isnan(precision) or (recall + precision) == 0:
        f1 = np.nan
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return dict(
        recall    = recall,
        precision = precision,
        f1        = f1,
        n_gt      = n_gt,
        n_pred    = n_pred,
    )


def _compute_region_metrics(gt_arr, pred_prob, threshold=0.5, iou_thresh=0.3):
    """
    GT binary array와 pred probability array로 region-overlap 기반 지표 계산.

    Returns dict:
      site_recall    - GT region 중 IoU >= iou_thresh 인 비율  (= TP_gt / n_gt)
      site_precision - Pred region 중 IoU >= iou_thresh 인 비율 (= TP_pred / n_pred)
      site_f1        - 2 * precision * recall / (precision + recall)
      mean_iou       - GT region별 best-match IoU 평균
      n_gt           - GT region 수
      n_pred         - Pred region 수
    """
    pred_bin     = (np.asarray(pred_prob) >= threshold).astype(int)
    gt_regions   = _get_regions(gt_arr)
    pred_regions = _get_regions(pred_bin)

    if not gt_regions:
        return dict(site_recall=np.nan, site_precision=np.nan,
                    site_f1=np.nan, mean_iou=np.nan,
                    n_gt=0, n_pred=len(pred_regions))

    # ── Recall: GT-centric ───────────────────────────────────────────────────
    ious, detected_gt = [], []
    for gt_r in gt_regions:
        if pred_regions:
            best_iou = max(_region_iou(gt_r, p) for p in pred_regions)
        else:
            best_iou = 0.0
        ious.append(best_iou)
        detected_gt.append(float(best_iou >= iou_thresh))

    site_recall = float(np.mean(detected_gt))

    # ── Precision: Pred-centric ──────────────────────────────────────────────
    if pred_regions:
        detected_pred = []
        for pred_r in pred_regions:
            best_iou = max(_region_iou(pred_r, g) for g in gt_regions)
            detected_pred.append(float(best_iou >= iou_thresh))
        site_precision = float(np.mean(detected_pred))
    else:
        site_precision = np.nan   # 예측 region 없음 → precision 정의 불가

    # ── F1 ──────────────────────────────────────────────────────────────────
    if np.isnan(site_precision) or (site_recall + site_precision) == 0:
        site_f1 = np.nan
    else:
        site_f1 = 2 * site_precision * site_recall / (site_precision + site_recall)

    return dict(
        site_recall    = site_recall,
        site_precision = site_precision,
        site_f1        = site_f1,
        mean_iou       = float(np.mean(ious)),
        n_gt           = len(gt_regions),
        n_pred         = len(pred_regions),
    )


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


def pick_top_mirnas(sub, model_cols, top_n, bsj_w=20, mode='bsj_prox'):
    L = sub['position'].max() + 1
    w = bsj_w
    rows = []
    for mirna, grp in sub.groupby('miRNA_ID'):
        grp = grp.sort_values('position')
        gt   = grp['ground_truth'].values
        if gt.sum() == 0:           # binding 없는 pair 제외
            continue
        preds = {m: grp[col].values for m, col in model_cols.items()}
        bsj  = gt[:w].sum() + gt[max(0, L - w):].sum()
        rows.append((mirna, bsj, gt, preds, L))
    rows.sort(key=lambda x: -x[1])
    return rows[:top_n]


def list_isoforms(df, model_cols, bsj_w=20):
    print(f"{'isoform_ID':<65} {'L':>5} {'n_mirna':>8} {'bsj_prox':>9} {'models'}")
    print('-' * 100)
    results = []
    for iso, grp in df.groupby('isoform_ID'):
        L = grp['position'].max() + 1
        n_mirna = grp['miRNA_ID'].nunique()
        w = bsj_w
        bsj = int(grp[(grp['position'] < w) | (grp['position'] >= L - w)]['ground_truth'].sum())
        results.append((iso, L, n_mirna, bsj))
    results.sort(key=lambda x: (-x[2], -x[3]))
    mnames = list(model_cols.keys())
    for iso, L, n_mirna, bsj in results:
        print(f"{iso[:63]:<65} {L:>5} {n_mirna:>8} {bsj:>9}  {mnames}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. Heatmap: GT + 모델별 히트맵 (miRNA × position)
# ══════════════════════════════════════════════════════════════════════════════
def plot_heatmap(sub, iso_full, model_cols, top_n=12, bsj_w=20, out_dir=None):
    top = pick_top_mirnas(sub, model_cols, top_n, bsj_w=bsj_w)
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
def plot_overlay(sub, iso_full, model_cols, top_n=6, bsj_w=20, out_dir=None):
    top = pick_top_mirnas(sub, model_cols, top_n, bsj_w=bsj_w)
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


def plot_per_model(sub, iso_full, model_cols, top_n=4, bsj_w=20, out_dir=None):
    """각 모델별 개별 PDF 저장 (circular + linear overlay)"""
    top = pick_top_mirnas(sub, model_cols, min(top_n, 4), bsj_w=bsj_w)
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
# 5. Per-pair: 각 miRNA pair별 개별 figure (전체 모델 오버레이 + F1 bar)
# ══════════════════════════════════════════════════════════════════════════════
def plot_per_pair(sub, iso_full, model_cols, bsj_w=20,
                  threshold=0.5, iou_thresh=0.3, tol=0, gap=0,
                  opt_threshold=False, val_thresholds=None, out_dir=None):
    """miRNA pair마다 하나의 figure: 좌=라인 오버레이, 우=모델별 F1 bar.

    threshold 우선순위: val_thresholds (from val set) > opt_threshold (oracle) > threshold (fixed)
    """
    if not model_cols:
        print("per_pair: no model columns found"); return

    model_list  = list(model_cols.keys())
    iso_short   = iso_full[:45] + '...' if len(iso_full) > 45 else iso_full
    generated   = 0

    for mirna, grp in sub.groupby('miRNA_ID'):
        grp = grp.sort_values('position')
        gt  = grp['ground_truth'].values
        if gt.sum() == 0:
            continue
        L = len(gt)
        x = np.arange(L)

        # ── 모델별 threshold 결정 및 metrics 계산 ────────────────────────────
        # 우선순위: val_thresholds > opt_threshold (oracle) > fixed threshold
        use_val = bool(val_thresholds)
        show_both = (tol > 0 or gap > 0)
        model_thresholds = {}
        metrics_strict = {}    # tol=0, gap=0
        metrics_constr = {}    # tol, gap as passed
        for m_label, m_col in model_cols.items():
            pred = grp[m_col].values[:L]
            if use_val and m_label in val_thresholds:
                thr = float(val_thresholds[m_label])
            elif opt_threshold:
                thr = _find_optimal_threshold(gt, pred)
            else:
                thr = threshold
            model_thresholds[m_label] = thr
            ms = _compute_position_metrics(gt, pred, threshold=thr, tol=0, gap=0)
            mc = _compute_position_metrics(gt, pred, threshold=thr, tol=tol, gap=gap)
            try:
                from sklearn.metrics import roc_auc_score
                auroc = float(roc_auc_score(gt, pred)) if gt.sum() > 0 and (1-gt).sum() > 0 else np.nan
            except Exception:
                auroc = np.nan
            ms['auroc'] = auroc
            mc['auroc'] = auroc
            metrics_strict[m_label] = ms
            metrics_constr[m_label] = mc

        # ── Figure: 좌(3) line overlay / 우(2) AUROC+F1 bar ──────────────────
        fig = plt.figure(figsize=(16, 4.5))
        fig.patch.set_facecolor('white')
        gs = GridSpec(1, 5, figure=fig, wspace=0.35,
                      left=0.05, right=0.97, top=0.88, bottom=0.15)
        ax_line  = fig.add_subplot(gs[0, :3])
        ax_auroc = fig.add_subplot(gs[0, 3])
        ax_f1    = fig.add_subplot(gs[0, 4])

        # ── 라인 오버레이 ──────────────────────────────────────────────────────
        ax_line.fill_between(x, gt, alpha=0.20, color=BIND_COLOR,
                             label='Ground Truth', zorder=1)
        ax_line.step(x, gt, color=BIND_COLOR, lw=0.8, alpha=0.5, zorder=2)
        for m_idx, (m_label, m_col) in enumerate(model_cols.items()):
            color = get_model_color(m_label, m_idx)
            pred  = grp[m_col].values[:L]
            ax_line.plot(x, pred, color=color, lw=1.3, alpha=0.85,
                         label=m_label, zorder=3 + m_idx)
            if opt_threshold:
                thr = model_thresholds[m_label]
                ax_line.axhline(thr, color=color, lw=0.7, ls=':',
                                alpha=0.5, zorder=2)
        if not opt_threshold:
            ax_line.axhline(threshold, color='#888', lw=0.9, ls=':',
                            alpha=0.7, label=f'threshold={threshold}')
        ax_line.axvline(0,     color=BSJ_COLOR, lw=1.5, ls='--', alpha=0.7)
        ax_line.axvline(L - 1, color=BSJ_COLOR, lw=1.5, ls='--', alpha=0.7)
        ax_line.set_xlim(0, L - 1)
        ax_line.set_ylim(-0.05, 1.15)
        ax_line.set_xlabel('Position', fontsize=9)
        ax_line.set_ylabel('Prediction probability', fontsize=9)
        ax_line.legend(fontsize=7.5, loc='upper right', ncol=2,
                       framealpha=0.85, edgecolor='#ccc')
        ax_line.spines['top'].set_visible(False)
        ax_line.spines['right'].set_visible(False)
        ax_line.tick_params(labelsize=8)

        bsj_prox = int(gt[:bsj_w].sum() + gt[max(0, L - bsj_w):].sum())
        n_gt_sites = int(gt.sum())
        ax_line.set_title(
            f'{mirna}  |  GT binding sites: {n_gt_sites}/{L}'
            f'  (BSJ-proximal: {bsj_prox})',
            fontsize=10, fontweight='bold')

        # ── AUROC horizontal bar (threshold-independent) ──────────────────────
        colors_bar = [get_model_color(ml, i) for i, ml in enumerate(model_list)]
        y = np.arange(len(model_list))

        auroc_vals = []
        for ml in model_list:
            v = metrics_strict[ml].get('auroc', np.nan)
            auroc_vals.append(0.5 if (v is None or np.isnan(v)) else float(v))
        ax_auroc.barh(y, auroc_vals, color=colors_bar, height=0.6, edgecolor='white')
        ax_auroc.axvline(0.5, color='#aaa', lw=0.8, ls='--')
        for yi, v in zip(y, auroc_vals):
            ax_auroc.text(min(v + 0.01, 1.02), yi, f'{v:.2f}',
                          va='center', fontsize=7.5, fontweight='bold')
        ax_auroc.set_yticks(y)
        ax_auroc.set_yticklabels(model_list, fontsize=8)
        ax_auroc.set_xlim(0.3, 1.05)
        ax_auroc.set_xlabel('AUROC', fontsize=8.5)
        ax_auroc.set_title('AUROC', fontsize=9, fontweight='bold', color='#2980B9')
        ax_auroc.spines['top'].set_visible(False)
        ax_auroc.spines['right'].set_visible(False)
        ax_auroc.tick_params(axis='x', labelsize=7)
        ax_auroc.invert_yaxis()

        # ── F1 horizontal bar: strict + constrained ────────────────────────────
        if use_val:
            thr_str = 'val_thr'
        elif opt_threshold:
            thr_str = 'opt_thr'
        else:
            thr_str = f'thr={threshold}'

        if show_both:
            bh = 0.28
            ys = y - bh / 2 - 0.03
            yc = y + bh / 2 + 0.03
            f1_strict = [max(0.0, v if not np.isnan(v) else 0.0)
                         for v in [metrics_strict[ml].get('f1', 0.0) or 0.0 for ml in model_list]]
            f1_constr = [max(0.0, v if not np.isnan(v) else 0.0)
                         for v in [metrics_constr[ml].get('f1', 0.0) or 0.0 for ml in model_list]]
            ax_f1.barh(ys, f1_strict, color=colors_bar, height=bh, edgecolor='white',
                       alpha=0.50, hatch='////', label='strict (nt-level)')
            ax_f1.barh(yc, f1_constr, color=colors_bar, height=bh, edgecolor='white',
                       alpha=1.0, label=f'tol={tol},gap={gap}')
            for yi, v in zip(ys, f1_strict):
                if v > 0.01:
                    ax_f1.text(v + 0.01, yi, f'{v:.2f}',
                               va='center', fontsize=6.5, alpha=0.7)
            for yi, v in zip(yc, f1_constr):
                if v > 0.01:
                    ax_f1.text(v + 0.01, yi, f'{v:.2f}',
                               va='center', fontsize=7.5, fontweight='bold')
            ax_f1.legend(fontsize=6.5, loc='lower right', framealpha=0.85)
            ax_f1.set_title(f'F1  ({thr_str})\nstrict vs constrained',
                            fontsize=8.5, fontweight='bold', color='#8E44AD')
        else:
            f1_vals = [max(0.0, v if not np.isnan(v) else 0.0)
                       for v in [metrics_strict[ml].get('f1', 0.0) or 0.0 for ml in model_list]]
            ax_f1.barh(y, f1_vals, color=colors_bar, height=0.6, edgecolor='white')
            for yi, v in zip(y, f1_vals):
                if v > 0.01:
                    ax_f1.text(v + 0.01, yi, f'{v:.2f}',
                               va='center', fontsize=7.5, fontweight='bold')
            ax_f1.set_title(f'F1  ({thr_str})',
                            fontsize=9, fontweight='bold', color='#8E44AD')
        ax_f1.set_yticks(y)
        ax_f1.set_yticklabels(model_list, fontsize=8)
        ax_f1.set_xlim(0, 1.05)
        ax_f1.set_xlabel('F1', fontsize=8.5)
        ax_f1.spines['top'].set_visible(False)
        ax_f1.spines['right'].set_visible(False)
        ax_f1.tick_params(axis='x', labelsize=7)
        ax_f1.invert_yaxis()

        fig.suptitle(iso_short, fontsize=9, color='#555', y=0.97)
        plt.tight_layout()

        stem = f'viz_pair_{sanitize(mirna[:35])}_{sanitize(iso_full[:25])}'
        _save(fig, out_dir, stem)
        generated += 1

    if generated == 0:
        print("per_pair: no binding pairs found")
    else:
        print(f'per_pair: generated {generated} figures')


# ══════════════════════════════════════════════════════════════════════════════
# 6. Site Metrics: position-level Recall / Precision / F1
# ══════════════════════════════════════════════════════════════════════════════
def plot_region_overlap(sub, iso_full, model_cols, bsj_w=20,
                        threshold=0.5, iou_thresh=0.3, tol=0, gap=0,
                        opt_threshold=False, val_thresholds=None, out_dir=None):
    """
    Position-level Recall / Precision / F1 모델 비교 (bar chart).
    tol>0 or gap>0인 경우, strict (nt-level, tol=0, gap=0) 와
    constrained (tol, gap) 를 grouped bar로 나란히 표시.
    AUROC는 threshold-independent이므로 하나만 표시.
    threshold 우선순위: val_thresholds > opt_threshold (oracle) > threshold (fixed)
    """
    if not model_cols:
        print("region_overlap: no model columns found"); return

    show_both = (tol > 0 or gap > 0)

    METRICS_DUAL = [
        ('recall',    'Recall',    '#E74C3C'),
        ('precision', 'Precision', '#E67E22'),
        ('f1',        'F1',        '#8E44AD'),
    ]

    # ── 집계: miRNA × model ───────────────────────────────────────────────────
    from collections import defaultdict
    results_strict = defaultdict(list)      # tol=0, gap=0
    results_constr = defaultdict(list)      # tol, gap as passed

    for mirna, grp in sub.groupby('miRNA_ID'):
        grp = grp.sort_values('position')
        gt  = grp['ground_truth'].values
        if gt.sum() == 0:
            continue
        use_val = bool(val_thresholds)
        for m_label, m_col in model_cols.items():
            pred = grp[m_col].values
            if use_val and m_label in (val_thresholds or {}):
                thr = float(val_thresholds[m_label])
            elif opt_threshold:
                thr = _find_optimal_threshold(gt, pred)
            else:
                thr = threshold
            m_s = _compute_position_metrics(gt, pred, threshold=thr, tol=0, gap=0)
            m_c = _compute_position_metrics(gt, pred, threshold=thr, tol=tol, gap=gap)
            try:
                from sklearn.metrics import roc_auc_score
                auroc = float(roc_auc_score(gt, pred)) if gt.sum() > 0 and (1-gt).sum() > 0 else np.nan
            except Exception:
                auroc = np.nan
            m_s['auroc'] = auroc
            m_c['auroc'] = auroc
            if m_s['n_gt'] > 0:
                results_strict[m_label].append(m_s)
                results_constr[m_label].append(m_c)

    # ── 그리기 ────────────────────────────────────────────────────────────────
    # Layout: 3 dual-metric panels (Recall/Prec/F1) + 1 AUROC panel
    n_panels = 4
    fig, axes = plt.subplots(1, n_panels, figsize=(16.0, 4.8), sharey=False)
    fig.patch.set_facecolor('white')

    model_labels = list(model_cols.keys())
    x = np.arange(len(model_labels))

    def _agg(results_dict, key):
        vals, errs = [], []
        for ml in model_labels:
            data = results_dict.get(ml, [])
            vl = [d[key] for d in data if d[key] is not None and not np.isnan(d[key])]
            if vl:
                vals.append(float(np.mean(vl)))
                errs.append(float(np.std(vl) / np.sqrt(len(vl))) if len(vl) > 1 else 0.0)
            else:
                vals.append(0.0)
                errs.append(0.0)
        return vals, errs

    colors_bar = [get_model_color(ml, i) for i, ml in enumerate(model_labels)]

    # ── Recall / Precision / F1: grouped bars (strict + constrained) ──────────
    for m_i, (metric_key, metric_label, bar_color) in enumerate(METRICS_DUAL):
        ax = axes[m_i]

        vals_s, errs_s = _agg(results_strict, metric_key)
        vals_c, errs_c = _agg(results_constr, metric_key)

        if show_both:
            bw = 0.38
            xs = x - bw / 2 - 0.01
            xc = x + bw / 2 + 0.01
            # strict bars (hatched)
            bars_s = ax.bar(xs, vals_s, yerr=errs_s, color=colors_bar,
                            width=bw, edgecolor='white', alpha=0.55,
                            hatch='////', capsize=3,
                            error_kw=dict(lw=1.2, alpha=0.6), label='strict (nt-level)')
            # constrained bars (solid)
            bars_c = ax.bar(xc, vals_c, yerr=errs_c, color=colors_bar,
                            width=bw, edgecolor='white', alpha=1.0,
                            capsize=3, error_kw=dict(lw=1.2, alpha=0.6),
                            label=f'constrained (tol={tol}, gap={gap})')
            max_e = max(max(errs_s), max(errs_c)) if (errs_s or errs_c) else 0
            for bar, v in zip(bars_s, vals_s):
                if v > 0.02:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + max_e + 0.01,
                            f'{v:.2f}', ha='center', va='bottom',
                            fontsize=7.0, fontweight='bold', alpha=0.65)
            for bar, v in zip(bars_c, vals_c):
                if v > 0.02:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + max_e + 0.01,
                            f'{v:.2f}', ha='center', va='bottom',
                            fontsize=7.5, fontweight='bold')
            if m_i == 0:
                ax.legend(fontsize=7, loc='upper right', framealpha=0.85)
        else:
            bw = 0.6
            bars = ax.bar(x, vals_s, yerr=errs_s, color=colors_bar,
                          width=bw, edgecolor='white',
                          capsize=4, error_kw=dict(lw=1.5, alpha=0.7))
            max_e = max(errs_s) if errs_s else 0
            for bar, v in zip(bars, vals_s):
                if v > 0.005:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + max_e * 1.1 + 0.01,
                            f'{v:.2f}', ha='center', va='bottom',
                            fontsize=8.5, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, fontsize=9, rotation=30, ha='right')
        ax.set_ylim(0, 1.22)
        ax.set_title(metric_label, fontsize=12, fontweight='bold', color=bar_color)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', linestyle='--', alpha=0.3)

    # ── AUROC: single bar (threshold-independent) ─────────────────────────────
    ax_a = axes[3]
    vals_a, errs_a = _agg(results_strict, 'auroc')
    bw = 0.6
    bars_a = ax_a.bar(x, vals_a, yerr=errs_a, color=colors_bar,
                      width=bw, edgecolor='white',
                      capsize=4, error_kw=dict(lw=1.5, alpha=0.7))
    max_e = max(errs_a) if errs_a else 0
    for bar, v in zip(bars_a, vals_a):
        if v > 0.005:
            ax_a.text(bar.get_x() + bar.get_width() / 2,
                      bar.get_height() + max_e * 1.1 + 0.01,
                      f'{v:.2f}', ha='center', va='bottom',
                      fontsize=8.5, fontweight='bold')
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(model_labels, fontsize=9, rotation=30, ha='right')
    ax_a.set_ylim(0, 1.22)
    ax_a.set_title('AUROC', fontsize=12, fontweight='bold', color='#2980B9')
    ax_a.axhline(0.5, color='#aaa', lw=0.8, linestyle='--')
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)
    ax_a.grid(axis='y', linestyle='--', alpha=0.3)

    iso_short = iso_full[:55] + '...' if len(iso_full) > 55 else iso_full
    n_pairs   = sum(1 for _, grp in sub.groupby('miRNA_ID')
                    if grp['ground_truth'].sum() > 0)
    if val_thresholds:
        thr_label = 'val_thr (per-model, from val set)'
    elif opt_threshold:
        thr_label = 'opt_thr (per-model, oracle)'
    else:
        thr_label = f'threshold={threshold}'
    dual_note = f'  |  hatched=strict(nt-level), solid=constrained(tol={tol},gap={gap})' if show_both else ''
    fig.suptitle(
        f'Binding Site Prediction  |  {iso_short}\n'
        f'({thr_label},  averaged over {n_pairs} miRNA pairs{dual_note})',
        fontsize=10, fontweight='bold'
    )
    plt.tight_layout()
    _save(fig, out_dir, f'viz_site_metrics_{sanitize(iso_full[:40])}')


# ── 저장 유틸 ──────────────────────────────────────────────────────────────────
_SAVE_PDF = True   # --no_pdf 플래그로 제어

def _save(fig, out_dir, stem):
    out = Path(out_dir) if out_dir else Path(__file__).parent
    if _SAVE_PDF:
        fig.savefig(out / f'{stem}.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(out / f'{stem}.png', bbox_inches='tight', dpi=150)
    saved = f'{stem}.png' + (f' + .pdf' if _SAVE_PDF else '')
    print(f'Saved: {saved}')
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
    parser.add_argument('--bsj_w', type=int, default=20,
                        help='BSJ-proximal window size for region analysis (default: 20)')
    parser.add_argument('--zoom_w', type=int, default=50,
                        help='BSJ zoom window size for bsj_zoom plot (default: 50)')
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--list_isoforms', action='store_true')
    parser.add_argument('--plots', nargs='+',
                        choices=['heatmap', 'overlay', 'per_model',
                                 'bsj_zoom', 'region_overlap', 'per_pair', 'all'],
                        default=['all'],
                        help='Which plots to generate (default: all)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Pred binarization threshold (default: 0.5)')
    parser.add_argument('--iou_thresh', type=float, default=0.3,
                        help='IoU threshold (legacy, unused) (default: 0.3)')
    parser.add_argument('--tol', type=int, default=5,
                        help='GT tolerance: ±tol bp 범위 안 예측도 TP로 허용 (default: 5)')
    parser.add_argument('--gap', type=int, default=3,
                        help='Prediction gap-fill: ≤gap bp 빈틈은 연속 region으로 합침 (default: 3)')
    parser.add_argument('--opt_threshold', action='store_true',
                        help='Use per-model optimal threshold (F1-maximizing on TEST data — oracle). '
                             'For proper evaluation use --threshold_file instead.')
    parser.add_argument('--threshold_file', type=str, default=None,
                        help='JSON file with per-model thresholds from val set '
                             '(e.g. docs/paper_cmi/model_thresholds_s1.json). '
                             'Overrides --threshold and --opt_threshold.')
    parser.add_argument('--no_pdf', action='store_true',
                        help='Skip PDF output, save PNG only')
    parser.add_argument('--mirna_ids', nargs='+', default=None,
                        help='Filter to specific miRNA IDs (e.g. hsa-miR-449a hsa-miR-34b-5p). '
                             'If not given, all binding pairs are used.')
    args = parser.parse_args()

    if args.no_pdf:
        _SAVE_PDF = False

    # ── per-model threshold 결정 ─────────────────────────────────────────────
    # 우선순위: --threshold_file > --opt_threshold > --threshold (fixed)
    import json as _json
    val_thresholds = {}   # {'circmac': 0.712, ...}  — 비어 있으면 fixed threshold 사용
    if args.threshold_file:
        with open(args.threshold_file) as _f:
            val_thresholds = _json.load(_f)
        print(f'Loaded val thresholds from: {args.threshold_file}')
        print(f'  {val_thresholds}')

    df = pd.read_csv(args.csv)
    model_cols = get_model_cols(df)
    print(f'Models in CSV: {list(model_cols.keys())}')

    if args.list_isoforms:
        list_isoforms(df, model_cols, bsj_w=args.bsj_w)
        exit(0)

    # isoform 선택
    if args.isoform is None:
        best, best_score = None, -1
        for iso, grp in df.groupby('isoform_ID'):
            L = grp['position'].max() + 1
            w = args.bsj_w
            bsj = int(grp[(grp['position'] < w) |
                          (grp['position'] >= L - w)]['ground_truth'].sum())
            score = grp['miRNA_ID'].nunique() * 10 + bsj
            if score > best_score:
                best_score, best = score, iso
        print(f'Auto-selected: {best[:60]}')
        sub, iso_full = df[df['isoform_ID'] == best], best
    else:
        sub, iso_full = load_isoform(df, args.isoform)

    # miRNA 필터링
    if args.mirna_ids:
        before = sub['miRNA_ID'].nunique()
        sub = sub[sub['miRNA_ID'].isin(args.mirna_ids)]
        print(f'miRNA filter: {before} → {sub["miRNA_ID"].nunique()} pairs '
              f'({args.mirna_ids})')

    L = sub['position'].max() + 1
    print(f'isoform: {iso_full[:60]}  L={L}  miRNAs={sub["miRNA_ID"].nunique()}')

    do_all   = 'all' in args.plots
    do_plot  = lambda name: do_all or name in args.plots

    if do_plot('heatmap'):
        plot_heatmap(sub, iso_full, model_cols,
                     top_n=args.top_mirna, bsj_w=args.bsj_w, out_dir=args.out_dir)
    if do_plot('overlay'):
        plot_overlay(sub, iso_full, model_cols,
                     top_n=min(args.top_mirna, 6), bsj_w=args.bsj_w, out_dir=args.out_dir)
    if do_plot('per_model'):
        plot_per_model(sub, iso_full, model_cols,
                       top_n=4, bsj_w=args.bsj_w, out_dir=args.out_dir)
    if do_plot('bsj_zoom'):
        plot_bsj_zoom(sub, iso_full, model_cols,
                      top_n=min(args.top_mirna, 6),
                      zoom_w=args.zoom_w, out_dir=args.out_dir)
    if do_plot('region_overlap'):
        plot_region_overlap(sub, iso_full, model_cols,
                            bsj_w=args.bsj_w,
                            threshold=args.threshold,
                            iou_thresh=args.iou_thresh,
                            tol=args.tol,
                            gap=args.gap,
                            opt_threshold=args.opt_threshold,
                            val_thresholds=val_thresholds,
                            out_dir=args.out_dir)
    if do_plot('per_pair'):
        plot_per_pair(sub, iso_full, model_cols,
                      bsj_w=args.bsj_w,
                      threshold=args.threshold,
                      iou_thresh=args.iou_thresh,
                      tol=args.tol,
                      gap=args.gap,
                      opt_threshold=args.opt_threshold,
                      val_thresholds=val_thresholds,
                      out_dir=args.out_dir)
