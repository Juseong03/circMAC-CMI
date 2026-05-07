"""
3개 case (circCDYL2, circMAPK1, circAPP)를 하나의 figure로 통합.

Layout (6 row × 3 col):
  Row 0 : Heatmap     — GT (상단) + circmac 예측 (하단)  [imshow, miRNA × position]
  Row 1 : Circular    — circmac 예측 (원형 다이어그램, representative pair)
  Row 2 : 5' BSJ zoom — representative pair, 전 모델 오버레이
  Row 3 : 3' BSJ zoom — same pair
  Row 4 : F1 bar      — strict (nt-level) vs constrained (tol, gap)
  Row 5 : AUROC bar

사용법:
  python docs/paper_cmi/plot_combined_cases.py \
      --threshold_file docs/paper_cmi/model_thresholds_s1.json \
      --out_dir docs/paper_cmi/paper_figures_v2
"""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Wedge
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np
import pandas as pd


# ── 색상 ──────────────────────────────────────────────────────────────────────
BSJ_COLOR    = '#F39C12'
BIND_COLOR   = '#E74C3C'
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
FALLBACK = ['#E67E22','#2980B9','#27AE60','#8E44AD',
            '#16A085','#C0392B','#F39C12','#95A5A6']

def mcolor(label, idx=0):
    return MODEL_COLORS.get(label, FALLBACK[idx % len(FALLBACK)])


# ── Metrics helpers ────────────────────────────────────────────────────────────
def _fill_gaps(arr, gap):
    if gap <= 0: return arr.copy()
    a = arr.copy()
    i = 0
    while i < len(a):
        if a[i] == 0:
            j = i
            while j < len(a) and a[j] == 0: j += 1
            if (j - i) <= gap and i > 0 and j < len(a): a[i:j] = 1
            i = j
        else: i += 1
    return a

def _dilate(arr, tol):
    if tol <= 0: return arr.astype(int)
    from scipy.ndimage import binary_dilation
    return binary_dilation(arr.astype(bool), structure=np.ones(2*tol+1, bool)).astype(int)

def compute_metrics(gt, pred_prob, thr, tol=0, gap=0):
    gt   = np.asarray(gt, int)
    pred = _fill_gaps((np.asarray(pred_prob) >= thr).astype(int), gap)
    gt_d = _dilate(gt, tol); pd_ = _dilate(pred, tol)
    n_gt = gt.sum(); n_pr = pred.sum()
    recall    = ((gt==1)&(pd_==1)).sum() / n_gt  if n_gt > 0 else 0.0
    precision = ((pred==1)&(gt_d==1)).sum() / n_pr if n_pr > 0 else np.nan
    f1 = (2*precision*recall/(precision+recall)
          if not np.isnan(precision) and (precision+recall)>0 else np.nan)
    try:
        from sklearn.metrics import roc_auc_score
        auroc = float(roc_auc_score(gt, pred_prob)) if gt.sum()>0 and (1-gt).sum()>0 else np.nan
    except Exception:
        auroc = np.nan
    return dict(recall=recall, precision=precision, f1=f1, auroc=auroc)


# ── Case 정의 ──────────────────────────────────────────────────────────────────
# ── Model group definitions ───────────────────────────────────────────────────
MODEL_GROUPS = {
    'all':        None,   # all models in CSV
    'encoder':    ['circmac', 'mamba', 'lstm', 'transformer', 'hymba'],
    'pretrained': ['circmac', 'rnabert', 'rnaernie', 'rnamsm', 'rnafm'],
}

# ── Case definitions: single representative pair per case ─────────────────────
CASES_SINGLE = [
    dict(
        label     = 'circCDYL2  (chr4)',
        gene      = 'CDYL2',
        csv       = 'docs/paper_cmi/results/chr4_84678168_84679116_test_binding_only_bsjw20_s1/'
                    'binding_visualization_chr4_84678168_84679116_with_pred.csv',
        isoform   = 'chr4|84678168',
        # miR-449a: miR-34/449 family (p53 target, tumor suppressor), AUROC=1.00
        rep_mirna = 'hsa-miR-449a',
        mirna_ids = ['hsa-miR-449a'],
    ),
    dict(
        label     = 'circMAPK1  (chr22)',
        gene      = 'MAPK1',
        csv       = 'docs/paper_cmi/results/chr22_21799012_21805850_21807664_test_binding_only_bsjw20_s1/'
                    'binding_visualization_chr22_21799012_21805850_21807664_with_pred.csv',
        isoform   = 'chr22|21799012',
        # miR-12119: most BSJ-proximal, AUROC=1.00
        rep_mirna = 'hsa-miR-12119',
        mirna_ids = ['hsa-miR-12119'],
    ),
    dict(
        label     = 'circAPP  (chr21)',
        gene      = 'APP',
        csv       = 'docs/paper_cmi/results/chr21_25954590_25955627_25975070_25975954_25982344_test_binding_only_bsjw20_s1/'
                    'binding_visualization_chr21_25954590_25955627_25975070_25975954_25982344_with_pred.csv',
        isoform   = 'chr21|25954590',
        # miR-5001-3p: APP/Alzheimer's context, AUROC=0.999
        rep_mirna = 'hsa-miR-5001-3p',
        mirna_ids = ['hsa-miR-5001-3p'],
    ),
]

# ── Case definitions: all pairs per case ──────────────────────────────────────
CASES_ALL = [
    dict(
        label     = 'circCDYL2  (chr4)',
        gene      = 'CDYL2',
        csv       = 'docs/paper_cmi/results/chr4_84678168_84679116_test_binding_only_bsjw20_s1/'
                    'binding_visualization_chr4_84678168_84679116_with_pred.csv',
        isoform   = 'chr4|84678168',
        rep_mirna = 'hsa-miR-449a',
        mirna_ids = ['hsa-miR-34b-5p', 'hsa-miR-34c-5p', 'hsa-miR-449a', 'hsa-miR-449b-5p'],
    ),
    dict(
        label     = 'circMAPK1  (chr22)',
        gene      = 'MAPK1',
        csv       = 'docs/paper_cmi/results/chr22_21799012_21805850_21807664_test_binding_only_bsjw20_s1/'
                    'binding_visualization_chr22_21799012_21805850_21807664_with_pred.csv',
        isoform   = 'chr22|21799012',
        rep_mirna = 'hsa-miR-12119',
        mirna_ids = None,   # all 2 pairs
    ),
    dict(
        label     = 'circAPP  (chr21)',
        gene      = 'APP',
        csv       = 'docs/paper_cmi/results/chr21_25954590_25955627_25975070_25975954_25982344_test_binding_only_bsjw20_s1/'
                    'binding_visualization_chr21_25954590_25955627_25975070_25975954_25982344_with_pred.csv',
        isoform   = 'chr21|25954590',
        rep_mirna = 'hsa-miR-5001-3p',
        mirna_ids = ['hsa-miR-5001-3p', 'hsa-miR-1236-3p', 'hsa-miR-4732-5p'],
    ),
]

# backward-compat alias (used by existing code)
CASES = CASES_SINGLE


# ── Data loading ───────────────────────────────────────────────────────────────
def load_case(case, val_thresholds, bsj_w=20):
    df   = pd.read_csv(case['csv'])
    mask = df['isoform_ID'].str.contains(case['isoform'], regex=False)
    sub  = df[mask].copy()
    iso  = sub['isoform_ID'].iloc[0]
    if case['mirna_ids']:
        sub = sub[sub['miRNA_ID'].isin(case['mirna_ids'])]
    model_cols = {c[5:]: c for c in df.columns if c.startswith('pred_')}
    L = sub['position'].max() + 1
    w = bsj_w
    rep = case['rep_mirna']
    if rep is None:
        best, best_v = None, -1
        for mirna, grp in sub.groupby('miRNA_ID'):
            gt = grp.sort_values('position')['ground_truth'].values
            if gt.sum() == 0: continue
            v = int(gt[:w].sum() + gt[max(0,L-w):].sum())
            if v > best_v: best_v, best = v, mirna
        rep = best
    return sub, iso, model_cols, rep


def compute_case_metrics(sub, model_cols, val_thresholds, tol=0, gap=0):
    results_s  = defaultdict(list)   # val-thr, strict
    results_c  = defaultdict(list)   # val-thr, constrained
    results_05 = defaultdict(list)   # thr=0.5, strict
    for mirna, grp in sub.groupby('miRNA_ID'):
        grp = grp.sort_values('position')
        gt  = grp['ground_truth'].values
        if gt.sum() == 0: continue
        for m_label, m_col in model_cols.items():
            pred = grp[m_col].values
            thr  = float(val_thresholds.get(m_label, 0.5))
            results_s[m_label].append(compute_metrics(gt, pred, thr,  0,   0))
            results_c[m_label].append(compute_metrics(gt, pred, thr,  tol, gap))
            results_05[m_label].append(compute_metrics(gt, pred, 0.5, 0,   0))
    def _mean(rd, key):
        out = {}
        for ml, lst in rd.items():
            vs = [d[key] for d in lst if d[key] is not None and not np.isnan(d[key])]
            out[ml] = float(np.mean(vs)) if vs else 0.0
        return out
    return {
        'strict': {k: _mean(results_s,  k) for k in ['f1','auroc','recall','precision']},
        'constr': {k: _mean(results_c,  k) for k in ['f1','auroc','recall','precision']},
        'thr05':  {k: _mean(results_05, k) for k in ['f1','auroc','recall','precision']},
    }


# ── Draw helpers ───────────────────────────────────────────────────────────────

def draw_heatmap(ax_gt, ax_model, sub, model_cols, bsj_w=20,
                 val_thresholds=None, top_n=6, model_name='circmac'):
    """
    ax_gt    : GT imshow (miRNA × position, red)
    ax_model : model prediction imshow (miRNA × position, model color)
    Shows top_n miRNAs sorted by BSJ-proximal GT sites.
    """
    # pick top miRNAs
    L = sub['position'].max() + 1
    w = bsj_w
    rows = []
    for mirna, grp in sub.groupby('miRNA_ID'):
        grp = grp.sort_values('position')
        gt  = grp['ground_truth'].values
        if gt.sum() == 0: continue
        bsj = int(gt[:w].sum() + gt[max(0,L-w):].sum())
        rows.append((mirna, bsj, gt))
    rows.sort(key=lambda x: -x[1])
    rows = rows[:top_n]
    if not rows:
        ax_gt.set_visible(False); ax_model.set_visible(False); return

    mirna_labels = [r[0].replace('hsa-','') for r in rows]
    gt_mat = np.zeros((len(rows), L), dtype=float)
    for i, (_, _, gt_arr) in enumerate(rows):
        gt_mat[i, :len(gt_arr)] = gt_arr[:L]

    # find model col
    m_col = model_cols.get(model_name)
    if m_col is None:
        m_col = list(model_cols.values())[0]
        model_name = list(model_cols.keys())[0]

    pred_mat = np.zeros_like(gt_mat)
    for i, (mirna, _, _) in enumerate(rows):
        grp  = sub[sub['miRNA_ID']==mirna].sort_values('position')
        vals = grp[m_col].values
        n    = min(len(vals), L)
        pred_mat[i, :n] = vals[:n]

    def _draw(ax, mat, cmap, title, color, vmax=1.0):
        ax.imshow(mat, aspect='auto', cmap=cmap,
                  vmin=0, vmax=vmax, interpolation='nearest')
        ax.set_yticks(range(len(mirna_labels)))
        ax.set_yticklabels(mirna_labels, fontsize=7)
        ax.set_title(title, fontsize=8.5, fontweight='bold', pad=3, color=color)
        for x in [0, L-1]:
            ax.axvline(x, color=BSJ_COLOR, lw=1.5, ls='--', alpha=0.8)
        ax.tick_params(axis='x', labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    _draw(ax_gt, gt_mat, 'Reds', 'Ground Truth', BIND_COLOR)
    color = mcolor(model_name)
    cmap_m = matplotlib.colors.LinearSegmentedColormap.from_list(
        model_name, ['#f7f7f7', color])
    _draw(ax_model, pred_mat, cmap_m, f'{model_name}  prediction', color)
    ax_model.set_xlabel('Position', fontsize=8)


def draw_circular(ax, sub, rep_mirna, model_cols, val_thresholds,
                  model_name='circmac', title=''):
    """원형 다이어그램: GT (inner ring) + model prediction (outer ring)."""
    grp = sub[sub['miRNA_ID']==rep_mirna].sort_values('position')
    if grp.empty: ax.set_visible(False); return
    gt   = grp['ground_truth'].values
    L    = len(gt)
    m_col = model_cols.get(model_name, list(model_cols.values())[0])
    pred = grp[m_col].values[:L]
    color = mcolor(model_name)

    angles = np.linspace(90, 90-360, L, endpoint=False)
    arc_w  = 360 / L
    R_IN, R_OUT = 0.52, 0.74
    R_P_IN, R_P_OUT = R_OUT+0.02, R_OUT+0.10

    ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.35, 1.35)
    ax.set_aspect('equal'); ax.axis('off')

    # GT ring
    for i in range(L):
        ang = angles[i]
        fc  = BIND_COLOR if gt[i] > 0.5 else NOBIND_COLOR
        w   = Wedge((0,0), R_OUT, ang-arc_w*0.45, ang+arc_w*0.45,
                    width=R_OUT-R_IN, facecolor=fc, edgecolor='white',
                    linewidth=0.1, alpha=1.0 if gt[i]>0.5 else 0.35, zorder=2)
        ax.add_patch(w)

    # Prediction ring
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('p', ['#f7f7f7', color])
    for i in range(L):
        ang = angles[i]
        w   = Wedge((0,0), R_P_OUT, ang-arc_w*0.45, ang+arc_w*0.45,
                    width=R_P_OUT-R_P_IN,
                    facecolor=cmap(float(pred[i])*0.85+0.1),
                    edgecolor='none', alpha=0.9, zorder=3)
        ax.add_patch(w)

    # BSJ marker
    bsj_rad = np.radians(90)
    ax.annotate('', xy=(np.cos(bsj_rad)*(R_P_OUT+0.03), np.sin(bsj_rad)*(R_P_OUT+0.03)),
                xytext=(np.cos(bsj_rad)*(R_IN-0.04), np.sin(bsj_rad)*(R_IN-0.04)),
                arrowprops=dict(arrowstyle='->', color=BSJ_COLOR, lw=1.8))
    ax.text(0, R_P_OUT+0.18, 'BSJ', ha='center', fontsize=8,
            color=BSJ_COLOR, fontweight='bold')

    # legend
    legend_handles = [
        mpatches.FancyArrow(0, 0, 0, 0, color=BIND_COLOR, label='GT binding'),
        mpatches.FancyArrow(0, 0, 0, 0, color=color, label=f'{model_name} pred'),
    ]
    ax.legend(handles=[
        plt.Line2D([0],[0], color=BIND_COLOR, lw=6, alpha=0.85, label='GT binding'),
        plt.Line2D([0],[0], color=color,      lw=6, alpha=0.75, label=f'{model_name} pred'),
    ], fontsize=6.5, loc='lower center', framealpha=0.75,
       bbox_to_anchor=(0.5, -0.12), ncol=2)

    n_bind = int((gt > 0.5).sum())
    ax.text(0,  0.08, f'{n_bind}/{L}', ha='center', fontsize=11,
            fontweight='bold', color='#2C3E50')
    ax.text(0, -0.12, 'binding\nsites', ha='center', fontsize=8, color='#777')

    mirna_short = rep_mirna.replace('hsa-','')
    full_title  = f'{title}\n{mirna_short}' if title else mirna_short
    ax.set_title(full_title, fontsize=8.5, fontweight='bold', pad=4, color=color)


def draw_bsj_zoom(ax, sub, iso, rep_mirna, model_cols, val_thresholds,
                  side='5p', zoom_w=50, title='', show_legend=False):
    grp = sub[sub['miRNA_ID']==rep_mirna].sort_values('position')
    if grp.empty: ax.set_visible(False); return
    gt = grp['ground_truth'].values
    L  = len(gt)
    w  = min(zoom_w, L//2)

    if side == '5p':
        s, e = 0, w
        x_rel   = np.arange(w)
        bsj_x   = 0
        xlabel  = "Distance from 5' BSJ (nt)"
    else:
        s, e    = max(0, L-w), L
        x_rel   = np.arange(-(e-s), 0)
        bsj_x   = 0
        xlabel  = "Distance from 3' BSJ (nt)"

    ax.fill_between(x_rel, gt[s:e], alpha=0.18, color=BIND_COLOR,
                    label='Ground Truth', zorder=1)
    ax.step(x_rel, gt[s:e], color=BIND_COLOR, lw=0.8, alpha=0.5, zorder=2)
    for m_i, (m_label, m_col) in enumerate(model_cols.items()):
        pred  = grp[m_col].values
        color = mcolor(m_label, m_i)
        ax.plot(x_rel, pred[s:e], color=color, lw=1.4, alpha=0.85,
                label=m_label, zorder=3+m_i)
        thr = float(val_thresholds.get(m_label, 0.5))
        ax.axhline(thr, color=color, lw=0.5, ls=':', alpha=0.30)

    ax.axvline(bsj_x, color=BSJ_COLOR, lw=2, ls='--', alpha=0.8)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlim(x_rel[0], x_rel[-1])
    region = "5' BSJ" if side=='5p' else "3' BSJ"
    mirna_short = rep_mirna.replace('hsa-','')
    head = f'{title}  |  ' if title else ''
    ax.set_title(f'{head}{mirna_short}  —  {region} (±{w} nt)',
                 fontsize=8.0, fontweight='bold', pad=2)
    ax.set_xlabel(xlabel, fontsize=7.5)
    ax.set_ylabel('Pred. prob.', fontsize=7.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=7)
    if show_legend:
        ax.legend(fontsize=6.5, loc='upper right', framealpha=0.85,
                  ncol=1, handlelength=1.2)


METRICS_4 = [
    ('recall',    'Recall',    '#E74C3C', 0.0, 1.0),
    ('precision', 'Precision', '#E67E22', 0.0, 1.0),
    ('f1',        'F1',        '#8E44AD', 0.0, 1.0),
    ('auroc',     'AUROC',     '#2980B9', 0.3, 1.0),
]

def draw_metrics_4(axes, metrics, model_cols):
    """
    4 axes: Recall / Precision / F1 / AUROC.
    Recall/Precision/F1: grouped bars — val-thr (solid) vs thr=0.5 (hatched).
    AUROC: single bar (threshold-independent).
    """
    model_list = list(model_cols.keys())
    colors = [mcolor(ml, i) for i, ml in enumerate(model_list)]
    y  = np.arange(len(model_list))
    bh = 0.35   # bar height for each of the two groups

    for ax, (key, label, color, xlim_lo, xlim_hi) in zip(axes, METRICS_4):
        if key == 'auroc':
            # AUROC: single bar (threshold-independent)
            vals = [metrics['strict'][key].get(ml, 0.0) for ml in model_list]
            ax.barh(y, vals, color=colors, height=0.55, edgecolor='white', alpha=0.92)
            ax.axvline(0.5, color='#bbb', lw=0.8, ls='--')
            for yi, v in zip(y, vals):
                if v > 0.02:
                    ax.text(min(v + 0.01, xlim_hi - 0.02), yi,
                            f'{v:.2f}', va='center', fontsize=7.0, fontweight='bold')
            ax.set_yticks(y)
            ax.set_yticklabels(model_list, fontsize=7.5)
        else:
            # Grouped bars: val-thr (top, solid) vs thr=0.5 (bottom, hatched)
            vals_v = [metrics['strict'][key].get(ml, 0.0) for ml in model_list]
            vals_5 = [metrics['thr05'][key].get(ml, 0.0)  for ml in model_list]
            y_v = y - bh / 2   # val-thr bar position
            y_5 = y + bh / 2   # thr=0.5 bar position

            ax.barh(y_v, vals_v, color=colors, height=bh,
                    edgecolor='white', alpha=0.92, label='val-thr')
            ax.barh(y_5, vals_5, color=colors, height=bh,
                    edgecolor='white', alpha=0.45, hatch='///', label='thr=0.5')

            for yi, v in zip(y_v, vals_v):
                if v > 0.02:
                    ax.text(min(v + 0.01, xlim_hi - 0.02), yi,
                            f'{v:.2f}', va='center', fontsize=6.5, fontweight='bold')
            for yi, v in zip(y_5, vals_5):
                if v > 0.02:
                    ax.text(min(v + 0.01, xlim_hi - 0.02), yi,
                            f'{v:.2f}', va='center', fontsize=6.5, color='#555')

            ax.set_yticks(y)
            ax.set_yticklabels(model_list, fontsize=7.5)
            if key == 'recall':
                ax.legend(fontsize=6, loc='lower right', framealpha=0.7,
                          handlelength=1.0, handletextpad=0.4)

        ax.set_xlim(xlim_lo, xlim_hi + 0.12)
        ax.set_xlabel(label, fontsize=8)
        ax.set_title(label, fontsize=8.5, fontweight='bold', color=color, pad=3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='x', labelsize=7)
        ax.invert_yaxis()


# ══════════════════════════════════════════════════════════════════════════════
# Comparison Figure A: GT + All-model Heatmaps (10 rows × 3 cols)
# ══════════════════════════════════════════════════════════════════════════════
def make_heatmap_comparison(case_data, val_thresholds, bsj_w=20, top_n=4,
                            out_dir='.', no_pdf=False, suffix=''):
    """
    viz_heatmap_comparison.png/pdf
    Row 0       : GT  heatmap
    Row 1..N    : model prediction heatmap (one row per model)
    Col 0..2    : case (circCDYL2 / circMAPK1 / circAPP)
    """
    model_list = list(case_data[0]['model_cols'].keys())
    n_models   = len(model_list)
    n_cols     = len(case_data)
    top_n      = min(top_n, 6)
    hm_h       = max(top_n * 0.28 + 0.4, 1.2)   # height per heatmap row

    fig_w = 6.0 * n_cols
    fig_h = hm_h * (1 + n_models) + 1.5
    fig   = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor('white')

    gs = GridSpec(1 + n_models, n_cols, figure=fig,
                  hspace=0.60, wspace=0.30,
                  top=0.93, bottom=0.03, left=0.08, right=0.97,
                  height_ratios=[hm_h] * (1 + n_models))

    for col_i, cd in enumerate(case_data):
        sub        = cd['sub']
        model_cols = cd['model_cols']
        L          = sub['position'].max() + 1

        # ── pick top miRNAs (BSJ-proximal) ────────────────────────────────────
        w    = bsj_w
        rows = []
        for mirna, grp in sub.groupby('miRNA_ID'):
            grp = grp.sort_values('position')
            gt  = grp['ground_truth'].values
            if gt.sum() == 0: continue
            bsj = int(gt[:w].sum() + gt[max(0,L-w):].sum())
            rows.append((mirna, bsj, gt))
        rows.sort(key=lambda x: -x[1])
        rows = rows[:top_n]
        if not rows: continue

        mirna_labels = [r[0].replace('hsa-','') for r in rows]
        gt_mat = np.zeros((len(rows), L), dtype=float)
        for i, (_, _, gt_arr) in enumerate(rows):
            gt_mat[i, :min(len(gt_arr), L)] = gt_arr[:L]

        def _hm(ax, mat, cmap, title, title_color, show_yticks=True):
            ax.imshow(mat, aspect='auto', cmap=cmap, vmin=0, vmax=1,
                      interpolation='nearest')
            ax.set_yticks(range(len(mirna_labels)))
            ax.set_yticklabels(mirna_labels if show_yticks else ['']*len(mirna_labels),
                               fontsize=6.5)
            ax.set_title(title, fontsize=8, fontweight='bold', pad=2, color=title_color)
            for x in [0, L-1]:
                ax.axvline(x, color=BSJ_COLOR, lw=1.2, ls='--', alpha=0.75)
            ax.tick_params(axis='x', labelsize=6)

        # Row 0: GT
        ax_gt = fig.add_subplot(gs[0, col_i])
        _hm(ax_gt, gt_mat, 'Reds',
            f'{cd["label"]}  (n={cd["n_pairs"]} pairs)\nGround Truth',
            BIND_COLOR, show_yticks=True)
        ax_gt.set_xlabel('Position', fontsize=7)

        # Rows 1..N: per-model prediction
        for m_i, m_label in enumerate(model_list):
            m_col = model_cols.get(m_label)
            if m_col is None: continue
            pred_mat = np.zeros_like(gt_mat)
            for i, (mirna, _, _) in enumerate(rows):
                grp  = sub[sub['miRNA_ID']==mirna].sort_values('position')
                vals = grp[m_col].values
                n    = min(len(vals), L)
                pred_mat[i, :n] = vals[:n]
            color = mcolor(m_label, m_i)
            cmap  = matplotlib.colors.LinearSegmentedColormap.from_list(
                        m_label, ['#f8f8f8', color])
            ax = fig.add_subplot(gs[1 + m_i, col_i])
            _hm(ax, pred_mat, cmap, m_label, color, show_yticks=(col_i==0))
            if m_i == n_models - 1:
                ax.set_xlabel('Position', fontsize=7)

    model_list = list(case_data[0]['model_cols'].keys())
    grp_label = f'Models: {", ".join(model_list)}'
    fig.suptitle(f'Heatmap Comparison  |  {grp_label}',
                 fontsize=11, fontweight='bold')
    out = Path(out_dir)
    stem = f'viz_heatmap_comparison{suffix}'
    fig.savefig(out / f'{stem}.png', bbox_inches='tight', dpi=150)
    print(f'Saved: {out}/{stem}.png')
    if not no_pdf:
        fig.savefig(out / f'{stem}.pdf', bbox_inches='tight', dpi=300)
        print(f'Saved: {out}/{stem}.pdf')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Comparison Figure B: All-model Circular Diagrams (3×3 grid × 3 cases)
# ══════════════════════════════════════════════════════════════════════════════
def make_circular_comparison(case_data, val_thresholds, out_dir='.', no_pdf=False, suffix=''):
    """
    viz_circular_comparison.png/pdf
    3 cases as major columns; within each case: 3×3 grid of 9 model circulars.
    """
    model_list = list(case_data[0]['model_cols'].keys())
    n_models   = len(model_list)
    n_cases    = len(case_data)

    # Each case column → 3 rows × 3 cols circular grid
    CIRC_ROWS, CIRC_COLS = 3, 3
    circ_sz = 2.8   # size of each circular cell (inches)

    fig_w = circ_sz * CIRC_COLS * n_cases + (n_cases - 1) * 0.5 + 1.0
    fig_h = circ_sz * CIRC_ROWS + 1.5
    fig   = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor('white')

    # top-level: 1 row × n_cases cols (each case occupies 3 sub-cols)
    outer_gs = GridSpec(1, n_cases, figure=fig,
                        hspace=0.1, wspace=0.12,
                        top=0.90, bottom=0.05, left=0.02, right=0.98)

    for col_i, cd in enumerate(case_data):
        sub        = cd['sub']
        model_cols = cd['model_cols']
        rep        = cd['rep']

        grp = sub[sub['miRNA_ID']==rep].sort_values('position')
        if grp.empty: continue
        gt   = grp['ground_truth'].values
        L    = len(gt)

        # nested 3×3 GridSpec for this case
        inner_gs = GridSpecFromSubplotSpec(CIRC_ROWS, CIRC_COLS,
                                           subplot_spec=outer_gs[0, col_i],
                                           hspace=0.50, wspace=0.15)

        # Case label as text above the grid (using figure text)
        # We'll put it as the title of the first circular in the grid instead

        for m_i, m_label in enumerate(model_list):
            r, c = divmod(m_i, CIRC_COLS)
            ax   = fig.add_subplot(inner_gs[r, c])

            m_col = model_cols.get(m_label)
            if m_col is None:
                ax.set_visible(False); continue

            pred  = grp[m_col].values[:L]
            color = mcolor(m_label, m_i)

            # Draw circular diagram
            angles = np.linspace(90, 90-360, L, endpoint=False)
            arc_w  = 360 / L
            R_IN, R_OUT = 0.50, 0.72
            R_P_IN, R_P_OUT = R_OUT+0.02, R_OUT+0.12

            ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
            ax.set_aspect('equal'); ax.axis('off')

            for i in range(L):
                ang = angles[i]
                fc  = BIND_COLOR if gt[i]>0.5 else NOBIND_COLOR
                w   = Wedge((0,0), R_OUT, ang-arc_w*0.45, ang+arc_w*0.45,
                            width=R_OUT-R_IN, facecolor=fc, edgecolor='white',
                            linewidth=0.08, alpha=1.0 if gt[i]>0.5 else 0.32, zorder=2)
                ax.add_patch(w)

            cmap_p = matplotlib.colors.LinearSegmentedColormap.from_list(
                        'p', ['#f8f8f8', color])
            for i in range(L):
                ang = angles[i]
                w   = Wedge((0,0), R_P_OUT, ang-arc_w*0.45, ang+arc_w*0.45,
                            width=R_P_OUT-R_P_IN,
                            facecolor=cmap_p(float(pred[i])*0.85+0.1),
                            edgecolor='none', alpha=0.9, zorder=3)
                ax.add_patch(w)

            bsj_rad = np.radians(90)
            ax.annotate('', xy=(np.cos(bsj_rad)*(R_P_OUT+0.02),
                                np.sin(bsj_rad)*(R_P_OUT+0.02)),
                        xytext=(np.cos(bsj_rad)*(R_IN-0.04),
                                np.sin(bsj_rad)*(R_IN-0.04)),
                        arrowprops=dict(arrowstyle='->', color=BSJ_COLOR, lw=1.5))
            ax.text(0, R_P_OUT+0.20, 'BSJ', ha='center', fontsize=7,
                    color=BSJ_COLOR, fontweight='bold')
            n_bind = int((gt>0.5).sum())
            ax.text(0,  0.06, f'{n_bind}/{L}', ha='center', fontsize=9,
                    fontweight='bold', color='#2C3E50')
            ax.text(0, -0.12, 'bind\nsites', ha='center', fontsize=7, color='#777')
            ax.set_title(m_label, fontsize=8.5, fontweight='bold',
                         pad=3, color=color)

            # case label on top-left cell only
            if m_i == 0:
                mirna_short = rep.replace('hsa-','')
                ax.text(0.02, 1.10,
                        f'{cd["label"]}\n{mirna_short}',
                        transform=ax.transAxes, fontsize=7.5,
                        fontweight='bold', va='top', color='#2C3E50')

    # shared legend
    legend_handles = [
        plt.Line2D([0],[0], color=BIND_COLOR, lw=6, alpha=0.85, label='GT binding'),
        plt.Line2D([0],[0], color='#888',     lw=6, alpha=0.30, label='GT non-binding'),
        plt.Line2D([0],[0], color='#aaa',     lw=6, alpha=0.90, label='model pred (outer ring)'),
    ]
    fig.legend(handles=legend_handles, fontsize=8, loc='lower center',
               ncol=3, bbox_to_anchor=(0.5, 0.00), framealpha=0.85)

    model_list = list(case_data[0]['model_cols'].keys())
    grp_label  = f'Models: {", ".join(model_list)}'
    fig.suptitle(f'Circular Diagram Comparison  |  {grp_label}',
                 fontsize=11, fontweight='bold')
    out = Path(out_dir)
    stem = f'viz_circular_comparison{suffix}'
    fig.savefig(out / f'{stem}.png', bbox_inches='tight', dpi=150)
    print(f'Saved: {out}/{stem}.png')
    if not no_pdf:
        fig.savefig(out / f'{stem}.pdf', bbox_inches='tight', dpi=300)
        print(f'Saved: {out}/{stem}.pdf')
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--threshold_file', default='docs/paper_cmi/model_thresholds_s1.json')
    parser.add_argument('--tol',       type=int, default=5)
    parser.add_argument('--gap',       type=int, default=3)
    parser.add_argument('--zoom_w',    type=int, default=50)
    parser.add_argument('--bsj_w',     type=int, default=20)
    parser.add_argument('--top_mirna', type=int, default=4,
                        help='# miRNAs shown in heatmap (default: 4)')
    parser.add_argument('--circ_model', default='circmac',
                        help='Model for circular/heatmap pred panel (default: circmac)')
    parser.add_argument('--out_dir', default='docs/paper_cmi/paper_figures_v2')
    parser.add_argument('--no_pdf', action='store_true')
    parser.add_argument('--mode', nargs='+',
                        choices=['combined', 'heatmap_cmp', 'circular_cmp', 'all'],
                        default=['all'],
                        help='Which figures to generate (default: all)')
    parser.add_argument('--pairs', choices=['single', 'all', 'both'],
                        default='single',
                        help='single=1 pair/case  all=all pairs  both=generate both versions')
    parser.add_argument('--model_group', nargs='+',
                        choices=['all', 'encoder', 'pretrained'],
                        default=['all'],
                        help='Model subset to show (default: all). '
                             'encoder=circmac/mamba/lstm/transformer/hymba  '
                             'pretrained=circmac+RNA-LMs  '
                             'Pass multiple to generate multiple versions.')
    args = parser.parse_args()

    with open(args.threshold_file) as f:
        val_thresholds = json.load(f)
    print(f'Thresholds: {val_thresholds}')

    # ── pair versions to generate ───────────────────────────────────────────────
    pair_versions = []
    if args.pairs == 'both':
        pair_versions = [('single', CASES_SINGLE), ('all', CASES_ALL)]
    elif args.pairs == 'all':
        pair_versions = [('all', CASES_ALL)]
    else:
        pair_versions = [('single', CASES_SINGLE)]

    # ── model group versions to generate ───────────────────────────────────────
    group_versions = args.model_group  # e.g. ['all', 'encoder', 'pretrained']

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for pair_tag, cases_def in pair_versions:
        for grp_tag in group_versions:
            _run_one(args, val_thresholds, cases_def, pair_tag, grp_tag, out)


def _run_one(args, val_thresholds, cases_def, pair_tag, grp_tag, out):
    """Generate all figures for one (pair_version, model_group) combination."""
    model_filter = MODEL_GROUPS[grp_tag]   # None = all, else list of names
    suffix = f'_{pair_tag}' if pair_tag != 'single' else ''
    suffix += f'_{grp_tag}' if grp_tag != 'all' else ''
    print(f'\n{"="*60}')
    print(f'  pairs={pair_tag}  model_group={grp_tag}  suffix="{suffix}"')
    print(f'{"="*60}')

    # ── 로드 ───────────────────────────────────────────────────────────────────
    case_data = []
    for case in cases_def:
        sub, iso, model_cols, rep = load_case(case, val_thresholds, args.bsj_w)
        # filter model columns to selected group
        if model_filter is not None:
            model_cols = {k: v for k, v in model_cols.items() if k in model_filter}
        metrics = compute_case_metrics(sub, model_cols, val_thresholds, args.tol, args.gap)
        n_pairs = sum(1 for _, g in sub.groupby('miRNA_ID') if g['ground_truth'].sum()>0)
        case_data.append(dict(
            label=case['label'], sub=sub, iso=iso,
            model_cols=model_cols, rep=rep,
            metrics=metrics, n_pairs=n_pairs,
        ))
        print(f"  {case['label']}: rep={rep}, pairs={n_pairs}, models={list(model_cols.keys())}")

    n_models  = len(case_data[0]['model_cols'])
    n_cols    = len(cases_def)
    # metrics 2×2 grid height (2 rows of grouped bars)
    metrics_h = n_models * 0.42 * 2 + 2.2

    # ── Figure / GridSpec ──────────────────────────────────────────────────────
    # 6 row × 3 col:
    #   row 0 : GT heatmap
    #   row 1 : circmac heatmap
    #   row 2 : circular diagram
    #   row 3 : 5' BSJ zoom
    #   row 4 : 3' BSJ zoom
    #   row 5 : 2×2 metrics (Recall/Prec/F1/AUROC, strict nt-level)
    top_n  = min(args.top_mirna, 6)
    hm_h   = max(top_n * 0.30 + 0.5, 1.5)
    circ_h = 3.2
    zoom_h = 2.2

    fig_h = hm_h*2 + circ_h + zoom_h*2 + metrics_h + 1.8
    fig   = plt.figure(figsize=(6.5 * n_cols, fig_h))
    fig.patch.set_facecolor('white')

    gs = GridSpec(6, n_cols, figure=fig,
                  hspace=0.65, wspace=0.42,
                  top=0.93, bottom=0.03,
                  left=0.07, right=0.97,
                  height_ratios=[hm_h, hm_h, circ_h, zoom_h, zoom_h, metrics_h])

    for col_i, cd in enumerate(case_data):
        n_p = cd['n_pairs']
        lbl = cd['label']

        # ── Row 0: GT heatmap ─────────────────────────────────────────────────
        ax_gt = fig.add_subplot(gs[0, col_i])
        # ── Row 1: circmac heatmap ────────────────────────────────────────────
        ax_hm = fig.add_subplot(gs[1, col_i])
        draw_heatmap(ax_gt, ax_hm, cd['sub'], cd['model_cols'],
                     bsj_w=args.bsj_w, val_thresholds=val_thresholds,
                     top_n=top_n, model_name=args.circ_model)
        # case title only on GT panel; clear inner title to avoid overlap
        ax_gt.set_title(f'{lbl}  (n={n_p} pairs)\nGround Truth',
                        fontsize=8.5, fontweight='bold', pad=4, color=BIND_COLOR)
        ax_hm.set_title(f'{args.circ_model}  prediction',
                        fontsize=8.0, fontweight='bold', pad=4,
                        color=mcolor(args.circ_model))

        # ── Row 2: Circular ───────────────────────────────────────────────────
        ax_circ = fig.add_subplot(gs[2, col_i])
        draw_circular(ax_circ, cd['sub'], cd['rep'],
                      cd['model_cols'], val_thresholds,
                      model_name=args.circ_model, title='')
        # case label as subtitle inside circular
        ax_circ.set_title(f'{args.circ_model}  |  {cd["rep"].replace("hsa-","")}',
                          fontsize=8.0, fontweight='bold', pad=3,
                          color=mcolor(args.circ_model))

        # ── Row 3: 5' BSJ zoom ────────────────────────────────────────────────
        ax_5p = fig.add_subplot(gs[3, col_i])
        draw_bsj_zoom(ax_5p, cd['sub'], cd['iso'], cd['rep'],
                      cd['model_cols'], val_thresholds,
                      side='5p', zoom_w=args.zoom_w,
                      title='', show_legend=(col_i == 0))

        # ── Row 4: 3' BSJ zoom ────────────────────────────────────────────────
        ax_3p = fig.add_subplot(gs[4, col_i])
        draw_bsj_zoom(ax_3p, cd['sub'], cd['iso'], cd['rep'],
                      cd['model_cols'], val_thresholds,
                      side='3p', zoom_w=args.zoom_w,
                      title='', show_legend=False)

        # ── Row 5: 2×2 metrics grid (Recall / Precision / F1 / AUROC) ─────────
        inner = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[5, col_i],
                                        hspace=0.70, wspace=0.55)
        metric_axes = [
            fig.add_subplot(inner[0, 0]),  # Recall
            fig.add_subplot(inner[0, 1]),  # Precision
            fig.add_subplot(inner[1, 0]),  # F1
            fig.add_subplot(inner[1, 1]),  # AUROC
        ]
        draw_metrics_4(metric_axes, cd['metrics'], cd['model_cols'])

    grp_label = {'all': 'All Models', 'encoder': 'Encoder Architectures',
                 'pretrained': 'Pretrained RNA LMs (+ CircMAC)'}[grp_tag]
    fig.suptitle(
        f'circRNA Binding Site Prediction: Case Studies  [{grp_label}]\n'
        f'Metrics: nucleotide-level strict (tol=0)  |  solid=val-thr  ///=thr0.5',
        fontsize=11, fontweight='bold', y=0.995)

    do_all      = 'all' in args.mode
    do_combined = do_all or 'combined'     in args.mode
    do_hm_cmp   = do_all or 'heatmap_cmp' in args.mode
    do_circ_cmp = do_all or 'circular_cmp' in args.mode

    def _save_fig(fig, stem):
        fig.savefig(out / f'{stem}.png', bbox_inches='tight', dpi=150)
        print(f'Saved: {out}/{stem}.png')
        if not args.no_pdf:
            fig.savefig(out / f'{stem}.pdf', bbox_inches='tight', dpi=300)
            print(f'Saved: {out}/{stem}.pdf')
        plt.close(fig)

    if do_combined:
        _save_fig(fig, f'viz_combined_cases{suffix}')
    else:
        plt.close(fig)

    if do_hm_cmp:
        print('\n[Heatmap comparison figure]')
        make_heatmap_comparison(case_data, val_thresholds,
                                bsj_w=args.bsj_w, top_n=top_n,
                                out_dir=str(out), no_pdf=args.no_pdf,
                                suffix=suffix)

    if do_circ_cmp:
        print('\n[Circular comparison figure]')
        make_circular_comparison(case_data, val_thresholds,
                                 out_dir=str(out), no_pdf=args.no_pdf,
                                 suffix=suffix)


if __name__ == '__main__':
    main()
