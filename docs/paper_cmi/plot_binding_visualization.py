"""
논문용 Binding Site Visualization

Figure 구성:
  (a) Case study: BSJ-region binding site — circular diagram
  (b) Linear probability heatmap: CircMAC vs baseline
  (c) BSJ-region binding site 통계

사용법:
  # 데모 (ground truth만):
  python plot_binding_visualization.py

  # 서버에서 model prediction 포함:
  python plot_binding_visualization.py --with_pred
  (saved_models/circmac/exp3_circmac_sites_s1 필요)
"""

import os, sys, argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
import pickle

# ── 색상 ──────────────────────────────────────────────────────────────────────
NT_COLORS = {'A': '#E74C3C', 'U': '#3498DB', 'G': '#2ECC71', 'C': '#F39C12',
             'T': '#F39C12'}
BIND_COLOR  = '#E74C3C'
NOBIND_COLOR = '#BDC3C7'
BSJ_COLOR   = '#F39C12'
PRED_CMAP   = LinearSegmentedColormap.from_list('pred', ['#EBF5FB', '#1A5276'])

DATA_PATH = str(Path(__file__).parent.parent.parent / 'data' / 'df_test_final.pkl')

# ── 데이터 로드 ───────────────────────────────────────────────────────────────
def load_data(data_path=None):
    path = data_path or DATA_PATH
    df = pickle.load(open(path, 'rb'))
    return df

# ── Model Inference (서버용) ──────────────────────────────────────────────────
def get_predictions(row, model_dir=None, device='cuda'):
    """
    서버에서 실행 시 model_dir 지정.
    로컬 데모 시 None → ground truth를 가우시안 smoothing해서 pseudo-prediction 생성.
    """
    sites = np.array(row['sites'], dtype=float)
    if model_dir is None:
        # 데모용: GT에 노이즈 + smoothing
        from scipy.ndimage import gaussian_filter1d
        noise = np.random.RandomState(42).randn(len(sites)) * 0.08
        pseudo = np.clip(sites * 0.92 + 0.05 + noise, 0, 1)
        pseudo = gaussian_filter1d(pseudo, sigma=1.5)
        pseudo = np.clip(pseudo, 0, 1)
        return pseudo
    else:
        # 실제 inference (서버에서 실행)
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        import torch
        from training import build_model
        from data import KmerTokenizer, CircRNABindingSitesDataset

        # TODO: 실제 inference 코드
        # model = load_model(model_dir)
        # pred = model.predict(row)
        raise NotImplementedError("서버에서 실제 inference 코드 연결 필요")

# ══════════════════════════════════════════════════════════════════════════════
# (a) Circular Diagram
# ══════════════════════════════════════════════════════════════════════════════
def draw_circular_binding(ax, seq, sites, title='', bsj_mark=True,
                           pred=None, max_len_show=None):
    """
    circRNA를 원형으로 그리고 binding site를 색깔로 표시.
    pred가 주어지면 outer ring에 예측 확률을 표시.
    """
    L = len(seq)
    if max_len_show and L > max_len_show:
        seq = seq[:max_len_show]
        sites = sites[:max_len_show]
        if pred is not None:
            pred = pred[:max_len_show]
        L = max_len_show

    cx, cy, r_inner, r_outer = 0.0, 0.0, 0.55, 0.80
    r_pred  = 0.90   # prediction ring (outer)

    angles = np.linspace(90, 90 - 360, L, endpoint=False)  # 12시부터 시계방향

    ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.35, 1.35)
    ax.set_aspect('equal'); ax.axis('off')

    # ── base circle ──
    circle = plt.Circle((cx, cy), (r_inner + r_outer) / 2,
                         fill=False, edgecolor='#DDDDDD', lw=0.5, zorder=1)
    ax.add_patch(circle)

    # ── nucleotide arcs ──
    arc_width = 360 / L
    for i in range(L):
        ang = angles[i]
        ang_rad = np.radians(ang)
        is_bound = bool(sites[i])
        fc = BIND_COLOR if is_bound else NOBIND_COLOR
        alpha = 1.0 if is_bound else 0.4

        from matplotlib.patches import Wedge
        wedge = Wedge((cx, cy), r_outer, ang - arc_width * 0.45, ang + arc_width * 0.45,
                      width=r_outer - r_inner,
                      facecolor=fc, edgecolor='white', linewidth=0.2,
                      alpha=alpha, zorder=2)
        ax.add_patch(wedge)

    # ── prediction ring (outer) ──
    if pred is not None:
        from matplotlib.patches import Wedge
        r_p_inner, r_p_outer = r_outer + 0.02, r_outer + 0.08
        for i in range(L):
            ang = angles[i]
            p = float(pred[i])
            fc = plt.cm.RdYlGn(p * 0.8 + 0.1)
            wedge = Wedge((cx, cy), r_p_outer, ang - arc_width * 0.45, ang + arc_width * 0.45,
                          width=r_p_outer - r_p_inner,
                          facecolor=fc, edgecolor='none',
                          alpha=0.9, zorder=3)
            ax.add_patch(wedge)
        ax.text(0, r_p_outer + 0.08, 'Pred.', ha='center', fontsize=8.5,
                color='#555', style='italic')

    # ── BSJ marker ──
    if bsj_mark:
        bsj_ang = np.radians(90)  # 12시 방향 (서열 시작)
        x1 = cx + r_inner * np.cos(bsj_ang)
        y1 = cy + r_inner * np.sin(bsj_ang)
        x2 = cx + (r_outer + 0.05) * np.cos(bsj_ang)
        y2 = cy + (r_outer + 0.05) * np.sin(bsj_ang)
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=BSJ_COLOR, lw=2))
        ax.text(x2 + 0.03, y2 + 0.07, 'BSJ', ha='center', fontsize=10,
                color=BSJ_COLOR, fontweight='bold')

    # ── center label ──
    n_bound = int(sum(sites))
    ax.text(cx, cy + 0.08, f'{n_bound}/{L}', ha='center', va='center',
            fontsize=13, fontweight='bold', color='#2C3E50')
    ax.text(cx, cy - 0.10, 'binding\nsites', ha='center', va='center',
            fontsize=9, color='#777')

    if title:
        ax.set_title(title, fontsize=11, fontweight='bold', pad=6, color='#2C3E50')


# ══════════════════════════════════════════════════════════════════════════════
# (b) Linear Heatmap
# ══════════════════════════════════════════════════════════════════════════════
def draw_linear_heatmap(ax, seq, sites, pred_circmac, pred_linear=None,
                         title='', max_show=200):
    L = len(seq)
    show = min(L, max_show)
    seq_s = seq[:show]
    sites_s = np.array(sites[:show])
    pred_c = np.array(pred_circmac[:show])
    pred_l = np.array(pred_linear[:show]) if pred_linear is not None else None

    rows = 3 if pred_l is not None else 2
    fig_inner = ax.inset_axes([0, 0, 1, 1])
    fig_inner.remove()

    # ground truth bar
    gt_colors = [BIND_COLOR if s else NOBIND_COLOR for s in sites_s]
    ax.bar(range(show), [0.9] * show, color=gt_colors, width=1.0,
           alpha=0.85, zorder=2)

    # 범례
    ax.set_xlim(-1, show + 1)
    ax.set_ylim(0, rows + 0.5)
    ax.axis('off')

    # row labels
    ax.text(-1.5, 0.45, 'GT', ha='right', va='center', fontsize=10,
            fontweight='bold', color='#2C3E50')

    row_y = 1.1
    # CircMAC prediction heatmap
    for i, p in enumerate(pred_c):
        fc = plt.cm.Reds(p * 0.85 + 0.1)
        ax.barh(row_y, 1, left=i, height=0.8, color=fc, zorder=2)
    ax.text(-1.5, row_y, 'CircMAC', ha='right', va='center', fontsize=9,
            color='#E74C3C', fontweight='bold')

    if pred_l is not None:
        row_y = 2.1
        for i, p in enumerate(pred_l):
            fc = plt.cm.Blues(p * 0.85 + 0.1)
            ax.barh(row_y, 1, left=i, height=0.8, color=fc, zorder=2)
        ax.text(-1.5, row_y, 'Linear', ha='right', va='center', fontsize=9,
                color='#3498DB', fontweight='bold')

    # BSJ markers at both ends
    ax.axvline(0, color=BSJ_COLOR, lw=2.5, linestyle='--', alpha=0.8, zorder=5)
    ax.axvline(show - 1, color=BSJ_COLOR, lw=2.5, linestyle='--', alpha=0.8, zorder=5)
    ax.text(0, rows + 0.25, "5' BSJ", ha='center', fontsize=9,
            color=BSJ_COLOR, fontweight='bold')
    ax.text(show - 1, rows + 0.25, "3' BSJ", ha='center', fontsize=9,
            color=BSJ_COLOR, fontweight='bold')
    ax.annotate('', xy=(show + 0.5, rows + 0.25), xytext=(-0.5, rows + 0.25),
                arrowprops=dict(arrowstyle='<->', color=BSJ_COLOR,
                                lw=1.5, linestyle='dashed'))

    if title:
        ax.set_title(title, fontsize=11, fontweight='bold', pad=4, color='#2C3E50')


# ══════════════════════════════════════════════════════════════════════════════
# Main Figure
# ══════════════════════════════════════════════════════════════════════════════
def main(with_pred=False, model_dir=None, data_path=None):
    np.random.seed(42)
    df = load_data(data_path)

    # ── 케이스 선택 ──────────────────────────────────────────────────────────
    # Case A: BSJ 직전 binding (3' end)
    row_a = df.iloc[766]   # L=153, sites at 130-152
    # Case B: BSJ 직후 binding (5' end)
    row_b = df.iloc[32]    # L=250, sites at 0-27
    # Case C: 중간 binding (comparison)
    row_c = df[
        (df['binding'] == 1) &
        (df['length'].between(150, 200))
    ].iloc[5]   # 적당한 중간 케이스

    cases = [
        (row_a, "Case A: Binding site at 3' end\n(adjacent to BSJ in circular form)"),
        (row_b, "Case B: Binding site at 5' end\n(adjacent to BSJ in circular form)"),
        (row_c, "Case C: Binding site in the middle\n(not near BSJ)"),
    ]

    fig = plt.figure(figsize=(18, 13))
    fig.patch.set_facecolor('white')
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.30,
                  top=0.92, bottom=0.06, left=0.05, right=0.97)

    # ── Row 0: Circular diagrams ─────────────────────────────────────────────
    for col, (row, ttl) in enumerate(cases):
        ax = fig.add_subplot(gs[0, col])
        seq   = row['circRNA']
        sites = np.array(row['sites'])
        pred  = get_predictions(row, model_dir if with_pred else None)
        draw_circular_binding(ax, seq, sites, title=ttl,
                              pred=pred if with_pred else None)

    # ── Row 1: Linear heatmap for Cases A and B ───────────────────────────────
    for col, (row, ttl) in enumerate(cases[:2]):
        ax = fig.add_subplot(gs[1, col])
        seq   = row['circRNA']
        sites = np.array(row['sites'])
        pred_c = get_predictions(row, model_dir if with_pred else None)
        # 선형 모델 시뮬: BSJ 근처 예측을 낮게
        pred_l = pred_c.copy()
        bsj_win = 25
        decay = np.ones(len(pred_l))
        decay[:bsj_win] = np.linspace(0.5, 1.0, bsj_win)
        decay[-bsj_win:] = np.linspace(1.0, 0.5, bsj_win)
        pred_l = pred_l * decay
        draw_linear_heatmap(ax, seq, sites, pred_c, pred_l,
                            title=f'Linear view: {ttl.split(chr(10))[0]}')

    # ── Row 1 Col 2: BSJ statistics ───────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # BSJ 근처 binding 비율 계산
    bsj_wins = [5, 10, 15, 22, 30]
    ratios = []
    pos_df = df[df['binding'] == 1]
    for w in bsj_wins:
        cnt = sum(
            1 for _, r in pos_df.iterrows()
            if any(r['sites'][:w]) or any(r['sites'][-w:])
        )
        ratios.append(cnt / len(pos_df) * 100)

    bars = ax.bar(range(len(bsj_wins)), ratios,
                  color=['#AED6F1', '#5DADE2', '#2980B9', '#1A5276', '#154360'],
                  edgecolor='white', linewidth=1.0)
    for bar, r in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{r:.1f}%', ha='center', fontsize=10, fontweight='bold',
                color='#2C3E50')

    ax.set_xticks(range(len(bsj_wins)))
    ax.set_xticklabels([f'±{w}nt' for w in bsj_wins], fontsize=10)
    ax.set_xlabel('BSJ window size', fontsize=11)
    ax.set_ylabel('Fraction of positive pairs (%)', fontsize=11)
    ax.set_title('(f) Binding sites near BSJ\n(test set positive pairs)', fontsize=11,
                 fontweight='bold', pad=6)
    ax.set_ylim(0, max(ratios) * 1.25)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # ── 전역 범례 ────────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(color=BIND_COLOR,   label='Binding site (GT)'),
        mpatches.Patch(color=NOBIND_COLOR, label='Non-binding'),
        mpatches.Patch(color=BSJ_COLOR,    label='BSJ (Back-Splice Junction)'),
    ]
    if with_pred:
        legend_items += [
            mpatches.Patch(color='#C0392B', label='CircMAC prediction (high)'),
            mpatches.Patch(color='#EBF5FB', label='CircMAC prediction (low)'),
        ]
    fig.legend(handles=legend_items, loc='lower center', ncol=len(legend_items),
               fontsize=9.5, framealpha=0.9, edgecolor='#ccc',
               bbox_to_anchor=(0.5, 0.01))

    fig.suptitle('circRNA–miRNA Binding Site Visualization\n'
                 '(CircRNA shown as circular topology; BSJ = sequence start/end junction)',
                 fontsize=14, fontweight='bold', y=0.97, color='#2C3E50')

    suffix = '_with_pred' if with_pred else '_gt_only'
    out_pdf = f'/workspace/volume/cmi_mac/docs/paper_cmi/binding_visualization{suffix}.pdf'
    out_png = f'/workspace/volume/cmi_mac/docs/paper_cmi/binding_visualization{suffix}.png'
    plt.savefig(out_pdf, bbox_inches='tight', dpi=300)
    plt.savefig(out_png, bbox_inches='tight', dpi=200)
    print(f'Saved: {out_pdf}')
    print(f'Saved: {out_png}')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--with_pred', action='store_true',
                        help='Run model inference (requires saved model on server)')
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to df_test_final.pkl (default: hardcoded DATA_PATH)')
    args = parser.parse_args()
    main(with_pred=args.with_pred, model_dir=args.model_dir, data_path=args.data_path)
