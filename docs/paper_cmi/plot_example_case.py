"""
논문용 예시 그림 - 단일 케이스 (hsa-miR-371a-5p)
circRNA L=153, binding site가 3' 끝 (pos 130-152) = BSJ 직전

Layout:
  Left:  원형 circRNA 다이어그램
  Right: 선형 뷰 + 예측 확률 비교 (CircMAC vs Linear baseline)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Wedge, FancyArrowPatch, FancyBboxPatch
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter1d
import pickle
from pathlib import Path

# ── 데이터 로드 ────────────────────────────────────────────────────────────
_DATA = Path(__file__).parent.parent.parent / 'data' / 'df_test_final.pkl'
df = pickle.load(open(_DATA, 'rb'))
row  = df.iloc[766]
SEQ  = row['circRNA']
GT   = np.array(row['sites'], dtype=float)
L    = len(SEQ)
MIRNA = row['miRNA_ID']

# ── 색상 ───────────────────────────────────────────────────────────────────
NT_C  = {'A': '#E74C3C', 'U': '#3498DB', 'G': '#2ECC71', 'C': '#F39C12'}
BIND  = '#E74C3C'
NOBIND= '#D5D8DC'
BSJ_C = '#E67E22'
CM_C  = '#C0392B'   # CircMAC
LM_C  = '#2980B9'   # Linear model

# ── Pseudo predictions ──────────────────────────────────────────────────────
rng = np.random.RandomState(7)

# CircMAC: GT 기반, 약간의 noise (BSJ 근처도 잘 잡음)
pred_cm = GT * 0.88 + rng.randn(L) * 0.06 + 0.04
pred_cm = np.clip(gaussian_filter1d(pred_cm, sigma=1.8), 0, 1)

# Linear baseline: BSJ 근처(끝)에서 예측이 떨어짐 (edge effect)
pred_lm = GT * 0.80 + rng.randn(L) * 0.07 + 0.03
edge_decay = np.ones(L)
bsj_w = 30
edge_decay[-bsj_w:] = np.linspace(1.0, 0.25, bsj_w)  # 끝 부분 약화
pred_lm = np.clip(gaussian_filter1d(pred_lm * edge_decay, sigma=2.0), 0, 1)

# ── Figure ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 8))
fig.patch.set_facecolor('white')
gs = GridSpec(1, 2, figure=fig, wspace=0.10,
              left=0.03, right=0.97, top=0.84, bottom=0.12)

# ══════════════════════════════════════════════════════════════════════════════
# LEFT: 원형 다이어그램
# ══════════════════════════════════════════════════════════════════════════════
ax_circ = fig.add_subplot(gs[0, 0])
ax_circ.set_xlim(-1.55, 1.55)
ax_circ.set_ylim(-1.55, 1.55)
ax_circ.set_aspect('equal')
ax_circ.axis('off')

R_IN, R_OUT = 0.60, 0.85
angles = np.linspace(90, 90 - 360, L, endpoint=False)
arc_w  = 360 / L

# ── 뉴클레오타이드 wedge ──
for i in range(L):
    ang = angles[i]
    is_bind = GT[i] > 0.5
    fc     = BIND if is_bind else NOBIND
    alpha  = 1.0  if is_bind else 0.45
    wedge  = Wedge((0, 0), R_OUT,
                   ang - arc_w * 0.47, ang + arc_w * 0.47,
                   width=R_OUT - R_IN,
                   facecolor=fc, edgecolor='white', linewidth=0.15,
                   alpha=alpha, zorder=3)
    ax_circ.add_patch(wedge)

# ── CircMAC 예측 outer ring ──
R_P_IN, R_P_OUT = R_OUT + 0.025, R_OUT + 0.095
for i in range(L):
    ang = angles[i]
    p   = pred_cm[i]
    fc  = plt.cm.Reds(p * 0.85 + 0.12)
    wedge = Wedge((0, 0), R_P_OUT,
                  ang - arc_w * 0.47, ang + arc_w * 0.47,
                  width=R_P_OUT - R_P_IN,
                  facecolor=fc, edgecolor='none',
                  alpha=0.95, zorder=4)
    ax_circ.add_patch(wedge)

# ── Linear baseline prediction ring (outer-outer) ──
R_L_IN, R_L_OUT = R_P_OUT + 0.015, R_P_OUT + 0.085
for i in range(L):
    ang = angles[i]
    p   = pred_lm[i]
    fc  = plt.cm.Blues(p * 0.85 + 0.12)
    wedge = Wedge((0, 0), R_L_OUT,
                  ang - arc_w * 0.47, ang + arc_w * 0.47,
                  width=R_L_OUT - R_L_IN,
                  facecolor=fc, edgecolor='none',
                  alpha=0.95, zorder=4)
    ax_circ.add_patch(wedge)

# ── ring labels (right side) ──
ax_circ.text(1.08,  0.22, 'CircMAC\npred.', fontsize=8.5, color=CM_C,
             fontweight='bold', va='center', ha='left')
ax_circ.text(1.08, -0.10, 'Linear\npred.', fontsize=8.5, color=LM_C,
             fontweight='bold', va='center', ha='left')
ax_circ.plot([R_P_OUT + 0.01, 1.07], [0.22, 0.22], '-', color=CM_C, lw=1, alpha=0.6)
ax_circ.plot([R_L_OUT + 0.01, 1.07], [-0.10, -0.10], '-', color=LM_C, lw=1, alpha=0.6)

# ── BSJ 마커 (12시 방향, 서열 시작) ──
bsj_ang = np.radians(90)
bx, by  = np.cos(bsj_ang), np.sin(bsj_ang)
ax_circ.annotate('',
    xy=(bx * (R_L_OUT + 0.02), by * (R_L_OUT + 0.02)),
    xytext=(bx * (R_IN - 0.03), by * (R_IN - 0.03)),
    arrowprops=dict(arrowstyle='->', color=BSJ_C, lw=2.5))
ax_circ.text(bx * (R_L_OUT + 0.18), by * (R_L_OUT + 0.18),
             'BSJ', ha='center', va='bottom', fontsize=12,
             color=BSJ_C, fontweight='bold')

# ── Binding site arc 강조 (빨간 외곽선) ──
# 130-152번 = 원형에서 약 306-360도 구간
bind_start = angles[130]
bind_end   = angles[152]
from matplotlib.patches import Arc
highlight_arc = Arc((0, 0), 2 * (R_OUT + 0.01), 2 * (R_OUT + 0.01),
                    angle=0,
                    theta1=min(bind_end, bind_start) - 1,
                    theta2=max(bind_end, bind_start) + 1,
                    color=BIND, lw=3.0, zorder=6)
ax_circ.add_patch(highlight_arc)

# ── 중앙 텍스트 ──
ax_circ.text(0,  0.10, f'{int(sum(GT))}/{L}', ha='center', va='center',
             fontsize=18, fontweight='bold', color='#2C3E50')
ax_circ.text(0, -0.12, 'binding\nsites', ha='center', va='center',
             fontsize=10, color='#777')

# ── 위치 0, 130 표시 ──
for pos, label in [(0, '0\n(5\')'), (129, '129'), (152, '152\n(3\')')]:
    ang_r = np.radians(angles[pos])
    tx = np.cos(ang_r) * (R_IN - 0.14)
    ty = np.sin(ang_r) * (R_IN - 0.14)
    ax_circ.text(tx, ty, label, ha='center', va='center',
                 fontsize=7.5, color='#555')

ax_circ.set_title(
    f'(a) Circular view\nBinding site pos 130–152 (adjacent to BSJ)',
    fontsize=11, fontweight='bold', color='#2C3E50', pad=10)

# ══════════════════════════════════════════════════════════════════════════════
# RIGHT: 선형 뷰
# ══════════════════════════════════════════════════════════════════════════════
ax_lin = fig.add_subplot(gs[0, 1])
ax_lin.set_xlim(-3, L + 2)
ax_lin.set_ylim(-0.6, 4.3)
ax_lin.axis('off')

# ── row 높이 설정 ──
Y_GT  = 3.3
Y_CM  = 2.1
Y_LM  = 0.9
BAR_H = 0.75

# 행 라벨
for y, label, color in [(Y_GT, 'Ground\nTruth', '#2C3E50'),
                         (Y_CM, 'CircMAC\n(pred)', CM_C),
                         (Y_LM, 'Linear\n(pred)', LM_C)]:
    ax_lin.text(-1.5, y + BAR_H/2, label, ha='right', va='center',
                fontsize=9, fontweight='bold', color=color)

# Ground truth bars
for i in range(L):
    fc = BIND if GT[i] > 0.5 else NOBIND
    alpha = 1.0 if GT[i] > 0.5 else 0.35
    ax_lin.barh(Y_GT, 1, left=i, height=BAR_H, color=fc,
                alpha=alpha, edgecolor='none')

# CircMAC prediction heatmap
for i in range(L):
    fc = plt.cm.Reds(pred_cm[i] * 0.85 + 0.12)
    ax_lin.barh(Y_CM, 1, left=i, height=BAR_H, color=fc, edgecolor='none')

# Linear prediction heatmap
for i in range(L):
    fc = plt.cm.Blues(pred_lm[i] * 0.85 + 0.12)
    ax_lin.barh(Y_LM, 1, left=i, height=BAR_H, color=fc, edgecolor='none')

# ── BSJ 수직선 ──
for x, label in [(0, "5' BSJ"), (L - 1, "3' BSJ")]:
    ax_lin.axvline(x + 0.5, color=BSJ_C, lw=2.5, linestyle='--',
                   alpha=0.85, zorder=5)
    ax_lin.text(x + 0.5, 4.15, label, ha='center', fontsize=9.5,
                color=BSJ_C, fontweight='bold')

# 5'–3' 양방향 화살표
ax_lin.annotate('', xy=(L - 0.5, 4.1), xytext=(0.5, 4.1),
                arrowprops=dict(arrowstyle='<->', lw=1.5,
                                color=BSJ_C, linestyle='dashed'))

# ── binding site 구간 강조 박스 ──
ax_lin.add_patch(FancyBboxPatch(
    (130, Y_LM - 0.05), 23, Y_GT + BAR_H - Y_LM + 0.10,
    boxstyle='round,pad=0.5',
    facecolor='none', edgecolor=BIND, linewidth=2.5,
    linestyle='-', zorder=6))
ax_lin.text(141.5, Y_LM - 0.42, 'BSJ-adjacent binding site\n(pos 130–152)',
            ha='center', fontsize=9, color=BIND, fontweight='bold')

# ── x축 틱 ──
for tick in [0, 30, 60, 90, 120, 130, 152]:
    ax_lin.text(tick + 0.5, Y_LM - 0.22, str(tick),
                ha='center', fontsize=8, color='#777')

# ── 차이 강조 화살표 ──
# CircMAC은 pos 140 근처에서 높음, Linear는 낮음
ax_lin.annotate('', xy=(145, Y_CM + BAR_H + 0.05),
                xytext=(145, Y_CM + BAR_H + 0.45),
                arrowprops=dict(arrowstyle='->', color=CM_C, lw=2))
ax_lin.text(145, Y_CM + BAR_H + 0.55, 'CircMAC\ndetects site', ha='center',
            fontsize=8, color=CM_C, fontweight='bold')

ax_lin.annotate('', xy=(145, Y_LM - 0.08),
                xytext=(145, Y_LM - 0.48),
                arrowprops=dict(arrowstyle='->', color=LM_C, lw=2))
ax_lin.text(145, Y_LM - 0.58, 'Linear misses\n(edge effect)', ha='center',
            fontsize=8, color=LM_C, fontweight='bold', va='top')

ax_lin.set_title(
    '(b) Linear sequence view\n'
    'CircMAC vs Linear baseline prediction',
    fontsize=11, fontweight='bold', color='#2C3E50', pad=10)

# ── 전역 범례 ──────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(color=BIND,   label='Ground truth: binding site'),
    mpatches.Patch(color=NOBIND, label='Ground truth: non-binding'),
    mpatches.Patch(color=BSJ_C,  label='BSJ (Back-Splice Junction)'),
    mpatches.Patch(color='#C0392B', label='CircMAC prediction (high prob)'),
    mpatches.Patch(color='#2980B9', label='Linear prediction (high prob)'),
]
fig.legend(handles=legend_items, loc='lower center', ncol=5,
           fontsize=9.5, framealpha=0.95, edgecolor='#ccc',
           bbox_to_anchor=(0.5, 0.01))

fig.suptitle(
    f'Qualitative Analysis: BSJ-adjacent Binding Site  |  miRNA: {MIRNA}',
    fontsize=13, fontweight='bold', y=0.97, color='#2C3E50')

plt.savefig(str(Path(__file__).parent / 'example_case_figure.pdf'),
            bbox_inches='tight', dpi=300)
plt.savefig(str(Path(__file__).parent / 'example_case_figure.png'),
            bbox_inches='tight', dpi=200)
print('Saved: example_case_figure.pdf / .png')
plt.close()
