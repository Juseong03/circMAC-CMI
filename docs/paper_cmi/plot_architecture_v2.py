"""
CircMAC Architecture Figure v2 — 논문용
3-panel layout:
  (a) 전체 파이프라인 (좌)
  (b) CircMACBlock 상세 (우 상단)
  (c) Circular 특성 설명 (우 하단)
"""

import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Arc
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

C = dict(
    attn='#7D3C98', mamba='#1A5276', cnn='#1E8449',
    router='#D35400', head='#922B21', embed='#5D6D7E',
    cross='#0E6655', dark='#1C2833', gold='#D4AC0D',
    light='#F2F3F4', mid='#D5D8DC', white='white',
    skip='#AAB7B8', bg='white',
)

def box(ax, x, y, w, h, fc, ec='white', lw=1.5, r=0.25, alpha=1.0, zo=3):
    p = FancyBboxPatch((x-w/2, y-h/2), w, h,
                       boxstyle=f'round,pad={r}',
                       facecolor=fc, edgecolor=ec, linewidth=lw,
                       alpha=alpha, zorder=zo)
    ax.add_patch(p)

def arr(ax, x0, y0, x1, y1, c='#555', lw=1.8, zo=5, rad=0, style='->'):
    ax.annotate('', xy=(x1,y1), xytext=(x0,y0),
                arrowprops=dict(arrowstyle=style, color=c, lw=lw,
                                connectionstyle=f'arc3,rad={rad}'), zorder=zo)

def t(ax, x, y, s, sz=10, c='white', bold=True, ha='center', va='center', zo=6, italic=False):
    ax.text(x, y, s, fontsize=sz, color=c,
            fontweight='bold' if bold else 'normal',
            fontstyle='italic' if italic else 'normal',
            ha=ha, va=va, zorder=zo)

# ── Figure 설정 ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 12))
fig.patch.set_facecolor(C['bg'])
gs_main = GridSpec(1, 2, figure=fig,
                   left=0.02, right=0.98, top=0.93, bottom=0.06,
                   wspace=0.06, width_ratios=[1, 1.15])
gs_right = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_main[0,1],
                                   hspace=0.10, height_ratios=[1.1, 0.9])

# ══════════════════════════════════════════════════════════════════════════════
# (a) 전체 파이프라인
# ══════════════════════════════════════════════════════════════════════════════
ax = fig.add_subplot(gs_main[0, 0])
ax.set_xlim(0, 10); ax.set_ylim(0, 22.5)
ax.axis('off')

t(ax, 5, 22.0, '(a)  CircMAC Pipeline', sz=13, c=C['dark'])

# ─ 입력 ─────────────────────────────────────────────────────────────────────
for x, w, label in [(3.2, 4.5, 'circRNA sequence'), (7.8, 2.6, 'miRNA sequence')]:
    box(ax, x, 21.1, w, 0.80, C['embed'], ec=C['mid'], lw=1.2)
    t(ax, x, 21.1, label, sz=10, c=C['white'])

# ─ 임베딩 ───────────────────────────────────────────────────────────────────
for x, w, label in [(3.2, 4.5, 'K-mer Tokenizer  +  Embedding'), (7.8, 2.6, 'Embedding')]:
    arr(ax, x, 20.70, x, 20.10)
    box(ax, x, 19.75, w, 0.65, C['embed'])
    t(ax, x, 19.75, label, sz=9.5)

# ─ CircMAC Encoder ───────────────────────────────────────────────────────────
arr(ax, 3.2, 19.42, 3.2, 18.82)
box(ax, 3.2, 18.50, 4.8, 0.60, C['mid'], ec='#AAA', lw=1)
t(ax, 3.2, 18.50, 'Downsample ×½   (Conv1D stride=2)', sz=9, c=C['dark'])

arr(ax, 3.2, 18.20, 3.2, 17.65)

# N × CircMACBlock
bx, N = 3.2, 3
by_top = 17.35
bh, gap = 0.70, 0.20
for i in range(N):
    yc = by_top - i*(bh+gap)
    alpha = 1.0 - i*0.22
    box(ax, bx, yc, 5.2, bh, C['dark'], alpha=alpha)
    if i == 0:
        t(ax, bx, yc, 'CircMACBlock   ×   N', sz=10.5)
    else:
        t(ax, bx, yc, '· · ·', sz=12)
    if i < N-1:
        arr(ax, bx, yc-bh/2, bx, yc-bh/2-gap)

# bracket
brk_y0 = by_top - (N-1)*(bh+gap) - bh/2
brk_y1 = by_top + bh/2
ax.plot([0.15, 0.50, 0.50], [brk_y0, brk_y0, brk_y1], color='#999', lw=1.5)
ax.plot([0.15, 0.50, 0.50], [brk_y1, brk_y1, brk_y0], color='#999', lw=1.5)
t(ax, 0.08, (brk_y0+brk_y1)/2, f'N\nlayers', sz=8, c='#888', bold=False)

skip_y_top = 19.42
skip_y_bot = brk_y0 - 0.05
arr(ax, 3.2, brk_y0, 3.2, brk_y0 - 0.60)

box(ax, 3.2, brk_y0 - 0.93, 5.2, 0.60, C['mid'], ec='#AAA', lw=1)
t(ax, 3.2, brk_y0 - 0.93, 'Upsample ×2   +   Skip connection', sz=9, c=C['dark'])

arr(ax, 3.2, brk_y0 - 1.23, 3.2, brk_y0 - 1.78)

e_circ_y = brk_y0 - 2.08
box(ax, 3.2, e_circ_y, 5.0, 0.68, C['dark'])
t(ax, 3.2, e_circ_y + 0.13, 'E_circ', sz=11)
t(ax, 3.2, e_circ_y - 0.17, '[B, L, D]', sz=9, bold=False)

# skip 점선
ax.annotate('', xy=(0.7, brk_y0 - 1.08),
            xytext=(0.7, skip_y_top),
            arrowprops=dict(arrowstyle='->', color=C['skip'], lw=1.5,
                            linestyle='dashed', connectionstyle='arc3,rad=0'))
t(ax, 0.38, (skip_y_top + brk_y0 - 1.08)/2, 'skip', sz=8, c=C['skip'],
  bold=False, italic=True)

# miRNA encoder arrow (긴 수직선)
arr(ax, 7.8, 19.42, 7.8, e_circ_y + 0.35)
box(ax, 7.8, e_circ_y, 2.6, 0.68, C['embed'])
t(ax, 7.8, e_circ_y + 0.13, 'E_mirna', sz=11)
t(ax, 7.8, e_circ_y - 0.17, '[B, M, D]', sz=9, bold=False)

# ─ Cross-Attention ───────────────────────────────────────────────────────────
arr(ax, 3.2, e_circ_y - 0.34, 3.2, e_circ_y - 0.92)
arr(ax, 7.8, e_circ_y - 0.34, 5.5,  e_circ_y - 0.88, c=C['cross'], rad=-0.15)

ca_y = e_circ_y - 1.22
box(ax, 3.2, ca_y, 6.0, 0.68, C['cross'])
t(ax, 3.2, ca_y + 0.14, 'Cross-Attention', sz=10.5)
t(ax, 3.2, ca_y - 0.16, 'Q: E_circ  ←  K,V: E_mirna', sz=8.5, bold=False)

arr(ax, 3.2, ca_y - 0.34, 3.2, ca_y - 0.92)

# ─ Feature Enhancer + Site Head ─────────────────────────────────────────────
fe_y = ca_y - 1.22
box(ax, 3.2, fe_y, 5.6, 0.68, C['router'])
t(ax, 3.2, fe_y + 0.14, 'MultiKernelCNN  Feature Enhancer', sz=9.5)
t(ax, 3.2, fe_y - 0.16, 'k = 3, 5, 7  parallel  +  Residual + Norm', sz=8.5, bold=False)

arr(ax, 3.2, fe_y - 0.34, 3.2, fe_y - 0.92)

sh_y = fe_y - 1.22
box(ax, 3.2, sh_y, 5.6, 0.68, C['head'])
t(ax, 3.2, sh_y + 0.14, 'Conv1D Site Head', sz=10.5)
t(ax, 3.2, sh_y - 0.16, 'Conv1D(D→D/2, k=3) → Conv1D(D/2→2, k=1)', sz=8.5, bold=False)

arr(ax, 3.2, sh_y - 0.34, 3.2, sh_y - 0.88)

pred_y = sh_y - 1.15
box(ax, 3.2, pred_y, 5.0, 0.62, '#7B241C')
t(ax, 3.2, pred_y, 'Binding Site Prediction  [B, L, 2]', sz=10)

arr(ax, 3.2, pred_y - 0.31, 3.2, pred_y - 0.82)

loss_y = pred_y - 1.08
box(ax, 3.2, loss_y, 4.4, 0.55, '#922B21')
t(ax, 3.2, loss_y, 'CrossEntropy Loss  (site-level)', sz=9.5)

# ─ Binding head (side) ───────────────────────────────────────────────────────
arr(ax, 5.9, ca_y - 0.34, 7.5, fe_y + 0.35, c=C['mid'], lw=1.2, rad=-0.2)
box(ax, 8.3, fe_y, 2.6, 0.68, C['embed'])
t(ax, 8.3, fe_y + 0.14, 'Binding Head', sz=9)
t(ax, 8.3, fe_y - 0.16, '(binary cls.)', sz=8.5, bold=False)

# ─ Config table ──────────────────────────────────────────────────────────────
cfg_y = loss_y - 0.55
box(ax, 7.5, cfg_y, 4.2, 3.0, C['light'], ec=C['mid'], lw=1.2, r=0.2)
t(ax, 7.5, cfg_y + 1.2, 'Default Config', sz=10, c=C['dark'])
for i,(k,v) in enumerate([('d_model','128'),('n_layer','6'),
                           ('n_heads','8'),('max_len','1022'),
                           ('kernel_size','7'),('d_state','16')]):
    yy = cfg_y + 0.78 - i*0.37
    t(ax, 5.8, yy, k, sz=9, c='#555', bold=False, ha='left')
    t(ax, 9.2, yy, v, sz=9, c=C['dark'], ha='right')

# ══════════════════════════════════════════════════════════════════════════════
# (b) CircMACBlock 내부
# ══════════════════════════════════════════════════════════════════════════════
ax2 = fig.add_subplot(gs_right[0])
ax2.set_xlim(0, 10); ax2.set_ylim(0, 11.5)
ax2.axis('off')
t(ax2, 5, 11.1, '(b)  CircMACBlock  (single layer)', sz=13, c=C['dark'])

# Input
box(ax2, 5, 10.5, 6.2, 0.72, C['dark'])
t(ax2, 5, 10.5, 'Input   x   [B, L, D]', sz=10.5)

arr(ax2, 5, 10.14, 5, 9.62)

# in_proj
box(ax2, 5, 9.3, 6.2, 0.60, '#566573')
t(ax2, 5, 9.3, 'in_proj  (Linear: D → 4D)', sz=10)

arr(ax2, 5, 9.00, 5, 8.52)
t(ax2, 5, 8.42, 'Q, K, V  [D each]                  base  [D]',
  sz=8.5, c='#555', bold=False)

# 분기 선
ax2.plot([1.5, 8.5], [8.2, 8.2], color=C['mid'], lw=1.3)
for x in [2.0, 5.0, 8.0]:
    ax2.plot([x, x], [8.2, 7.65], color=C['mid'], lw=1.3)

# 세 브랜치
for x, fc, title, sub in [
    (2.0, C['attn'], 'Multi-Head', 'Self-Attention'),
    (5.0, C['mamba'], 'Mamba SSM', '(causal / sequential)'),
    (8.0, C['cnn'],  'Depthwise CNN', '(k=7, circular pad)'),
]:
    box(ax2, x, 7.25, 3.2, 0.72, fc)
    t(ax2, x, 7.40, title, sz=9.5)
    t(ax2, x, 7.08, sub, sz=8.5, bold=False)

# circular bias tag
box(ax2, 2.0, 6.35, 3.0, 0.52, '#A569BD', r=0.15)
t(ax2, 2.0, 6.35, '+ Circular Rel. Bias', sz=8.5)
arr(ax2, 2.0, 6.89, 2.0, 6.62)

# RMSNorm
for x, fc in [(2.0, C['attn']), (5.0, C['mamba']), (8.0, C['cnn'])]:
    yy = 5.62 if x == 2.0 else 5.98
    arr(ax2, x, (6.62 if x==2.0 else 6.89), x, yy)
    box(ax2, x, yy - 0.28, 2.8, 0.50, fc, alpha=0.55)
    t(ax2, x, yy - 0.28, 'RMSNorm', sz=9)

# 합류 선
for x, yb in [(2.0, 5.34), (5.0, 5.70), (8.0, 5.70)]:
    arr(ax2, x, yb, x, 4.90)
ax2.plot([1.5, 8.5], [4.87, 4.87], color=C['mid'], lw=1.3)
arr(ax2, 5, 4.87, 5, 4.38)

# Router
box(ax2, 5, 4.05, 7.5, 0.62, C['router'])
t(ax2, 5, 4.22, 'Adaptive Router', sz=10.5)
t(ax2, 5, 3.88, 'gates = Softmax(Linear([mean(attn), mean(mamba), mean(cnn)]))', sz=8)

arr(ax2, 5, 3.74, 5, 3.22)

# out_proj
box(ax2, 5, 2.92, 5.5, 0.55, '#566573')
t(ax2, 5, 2.92, 'out_proj  (Linear: D → D)', sz=10)

arr(ax2, 5, 2.64, 5, 2.12)

# Residual output
box(ax2, 5, 1.82, 6.2, 0.55, C['dark'])
t(ax2, 5, 1.82, 'x   +   out_proj( · )   [Residual]', sz=10)

# residual 점선
ax2.annotate('', xy=(9.2, 1.82), xytext=(9.2, 10.5),
             arrowprops=dict(arrowstyle='->', color=C['skip'], lw=1.5,
                             linestyle='dashed'))
t(ax2, 9.55, 6.2, 'residual', sz=8, c=C['skip'], bold=False, italic=True)

# ══════════════════════════════════════════════════════════════════════════════
# (c) CircRNA-specific Components
# ══════════════════════════════════════════════════════════════════════════════
ax3 = fig.add_subplot(gs_right[1])
ax3.set_xlim(0, 10); ax3.set_ylim(0, 8.5)
ax3.axis('off')
t(ax3, 5, 8.15, '(c)  CircRNA-specific Components', sz=13, c=C['dark'])

# ── 왼쪽: Circular Relative Bias ─────────────────────────────────────────────
t(ax3, 2.5, 7.55, 'Circular Relative Position Bias', sz=10, c=C['attn'])

# 원형 배치
L_ex = 8
cx, cy, r_ex = 2.5, 5.55, 1.35
angles_ex = np.linspace(90, 90-360, L_ex, endpoint=False)
for i in range(L_ex):
    a = np.radians(angles_ex[i])
    x = cx + r_ex*np.cos(a); y = cy + r_ex*np.sin(a)
    hi = i in [0, L_ex-1]
    circle = plt.Circle((x, y), 0.28, color=C['attn'] if hi else C['mid'], zorder=4)
    ax3.add_patch(circle)
    t(ax3, x, y, str(i), sz=8.5, c='white' if hi else C['dark'], bold=hi)

# 0-7 호
a0 = np.radians(angles_ex[0]); a7 = np.radians(angles_ex[7])
x0 = cx + r_ex*np.cos(a0); y0 = cy + r_ex*np.sin(a0)
x7 = cx + r_ex*np.cos(a7); y7 = cy + r_ex*np.sin(a7)
ax3.annotate('', xy=(x7, y7+0.28), xytext=(x0, y0+0.28),
             arrowprops=dict(arrowstyle='<->', color=C['attn'], lw=2,
                             connectionstyle='arc3,rad=-0.35'))
t(ax3, cx, cy + r_ex + 0.65, 'd_circ(0, 7) = 1\n(not L−1 = 7)', sz=8.5, c=C['attn'])

# 수식 박스
box(ax3, 2.5, 2.85, 4.6, 1.55, '#F4ECF7', ec='#C39BD3', lw=1.2, r=0.18)
t(ax3, 2.5, 3.45, 'd_linear(i,j)  =  |i − j|', sz=9, c='#922B21', bold=False)
t(ax3, 2.5, 3.00, 'd_circular(i,j)  =  min(|i−j|, L−|i−j|)', sz=9, c=C['attn'])
t(ax3, 2.5, 2.55, 'Bias  =  −slope × d_circular(i, j)', sz=8.5, c='#666', bold=False, italic=True)

# ── 오른쪽: Circular CNN Padding ─────────────────────────────────────────────
t(ax3, 7.5, 7.55, 'Circular Padding  (CNN Branch)', sz=10, c=C['cnn'])

nt_c = {'G':'#1E8449','C':'#D4AC0D','A':'#C0392B','U':'#1A5276'}
seq6 = ['G','C','A','U','G','C']; sc = [nt_c[n] for n in seq6]

def draw_row(ax, y, label, pads, pad_cols, bords, box_idx=None, lbl_color='#555'):
    x0 = 4.85
    gap = 0.56
    ax.text(4.75, y, label, fontsize=8.5, ha='right', va='center', color=lbl_color)
    for i,(nt,fc,bd) in enumerate(zip(pads, pad_cols, bords)):
        x = x0 + i*gap
        c = plt.Circle((x,y), 0.23, color=fc, zorder=4)
        ax.add_patch(c)
        if bd:
            c2 = plt.Circle((x,y), 0.23, fill=False, edgecolor=C['cnn'], lw=2, zorder=5)
            ax.add_patch(c2)
        ax.text(x, y, nt, ha='center', va='center', fontsize=8,
                fontweight='bold', color='white', zorder=6)
    if box_idx is not None:
        xb = x0 + box_idx*gap
        p = FancyBboxPatch((xb-0.23*1.5, y-0.36), 0.56*3, 0.72,
                           boxstyle='round,pad=0.04',
                           facecolor='none', edgecolor=lbl_color, lw=2.2, zorder=7)
        ax.add_patch(p)

# Zero padding row
z_pads   = ['0','0'] + seq6 + ['0','0']
z_cols   = ['#BDC3C7','#BDC3C7'] + sc + ['#BDC3C7','#BDC3C7']
z_bords  = [False]*10
draw_row(ax3, 6.45, 'Zero pad', z_pads, z_cols, z_bords, box_idx=4, lbl_color='#922B21')
t(ax3, 4.85+4*0.56, 5.95, '← kernel', sz=7.5, c='#922B21', bold=False)
t(ax3, 4.85+4*0.56, 5.70, '(sees zeros\nat boundary)', sz=7.5, c='#922B21', bold=False)

# Circular padding row
c_pads  = [seq6[-2], seq6[-1]] + seq6 + [seq6[0], seq6[1]]
c_cols  = [sc[-2], sc[-1]] + sc + [sc[0], sc[1]]
c_bords = [True, True] + [False]*6 + [True, True]
draw_row(ax3, 4.85, 'Circ pad', c_pads, c_cols, c_bords, box_idx=4, lbl_color=C['cnn'])
t(ax3, 4.85+4*0.56, 4.35, '← kernel', sz=7.5, c=C['cnn'], bold=False)
t(ax3, 4.85+4*0.56, 4.10, '(sees real nt\nacross BSJ)', sz=7.5, c=C['cnn'], bold=False)

# 복사 화살표
ax3.annotate('', xy=(4.85+0.23, 5.08), xytext=(4.85+5.6*0.56, 6.22),
             arrowprops=dict(arrowstyle='->', color=C['cnn'], lw=1.5,
                             connectionstyle='arc3,rad=0.35', linestyle='dashed'))
ax3.annotate('', xy=(4.85+9.5*0.56, 5.08), xytext=(4.85+0.5*0.56, 6.22),
             arrowprops=dict(arrowstyle='->', color=C['cnn'], lw=1.5,
                             connectionstyle='arc3,rad=-0.35', linestyle='dashed'))

# key insight
box(ax3, 7.5, 2.85, 4.8, 1.55, '#EAFAF1', ec='#A9DFBF', lw=1.5, r=0.2)
t(ax3, 7.5, 3.45, 'CircRNA advantages:', sz=9.5, c=C['dark'])
t(ax3, 7.5, 3.00, '\u2022  Attn: BSJ positions attend each other', sz=9, c=C['attn'], bold=False)
t(ax3, 7.5, 2.55, '\u2022  CNN: kernel spans BSJ boundary', sz=9, c=C['cnn'], bold=False)

# ── 전역 범례 ─────────────────────────────────────────────────────────────────
handles = [
    mpatches.Patch(color=C['attn'],   label='Attention'),
    mpatches.Patch(color=C['mamba'],  label='Mamba'),
    mpatches.Patch(color=C['cnn'],    label='Circular CNN'),
    mpatches.Patch(color=C['router'], label='Adaptive Router'),
    mpatches.Patch(color=C['cross'],  label='Cross-Attention'),
    mpatches.Patch(color=C['head'],   label='Site Head'),
    mpatches.Patch(color=C['embed'],  label='Embedding / Misc'),
]
fig.legend(handles=handles, loc='lower center', ncol=7,
           fontsize=9.5, framealpha=0.95, edgecolor='#ccc',
           bbox_to_anchor=(0.5, 0.005))

fig.suptitle(
    'CircMAC: Circular-aware Multi-branch Architecture for circRNA–miRNA Binding Site Prediction',
    fontsize=14, fontweight='bold', y=0.985, color=C['dark'])

plt.savefig(str(Path(__file__).parent / 'architecture_figure_v2.pdf'),
            bbox_inches='tight', dpi=300)
plt.savefig(str(Path(__file__).parent / 'architecture_figure_v2.png'),
            bbox_inches='tight', dpi=200)
print('Saved: architecture_figure_v2.pdf / .png')
plt.close()
