"""
CircMAC 아키텍처 그림

Layout (세로):
  (a) 전체 파이프라인 — 좌측
  (b) CircMACBlock 내부 — 우측 상단
  (c) Circular 특성 (Attn bias, CNN padding) — 우측 하단
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc, Wedge
from matplotlib.gridspec import GridSpec

# ── 색상 ───────────────────────────────────────────────────────────────────
C = dict(
    attn   = '#8E44AD',   # 보라
    mamba  = '#2980B9',   # 파랑
    cnn    = '#27AE60',   # 초록
    router = '#E67E22',   # 주황
    head   = '#E74C3C',   # 빨강
    embed  = '#7F8C8D',   # 회색
    cross  = '#16A085',   # 청록
    bg     = '#FAFAFA',
    box    = '#ECF0F1',
    dark   = '#2C3E50',
    gold   = '#F39C12',
    light  = '#F8F9FA',
)

def rbox(ax, x, y, w, h, fc, ec='white', lw=1.5, radius=0.3, alpha=1.0, zorder=3):
    p = FancyBboxPatch((x - w/2, y - h/2), w, h,
                        boxstyle=f'round,pad={radius}',
                        facecolor=fc, edgecolor=ec, linewidth=lw,
                        alpha=alpha, zorder=zorder)
    ax.add_patch(p)
    return p

def arrow(ax, x0, y0, x1, y1, color='#555', lw=1.8, style='->', zorder=5):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle='arc3,rad=0'))

def txt(ax, x, y, s, size=10, color='white', bold=True, ha='center', va='center', zorder=6):
    ax.text(x, y, s, fontsize=size, color=color,
            fontweight='bold' if bold else 'normal',
            ha=ha, va=va, zorder=zorder)

fig = plt.figure(figsize=(18, 11))
fig.patch.set_facecolor('white')
gs = GridSpec(2, 2, figure=fig,
              left=0.03, right=0.97, top=0.93, bottom=0.04,
              wspace=0.08, hspace=0.12,
              width_ratios=[1, 1.1], height_ratios=[1, 0.85])

# ══════════════════════════════════════════════════════════════════════════════
# (a) 전체 파이프라인  — left column (spans both rows)
# ══════════════════════════════════════════════════════════════════════════════
ax = fig.add_subplot(gs[:, 0])
ax.set_xlim(0, 10); ax.set_ylim(0, 22)
ax.axis('off')
ax.set_facecolor(C['bg'])
ax.text(5, 21.4, '(a) CircMAC Pipeline', fontsize=13, fontweight='bold',
        ha='center', color=C['dark'])

# ── circRNA input ────────────────────────────────────────────────────────────
rbox(ax, 3, 20.4, 4.2, 0.8, C['embed'], ec='#bbb', lw=1)
txt(ax, 3, 20.4, 'circRNA sequence', size=10, color=C['dark'])
rbox(ax, 7.5, 20.4, 2.4, 0.8, C['embed'], ec='#bbb', lw=1)
txt(ax, 7.5, 20.4, 'miRNA seq.', size=10, color=C['dark'])

# ── Embedding ────────────────────────────────────────────────────────────────
arrow(ax, 3, 20.0, 3, 19.3)
arrow(ax, 7.5, 20.0, 7.5, 19.3)
rbox(ax, 3, 18.9, 4.2, 0.7, C['embed'])
txt(ax, 3, 18.9, 'K-mer Tokenizer + Embedding', size=9)
rbox(ax, 7.5, 18.9, 2.4, 0.7, C['embed'])
txt(ax, 7.5, 18.9, 'Embedding', size=9)

# ── CircMAC Encoder ──────────────────────────────────────────────────────────
arrow(ax, 3, 18.55, 3, 17.85)
# Downsample
rbox(ax, 3, 17.5, 4.6, 0.65, '#AAB7B8')
txt(ax, 3, 17.5, 'Downsample ×½  (Conv1D stride=2)', size=8.5, color=C['dark'])

arrow(ax, 3, 17.17, 3, 16.5)

# N× CircMACBlock (stacked)
for i, yc in enumerate([16.1, 15.1, 14.1]):
    alpha = 1.0 if i == 0 else (0.7 if i == 1 else 0.45)
    rbox(ax, 3, yc, 5.0, 0.72, C['dark'], alpha=alpha)
    if i == 0:
        txt(ax, 3, yc, 'CircMACBlock  ×  N', size=10)
    else:
        txt(ax, 3, yc, '         ·  ·  ·', size=10, color='white')
    if i < 2:
        arrow(ax, 3, yc - 0.36, 3, yc - 0.64)

# bracket & label
bx0, bx1, by0, by1 = 0.2, 0.55, 13.7, 16.48
ax.plot([bx0, bx0, bx1], [by0, by1, by1], color='#888', lw=1.5)
ax.plot([bx0, bx0, bx1], [by0, by0, bx1], color='#888', lw=1.5)
ax.text(0.05, (by0+by1)/2, f'N layers\n(default=6)', fontsize=8, color='#666',
        ha='center', va='center', rotation=90)

arrow(ax, 3, 13.74, 3, 13.1)

# Upsample
rbox(ax, 3, 12.75, 4.6, 0.65, '#AAB7B8')
txt(ax, 3, 12.75, 'Upsample ×2  +  Skip connection', size=8.5, color=C['dark'])

arrow(ax, 3, 12.42, 3, 11.8)

# skip connection 표시
ax.annotate('', xy=(3 - 2.5, 12.42), xytext=(3 - 2.5, 18.55),
            arrowprops=dict(arrowstyle='->', color='#AAB7B8',
                            lw=1.5, linestyle='dashed',
                            connectionstyle='arc3,rad=0'))
ax.text(0.18, 15.5, 'skip', fontsize=8, color='#AAB7B8', rotation=90, va='center')

# ── circRNA Encoder output ──
rbox(ax, 3, 11.45, 5.0, 0.65, C['dark'])
txt(ax, 3, 11.45, 'E_circ  [B, L, D]', size=10)

# miRNA encoder (간단 MLP)
arrow(ax, 7.5, 18.55, 7.5, 11.8)
rbox(ax, 7.5, 11.45, 2.4, 0.65, C['embed'])
txt(ax, 7.5, 11.45, 'E_mirna\n[B, M, D]', size=8.5)

# ── Cross-Attention (Interaction) ─────────────────────────────────────────────
arrow(ax, 3, 11.12, 3, 10.4)
arrow(ax, 7.5, 11.12, 7.5, 10.55)
ax.annotate('', xy=(4.8, 10.2), xytext=(7.3, 10.55),
            arrowprops=dict(arrowstyle='->', color=C['cross'], lw=1.5))

rbox(ax, 3, 10.0, 5.8, 0.72, C['cross'])
txt(ax, 3, 10.0, 'Cross-Attention  (circRNA ← miRNA)', size=9.5)
ax.text(6.8, 10.0, 'Q: E_circ\nK,V: E_mirna', fontsize=7.5,
        color=C['cross'], ha='left', va='center')

arrow(ax, 3, 9.64, 3, 9.0)

# ── Site Head ─────────────────────────────────────────────────────────────────
rbox(ax, 3, 8.65, 5.0, 0.65, '#E67E22')
txt(ax, 3, 8.65, 'MultiKernelCNN  (k=3,5,7)  feature enhancer', size=9)

arrow(ax, 3, 8.32, 3, 7.7)

rbox(ax, 3, 7.35, 5.0, 0.65, C['head'])
txt(ax, 3, 7.35, 'Conv1D Site Head  →  per-position logits', size=9)

arrow(ax, 3, 7.02, 3, 6.4)

# ── Output ────────────────────────────────────────────────────────────────────
rbox(ax, 3, 6.08, 4.4, 0.58, '#CB4335')
txt(ax, 3, 6.08, 'Binding Site Prediction  [B, L, 2]', size=9.5)

# ── Loss ─────────────────────────────────────────────────────────────────────
arrow(ax, 3, 5.79, 3, 5.18)
rbox(ax, 3, 4.9, 4.0, 0.55, '#922B21')
txt(ax, 3, 4.9, 'CrossEntropy Loss (site-level)', size=9)

# ── Binding Head (binary, on the side) ────────────────────────────────────────
ax.annotate('', xy=(7.0, 8.65), xytext=(5.85, 9.64),
            arrowprops=dict(arrowstyle='->', color='#7F8C8D', lw=1.2, linestyle='dashed'))
rbox(ax, 7.8, 8.65, 2.2, 0.55, '#7F8C8D')
txt(ax, 7.8, 8.65, 'Binding Head\n(binary cls.)', size=8.5)

# ── config box ────────────────────────────────────────────────────────────────
rbox(ax, 7.5, 4.5, 4.0, 3.0, C['light'], ec='#ccc', lw=1.2)
ax.text(7.5, 5.8, 'Default Config', fontsize=9.5, fontweight='bold',
        ha='center', color=C['dark'])
for i, (k, v) in enumerate([('d_model', '128'), ('n_layer', '6'),
                              ('n_heads', '8'), ('max_len', '1022'),
                              ('kernel_size', '7'), ('d_state', '16')]):
    ax.text(5.7, 5.42 - i*0.42, f'{k}', fontsize=8.5, color='#555', ha='left')
    ax.text(9.3, 5.42 - i*0.42, f'{v}', fontsize=8.5, color=C['dark'],
            ha='right', fontweight='bold')

# ══════════════════════════════════════════════════════════════════════════════
# (b) CircMACBlock 내부  — right top
# ══════════════════════════════════════════════════════════════════════════════
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_xlim(0, 10); ax2.set_ylim(0, 11)
ax2.axis('off')
ax2.text(5, 10.55, '(b) CircMACBlock (single layer)', fontsize=13,
         fontweight='bold', ha='center', color=C['dark'])

# Input
rbox(ax2, 5, 9.85, 6.0, 0.65, C['dark'])
txt(ax2, 5, 9.85, 'Input  x  [B, L, D]', size=10)

# in_proj
arrow(ax2, 5, 9.52, 5, 8.95)
rbox(ax2, 5, 8.65, 6.0, 0.55, '#626567')
txt(ax2, 5, 8.65, 'in_proj  (Linear: D → 4D)', size=9.5)

# split 화살표
arrow(ax2, 5, 8.37, 5, 7.85)
ax2.text(5, 7.78, 'Q, K, V  [D each]    base  [D]',
         fontsize=9, ha='center', color='#555')

# 세 브랜치 화살표
for x, label in [(2.2, 'Q,K,V'), (5.0, 'base'), (7.8, 'base')]:
    arrow(ax2, x if x != 5.0 else 3.5, 7.55,
          x, 6.85,
          color=C['attn'] if x == 2.2 else (C['mamba'] if x == 5.0 else C['cnn']))

# 세 브랜치 갈라지는 선
ax2.plot([1.2, 8.8], [7.6, 7.6], color='#bbb', lw=1.2, zorder=1)
for x in [2.2, 5.0, 7.8]:
    ax2.plot([x, x], [7.6, 7.0], color='#bbb', lw=1.2, zorder=1)

# Branch 1: Attention
rbox(ax2, 2.2, 6.55, 3.4, 0.72, C['attn'])
txt(ax2, 2.2, 6.7, 'Multi-Head', size=9.5)
txt(ax2, 2.2, 6.38, 'Self-Attention', size=9.5)

rbox(ax2, 2.2, 5.55, 3.2, 0.5, '#A569BD')
txt(ax2, 2.2, 5.55, '+ Circular Rel. Bias', size=8.5)

arrow(ax2, 2.2, 6.19, 2.2, 5.82)
arrow(ax2, 2.2, 5.30, 2.2, 4.72)

# Branch 2: Mamba
rbox(ax2, 5.0, 6.55, 3.4, 0.72, C['mamba'])
txt(ax2, 5.0, 6.7, 'Mamba SSM', size=9.5)
txt(ax2, 5.0, 6.38, '(causal sequential)', size=8.5)
arrow(ax2, 5.0, 6.19, 5.0, 4.72)

# Branch 3: CNN
rbox(ax2, 7.8, 6.55, 3.4, 0.72, C['cnn'])
txt(ax2, 7.8, 6.7, 'Depthwise CNN', size=9.5)
txt(ax2, 7.8, 6.38, '(k=7, circular pad)', size=8.5)
arrow(ax2, 7.8, 6.19, 7.8, 4.72)

# RMSNorm 아래 각 브랜치
for x, c in [(2.2, C['attn']), (5.0, C['mamba']), (7.8, C['cnn'])]:
    rbox(ax2, x, 4.45, 2.8, 0.45, c, alpha=0.6)
    txt(ax2, x, 4.45, 'RMSNorm', size=8.5)
    arrow(ax2, x, 4.22, x, 3.62)

# Router
ax2.plot([1.2, 8.8], [3.58, 3.58], color='#bbb', lw=1.2, zorder=1)
for x in [2.2, 5.0, 7.8]:
    ax2.plot([x, x], [3.58, 3.1], color='#bbb', lw=1.2, zorder=1)

rbox(ax2, 5.0, 2.8, 6.5, 0.72, C['router'])
txt(ax2, 5.0, 2.95, 'Adaptive Router', size=10)
txt(ax2, 5.0, 2.62, 'gates = Softmax(Linear([mean(attn), mean(mamba), mean(cnn)]))', size=8)

arrow(ax2, 5.0, 2.44, 5.0, 1.85)

# out_proj
rbox(ax2, 5.0, 1.55, 4.5, 0.55, '#626567')
txt(ax2, 5.0, 1.55, 'out_proj  (Linear: D → D)', size=9.5)

arrow(ax2, 5.0, 1.27, 5.0, 0.72)

# Residual + output
rbox(ax2, 5.0, 0.42, 6.0, 0.55, C['dark'])
txt(ax2, 5.0, 0.42, 'x  +  out_proj(·)    [Residual]', size=9.5)

# residual skip
ax2.annotate('', xy=(8.2, 0.42), xytext=(8.2, 9.85),
            arrowprops=dict(arrowstyle='->', color='#AAB7B8',
                            lw=1.5, linestyle='dashed'))
ax2.text(8.6, 5.1, 'residual', fontsize=8, color='#AAB7B8', rotation=90, va='center')

# ══════════════════════════════════════════════════════════════════════════════
# (c) Circular 특성 — right bottom
# ══════════════════════════════════════════════════════════════════════════════
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_xlim(0, 10); ax3.set_ylim(0, 9)
ax3.axis('off')
ax3.text(5, 8.55, '(c) CircRNA-specific Components', fontsize=13,
         fontweight='bold', ha='center', color=C['dark'])

# ── Left: Circular Relative Bias ─────────────────────────────────────────────
ax3.text(2.5, 7.95, 'Circular Relative Position Bias', fontsize=10,
         fontweight='bold', ha='center', color=C['attn'])

# 원형 거리 예시 그리기
L_ex = 8
cx, cy, r = 2.5, 5.7, 1.6
angles_ex = np.linspace(90, 90-360, L_ex, endpoint=False)
for i in range(L_ex):
    a = np.radians(angles_ex[i])
    x = cx + r * np.cos(a); y = cy + r * np.sin(a)
    fc = C['attn'] if i in [0, 7] else ('#D5D8DC')
    circle = plt.Circle((x, y), 0.22, color=fc, zorder=4)
    ax3.add_patch(circle)
    ax3.text(x, y, str(i), ha='center', va='center', fontsize=8,
             fontweight='bold', color='white' if i in [0,7] else C['dark'], zorder=5)

# 원형 거리 호 (0-7은 실제로 1칸)
a0 = np.radians(angles_ex[0]); a7 = np.radians(angles_ex[7])
x0 = cx + r * np.cos(a0); y0 = cy + r * np.sin(a0)
x7 = cx + r * np.cos(a7); y7 = cy + r * np.sin(a7)
ax3.annotate('', xy=(x7, y7+0.2), xytext=(x0, y0+0.22),
             arrowprops=dict(arrowstyle='<->', color=C['attn'],
                             lw=2, connectionstyle='arc3,rad=-0.3'))
ax3.text(cx - 0.05, cy + r + 0.55, 'd_circ(0,7) = 1', fontsize=8.5,
         ha='center', color=C['attn'], fontweight='bold')

rbox(ax3, 2.5, 3.85, 4.6, 1.05, '#F5EEF8', ec='#C39BD3', lw=1.2, radius=0.15)
ax3.text(2.5, 4.28, 'd_linear(i,j) = |i−j|', fontsize=9, ha='center', color='#922B21')
ax3.text(2.5, 3.92, 'd_circular(i,j) = min(|i−j|, L−|i−j|)', fontsize=9,
         ha='center', color=C['attn'], fontweight='bold')

rbox(ax3, 2.5, 2.95, 4.6, 0.75, '#F5EEF8', ec='#C39BD3', lw=1.2, radius=0.15)
ax3.text(2.5, 3.05, 'Attn bias = −slope × d_circular(i, j)', fontsize=9,
         ha='center', color=C['attn'], style='italic')

# ── Right: Circular CNN Padding ───────────────────────────────────────────────
ax3.text(7.5, 7.95, 'Circular Padding (CNN)', fontsize=10,
         fontweight='bold', ha='center', color=C['cnn'])

# 선형 패딩 vs 원형 패딩 비교
seq_nt = ['G', 'C', 'A', 'U', 'G', 'C']
seq_colors = [{'G': '#2ECC71', 'C': '#F39C12', 'A': '#E74C3C', 'U': '#3498DB'}[n]
              for n in seq_nt]

# Linear (zero padding)
ax3.text(7.5, 7.35, 'Linear (zero pad)', fontsize=9, ha='center', color='#922B21')
pad_seq_lin = ['0', '0'] + seq_nt + ['0', '0']
pad_col_lin = ['#D5D8DC', '#D5D8DC'] + seq_colors + ['#D5D8DC', '#D5D8DC']
for i, (nt, fc) in enumerate(zip(pad_seq_lin, pad_col_lin)):
    x = 4.9 + i * 0.54
    circle = plt.Circle((x, 6.9), 0.22, color=fc, zorder=4)
    ax3.add_patch(circle)
    ax3.text(x, 6.9, nt, ha='center', va='center', fontsize=7.5,
             fontweight='bold', color='white', zorder=5)
# kernel window
rbox(ax3, 4.9 + 3*0.54, 6.9, 1.58, 0.54, 'none', ec='#922B21', lw=2, radius=0.08, zorder=6)
ax3.text(4.9 + 3*0.54, 6.3, 'kernel', fontsize=7.5, ha='center', color='#922B21')

# Circular padding
ax3.text(7.5, 5.85, 'Circular pad (CircMAC)', fontsize=9, ha='center', color=C['cnn'])
pad_seq_circ = [seq_nt[-2], seq_nt[-1]] + seq_nt + [seq_nt[0], seq_nt[1]]
pad_col_circ = [seq_colors[-2], seq_colors[-1]] + seq_colors + [seq_colors[0], seq_colors[1]]
circ_border = [True, True] + [False]*len(seq_nt) + [True, True]
for i, (nt, fc, border) in enumerate(zip(pad_seq_circ, pad_col_circ, circ_border)):
    x = 4.9 + i * 0.54
    circle = plt.Circle((x, 5.38), 0.22, color=fc, zorder=4,
                         linewidth=2 if border else 0)
    ax3.add_patch(circle)
    if border:
        circle2 = plt.Circle((x, 5.38), 0.22, fill=False,
                              edgecolor=C['cnn'], linewidth=2, zorder=5)
        ax3.add_patch(circle2)
    ax3.text(x, 5.38, nt, ha='center', va='center', fontsize=7.5,
             fontweight='bold', color='white', zorder=6)
ax3.text(4.9 + 0.54*0.5, 5.38, '', ha='center')
# kernel window
rbox(ax3, 4.9 + 3*0.54, 5.38, 1.58, 0.54, 'none', ec=C['cnn'], lw=2, radius=0.08, zorder=7)
ax3.text(4.9 + 3*0.54, 4.8, 'kernel', fontsize=7.5, ha='center', color=C['cnn'])

# 복사 화살표
ax3.annotate('', xy=(4.9 + 0.54*0.5, 5.59), xytext=(4.9 + 5.54*0.54, 6.68),
             arrowprops=dict(arrowstyle='->', color=C['cnn'], lw=1.5,
                             connectionstyle='arc3,rad=0.3', linestyle='dashed'))
ax3.annotate('', xy=(4.9 + 9.5*0.54, 5.59), xytext=(4.9 + 0.54*0.54, 6.68),
             arrowprops=dict(arrowstyle='->', color=C['cnn'], lw=1.5,
                             connectionstyle='arc3,rad=-0.3', linestyle='dashed'))
ax3.text(9.8, 6.1, 'wrap\naround\nBSJ', fontsize=7.5, ha='center',
         color=C['cnn'], style='italic')

# key insight box
rbox(ax3, 7.5, 3.9, 4.8, 1.8, '#EAFAF1', ec='#A9DFBF', lw=1.5, radius=0.2)
ax3.text(7.5, 4.65, 'CircRNA-specific advantages:', fontsize=9,
         fontweight='bold', ha='center', color=C['dark'])
ax3.text(7.5, 4.25, u'\u2022 Attn: BSJ positions attend each other', fontsize=8.5,
         ha='center', color=C['attn'])
ax3.text(7.5, 3.88, u'\u2022 CNN: kernel spans BSJ boundary', fontsize=8.5,
         ha='center', color=C['cnn'])

# ── 전역 범례 ─────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(color=C['attn'],  label='Attention branch'),
    mpatches.Patch(color=C['mamba'], label='Mamba branch'),
    mpatches.Patch(color=C['cnn'],   label='Circular CNN branch'),
    mpatches.Patch(color=C['router'],label='Adaptive Router'),
    mpatches.Patch(color=C['cross'], label='Cross-Attention'),
    mpatches.Patch(color=C['head'],  label='Site Head'),
]
fig.legend(handles=legend_items, loc='lower center', ncol=6,
           fontsize=9.5, framealpha=0.95, edgecolor='#ccc',
           bbox_to_anchor=(0.5, 0.0))

fig.suptitle('CircMAC: Circular-aware Multi-branch Architecture for circRNA–miRNA Binding Site Prediction',
             fontsize=14, fontweight='bold', y=0.985, color=C['dark'])

plt.savefig('/workspace/volume/cmi_mac/docs/paper_cmi/architecture_figure.pdf',
            bbox_inches='tight', dpi=300)
plt.savefig('/workspace/volume/cmi_mac/docs/paper_cmi/architecture_figure.png',
            bbox_inches='tight', dpi=200)
print('Saved: architecture_figure.pdf / .png')
plt.close()
