"""
논문용 Figure: Self-supervised pretraining strategies for CircMAC
출력: pretraining_figure.pdf / pretraining_figure.png
"""

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.patches import FancyArrowPatch, Arc, FancyBboxPatch
from matplotlib.gridspec import GridSpec

# ── 색상 팔레트 ──────────────────────────────────────────────────────────────
C = {
    'A': '#E74C3C', 'U': '#3498DB', 'G': '#2ECC71', 'C': '#F39C12',
    'mask': '#95A5A6', 'paired': '#8E44AD', 'unpaired': '#BDC3C7',
    'arrow': '#2C3E50', 'bg': '#FAFAFA', 'box': '#ECF0F1',
    'highlight': '#F39C12', 'pos': '#27AE60', 'neg': '#E74C3C',
}

SEQ   = ['A', 'U', 'G', 'C', 'G', 'C', 'A', 'U']
PAIRS = [(0,7),(1,6),(2,5)]   # (i, j) 0-indexed
SS    = [1,1,1,0,0,1,1,1]     # paired=1, unpaired=0
MASK_IDX = [2, 5]

def draw_seq(ax, seq, x0, y, gap=0.9, mask_idx=None, colors=None,
             labels=None, fontsize=13, box=True):
    """서열 박스 그리기. 반환: 각 위치의 x 좌표 리스트"""
    xs = [x0 + i * gap for i in range(len(seq))]
    for i, (x, tok) in enumerate(zip(xs, seq)):
        is_mask = mask_idx and i in mask_idx
        fc = C['mask'] if is_mask else (colors[i] if colors else C.get(tok, '#AAB7B8'))
        if box:
            rect = FancyBboxPatch((x-0.36, y-0.30), 0.72, 0.60,
                                   boxstyle="round,pad=0.05",
                                   facecolor=fc, edgecolor='white', linewidth=1.5, zorder=3)
            ax.add_patch(rect)
        txt = '[M]' if is_mask else tok
        ax.text(x, y, txt, ha='center', va='center', fontsize=fontsize,
                fontweight='bold', color='white' if box else C.get(tok,'#2C3E50'), zorder=4)
        if labels:
            ax.text(x, y-0.60, str(labels[i]), ha='center', va='center',
                    fontsize=10, color=C['paired'] if labels[i]==1 else '#999', zorder=4)
    return xs

def panel_label(ax, letter, x=-0.12, y=1.05):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=16,
            fontweight='bold', va='top', ha='left')

# ── Figure 생성 ───────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('white')
gs = GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.38,
              top=0.93, bottom=0.05, left=0.04, right=0.98)

# ═══════════════════════════════════════════════════════════════════
# (a) MLM
# ═══════════════════════════════════════════════════════════════════
ax = fig.add_subplot(gs[0, 0])
ax.set_xlim(-0.5, 7.5); ax.set_ylim(-1.8, 2.0); ax.axis('off')
ax.set_facecolor(C['bg'])
panel_label(ax, '(a) MLM')

xs = draw_seq(ax, SEQ, 0, 1.1, mask_idx=MASK_IDX)

# 예측 화살표 & 정답
for idx in MASK_IDX:
    ax.annotate('', xy=(xs[idx], 0.35), xytext=(xs[idx], 0.75),
                arrowprops=dict(arrowstyle='->', color=C['arrow'], lw=2))
    fc = C.get(SEQ[idx], '#AAB7B8')
    rect = FancyBboxPatch((xs[idx]-0.36, -0.25), 0.72, 0.60,
                           boxstyle="round,pad=0.05",
                           facecolor=fc, edgecolor=C['highlight'], linewidth=2.5, zorder=3)
    ax.add_patch(rect)
    ax.text(xs[idx], 0.05, SEQ[idx], ha='center', va='center',
            fontsize=13, fontweight='bold', color='white', zorder=4)

ax.text(3.5, -0.85, 'Predict masked tokens\n(15% masking, PAD excluded)',
        ha='center', va='center', fontsize=10, color='#555',
        bbox=dict(boxstyle='round,pad=0.3', facecolor=C['box'], edgecolor='#ccc'))
ax.text(3.5, 1.65, 'Loss:  CrossEntropy (masked positions)',
        ha='center', fontsize=9.5, color='#777', style='italic')

# ═══════════════════════════════════════════════════════════════════
# (b) NTP
# ═══════════════════════════════════════════════════════════════════
ax = fig.add_subplot(gs[0, 1])
ax.set_xlim(-0.5, 7.5); ax.set_ylim(-1.8, 2.0); ax.axis('off')
panel_label(ax, '(b) NTP')

# 입력: shift right (첫 칸은 [BOS])
inp = ['▶'] + SEQ[:-1]
inp_colors = ['#7F8C8D'] + [C.get(t,'#AAB7B8') for t in SEQ[:-1]]
xs_in = draw_seq(ax, inp, 0, 1.1, colors=inp_colors)

# 출력: 원본 서열
xs_out = draw_seq(ax, SEQ, 0, 0.05)

# causal 화살표
for i in range(len(SEQ)):
    ax.annotate('', xy=(xs_out[i], 0.35), xytext=(xs_in[i], 0.75),
                arrowprops=dict(arrowstyle='->', color=C['arrow'], lw=1.5, alpha=0.6))

ax.text(3.5, -0.85, 'Predict next token (causal / autoregressive)\nAligns with Mamba branch',
        ha='center', va='center', fontsize=10, color='#555',
        bbox=dict(boxstyle='round,pad=0.3', facecolor=C['box'], edgecolor='#ccc'))
ax.text(3.5, 1.65, 'Loss:  CrossEntropy (t+1 prediction)',
        ha='center', fontsize=9.5, color='#777', style='italic')

# ═══════════════════════════════════════════════════════════════════
# (c) SSP
# ═══════════════════════════════════════════════════════════════════
ax = fig.add_subplot(gs[0, 2])
ax.set_xlim(-0.5, 7.5); ax.set_ylim(-1.8, 2.0); ax.axis('off')
panel_label(ax, '(c) SSP')

struct_str = ['(','(','(', '.', '.',')', ')', ')']
# 3-class labels: '.'→0  '('→1  ')'→2
SSP_LABEL_MAP = {'.': 0, '(': 1, ')': 2}
SSP_COLOR = {0: C['unpaired'], 1: '#8E44AD', 2: '#3498DB'}
SSP_TEXT_COLOR = {0: '#555555', 1: 'white', 2: 'white'}
ssp_labels = [SSP_LABEL_MAP[s] for s in struct_str]

xs = draw_seq(ax, SEQ, 0, 1.1)

# dot-bracket 기호 표시
for i, (x, s) in enumerate(zip(xs, struct_str)):
    color = '#8E44AD' if s=='(' else ('#3498DB' if s==')' else '#AAAAAA')
    ax.text(x, 0.50, s, ha='center', va='center', fontsize=16,
            color=color, fontweight='bold')

# 3-class 레이블 박스
for i, (x, lbl) in enumerate(zip(xs, ssp_labels)):
    fc = SSP_COLOR[lbl]
    rect = FancyBboxPatch((x-0.30, -0.42), 0.60, 0.52,
                           boxstyle="round,pad=0.04",
                           facecolor=fc, edgecolor='white', linewidth=1, zorder=3)
    ax.add_patch(rect)
    ax.text(x, -0.16, str(lbl), ha='center', va='center',
            fontsize=12, fontweight='bold', color=SSP_TEXT_COLOR[lbl], zorder=4)

# 범례: 0 / 1 / 2 설명
legend_x = 0.0
for lbl, sym, desc in [(1, '(', 'opening'), (2, ')', 'closing'), (0, '.', 'unpaired')]:
    fc = SSP_COLOR[lbl]
    rect = FancyBboxPatch((legend_x-0.25, -1.05), 0.50, 0.38,
                           boxstyle="round,pad=0.03",
                           facecolor=fc, edgecolor='white', zorder=3)
    ax.add_patch(rect)
    ax.text(legend_x, -0.86, str(lbl), ha='center', va='center',
            fontsize=10, fontweight='bold', color=SSP_TEXT_COLOR[lbl], zorder=4)
    ax.text(legend_x, -1.20, f"'{sym}'\n{desc}", ha='center', va='top',
            fontsize=8, color='#555')
    legend_x += 2.5

ax.text(3.5, 1.65, 'Loss:  CrossEntropy  (3 classes per position)',
        ha='center', fontsize=9.5, color='#777', style='italic')

# ═══════════════════════════════════════════════════════════════════
# (d) Pairing
# ═══════════════════════════════════════════════════════════════════
ax = fig.add_subplot(gs[1, 0])
ax.set_xlim(-0.5, 8.5); ax.set_ylim(-2.2, 2.2); ax.axis('off')
panel_label(ax, '(d) Pairing')

SEQ_P = ['A', 'U', 'G', 'C', 'G', 'C']   # shorter for clarity
PAIRS_P = [(0,5),(1,4),(2,3)]
gap = 0.85
xs_p = [0.3 + i * gap for i in range(len(SEQ_P))]
seq_y = 1.3

# 시퀀스 박스
for i, (x, tok) in enumerate(zip(xs_p, SEQ_P)):
    fc = C.get(tok, '#AAB7B8')
    rect = FancyBboxPatch((x-0.33, seq_y-0.28), 0.66, 0.56,
                           boxstyle="round,pad=0.05",
                           facecolor=fc, edgecolor='white', linewidth=1.5, zorder=3)
    ax.add_patch(rect)
    ax.text(x, seq_y, tok, ha='center', va='center', fontsize=13,
            fontweight='bold', color='white', zorder=4)
    ax.text(x, seq_y-0.55, str(i), ha='center', va='top', fontsize=9, color='#888')

# arc (아래쪽) — 반원형 염기쌍 연결
arc_cols = ['#8E44AD', '#3498DB', '#E67E22']
for k, (i, j) in enumerate(PAIRS_P):
    xi, xj = xs_p[i], xs_p[j]
    xm = (xi + xj) / 2
    width = xj - xi
    depth = 0.55 + k * 0.20
    arc = Arc((xm, seq_y - 0.28), width, depth * 2, angle=0,
              theta1=180, theta2=360, color=arc_cols[k], lw=2.5, zorder=5)
    ax.add_patch(arc)
    for xp in [xi, xj]:
        ax.plot(xp, seq_y - 0.28, 'o', color=arc_cols[k], ms=5, zorder=6)

# ── 오른쪽: 6×6 매트릭스 인셋 ──────────────────────────────────────
mat_x0 = 4.8; mat_y0 = -0.25
cell = 0.38
L_p = len(SEQ_P)
pair_set = set()
for (i, j) in PAIRS_P:
    pair_set.add((i, j)); pair_set.add((j, i))

ax.text(mat_x0 + (L_p * cell)/2, mat_y0 + L_p * cell + 0.25,
        'Pairing matrix\n(L × L)', ha='center', fontsize=9, color='#555', style='italic')

for r in range(L_p):
    for c2 in range(L_p):
        x = mat_x0 + c2 * cell
        y = mat_y0 + (L_p - 1 - r) * cell
        val = 1 if (r, c2) in pair_set else 0
        fc = '#8E44AD' if val == 1 else '#F0F0F0'
        rect = plt.Rectangle((x, y), cell*0.92, cell*0.92,
                              facecolor=fc, edgecolor='white', linewidth=1.2, zorder=3)
        ax.add_patch(rect)
        if val == 1:
            ax.text(x + cell*0.46, y + cell*0.46, '1',
                    ha='center', va='center', fontsize=8, color='white',
                    fontweight='bold', zorder=4)

# 행/열 인덱스
for k2 in range(L_p):
    ax.text(mat_x0 + k2*cell + cell*0.46, mat_y0 + L_p*cell + 0.05,
            str(k2), ha='center', fontsize=8, color='#888')
    ax.text(mat_x0 - 0.12, mat_y0 + (L_p-1-k2)*cell + cell*0.46,
            str(k2), ha='right', fontsize=8, color='#888')

# 화살표: arc → matrix
ax.annotate('', xy=(mat_x0 - 0.08, mat_y0 + 1.0),
            xytext=(3.0, seq_y - 0.8),
            arrowprops=dict(arrowstyle='->', color='#AAA', lw=1.5, linestyle='dashed'))

ax.text(3.5, -1.75, 'Predict L×L binary pairing matrix\n(which position pairs with which)',
        ha='center', va='center', fontsize=10, color='#555',
        bbox=dict(boxstyle='round,pad=0.3', facecolor=C['box'], edgecolor='#ccc'))
ax.text(3.5, 2.0, 'Loss:  BCEWithLogits (L×L matrix)',
        ha='center', fontsize=9.5, color='#777', style='italic')

# ═══════════════════════════════════════════════════════════════════
# (e) CPCL
# ═══════════════════════════════════════════════════════════════════
ax = fig.add_subplot(gs[1, 1])
ax.set_xlim(-1, 8); ax.set_ylim(-2.2, 2.2); ax.axis('off')
panel_label(ax, '(e) CPCL')

# 원형 서열 시각화
def draw_circle_seq(ax, seq, cx, cy, r=0.9, start_angle=90, color_offset=0, label_r_offset=0.28):
    n = len(seq)
    angles = [start_angle - i * 360/n for i in range(n)]
    xs, ys = [], []
    for i, (tok, ang) in enumerate(zip(seq, angles)):
        rad = np.radians(ang)
        x = cx + r * np.cos(rad)
        y = cy + r * np.sin(rad)
        xs.append(x); ys.append(y)
        fc = C.get(tok, '#AAB7B8')
        rect = FancyBboxPatch((x-0.25, y-0.22), 0.50, 0.44,
                               boxstyle="round,pad=0.04",
                               facecolor=fc, edgecolor='white', linewidth=1.5, zorder=4)
        ax.add_patch(rect)
        ax.text(x, y, tok, ha='center', va='center', fontsize=11,
                fontweight='bold', color='white', zorder=5)
    # 원 테두리
    circle = plt.Circle((cx, cy), r, fill=False,
                         edgecolor='#BDC3C7', lw=1.5, linestyle='--', zorder=2)
    ax.add_patch(circle)
    return xs, ys

# 버전 1 (offset=0): 왼쪽
xs1, ys1 = draw_circle_seq(ax, SEQ, cx=1.5, cy=0.5, r=1.0)
ax.text(1.5, -0.75, 'offset = 0', ha='center', fontsize=10, color='#555')

# 버전 2 (offset=3): 오른쪽
rot_seq = SEQ[3:] + SEQ[:3]
xs2, ys2 = draw_circle_seq(ax, rot_seq, cx=5.5, cy=0.5, r=1.0)
ax.text(5.5, -0.75, 'offset = 3', ha='center', fontsize=10, color='#555')

# 인코더 → 벡터
for cx, lbl, col in [(1.5, 'z₁', C['pos']), (5.5, 'z₂', C['pos'])]:
    ax.annotate('', xy=(cx, -1.15), xytext=(cx, -0.85),
                arrowprops=dict(arrowstyle='->', color=col, lw=2))
    rect = FancyBboxPatch((cx-0.45, -1.55), 0.90, 0.40,
                           boxstyle="round,pad=0.05",
                           facecolor=col, edgecolor='white', linewidth=1.5, zorder=3)
    ax.add_patch(rect)
    ax.text(cx, -1.35, lbl, ha='center', va='center',
            fontsize=12, fontweight='bold', color='white', zorder=4)

# sim ↑ 화살표 (양방향)
ax.annotate('', xy=(4.9, -1.35), xytext=(2.1, -1.35),
            arrowprops=dict(arrowstyle='<->', color=C['highlight'], lw=2.5))
ax.text(3.5, -1.10, 'sim ↑', ha='center', fontsize=11,
        color=C['highlight'], fontweight='bold')

ax.text(3.5, -1.85, 'NT-Xent (InfoNCE) loss\nBatch negatives: sim(z₁, z_neg) ↓',
        ha='center', va='center', fontsize=10, color='#555',
        bbox=dict(boxstyle='round,pad=0.3', facecolor=C['box'], edgecolor='#ccc'))
ax.text(3.5, 1.75, 'circRNA-specific: rotation-invariant representation',
        ha='center', fontsize=9.5, color='#777', style='italic')

# ═══════════════════════════════════════════════════════════════════
# (f) 요약 테이블
# ═══════════════════════════════════════════════════════════════════
ax = fig.add_subplot(gs[1, 2])
ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis('off')
panel_label(ax, '(f) Summary')

headers = ['Method', 'Objective', 'circRNA\nspecific', 'Key branch']
rows = [
    ['MLM',     'Token recovery',       '✗', 'Attn, CNN'],
    ['NTP',     'Next token',           '✗', 'Mamba'],
    ['SSP',     '3-class structure',    '△', 'Attn, CNN'],
    ['Pairing', 'Pairing matrix',       '△', 'Attn'],
    ['CPCL',    'Rotation invariance',  'Yes*', 'All'],
]

col_x = [0.2, 2.5, 5.8, 7.2]
row_h = 0.85
header_y = 6.2

# 헤더
for x, h in zip(col_x, headers):
    ax.text(x, header_y, h, fontsize=10.5, fontweight='bold',
            color='white', va='center')
header_bg = FancyBboxPatch((-0.1, header_y-0.35), 10.2, 0.72,
                            boxstyle="round,pad=0.05",
                            facecolor='#2C3E50', edgecolor='none', zorder=2)
ax.add_patch(header_bg)
for x, h in zip(col_x, headers):
    ax.text(x, header_y, h, fontsize=10.5, fontweight='bold',
            color='white', va='center', zorder=3)

# 행
for r, row in enumerate(rows):
    y = header_y - (r+1) * row_h
    fc = '#F8F9FA' if r % 2 == 0 else 'white'
    bg = FancyBboxPatch((-0.1, y-0.35), 10.2, 0.72,
                         boxstyle="round,pad=0.02",
                         facecolor=fc, edgecolor='#E0E0E0', linewidth=0.8, zorder=1)
    ax.add_patch(bg)
    method_colors = {'MLM':'#E74C3C','NTP':'#3498DB','SSP':'#2ECC71',
                     'Pairing':'#9B59B6','CPCL':'#F39C12'}
    for c, (x, txt) in enumerate(zip(col_x, row)):
        color = method_colors.get(txt, '#2C3E50')
        fw = 'bold' if c == 0 else 'normal'
        ax.text(x, y, txt, fontsize=10, va='center',
                color=color if c==0 else '#2C3E50', fontweight=fw, zorder=2)

# 범례 △
ax.text(0.0, 0.3, '△ Structure labels from RNAsubopt --circ',
        fontsize=8.5, color='#888', style='italic')

# ── Title ──────────────────────────────────────────────────────────────────
fig.suptitle('Self-Supervised Pretraining Strategies for CircRNA Representation Learning',
             fontsize=15, fontweight='bold', y=0.97, color='#2C3E50')

plt.savefig(str(Path(__file__).parent / 'pretraining_figure.pdf'),
            bbox_inches='tight', dpi=300)
plt.savefig(str(Path(__file__).parent / 'pretraining_figure.png'),
            bbox_inches='tight', dpi=200)
print("Saved: pretraining_figure.pdf / .png")
plt.show()
