"""
논문용 Results Figures
- Figure A: EXP3 — Encoder Architecture Comparison
- Figure B: EXP4 — CircMAC Ablation Study
- Figure C: EXP5 — Interaction Mechanism
- Figure D: EXP2 — Pretraining Strategy (exp2v3 임시; exp2v4로 업데이트 예정)
- Figure E: EXP6 — Site Head Structure

출력: results_figure_*.pdf / .png
"""

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.gridspec import GridSpec

# ── 색상 팔레트 ───────────────────────────────────────────────────────────────
BLUE   = '#2980B9'
GREEN  = '#27AE60'
ORANGE = '#E67E22'
RED    = '#E74C3C'
PURPLE = '#8E44AD'
TEAL   = '#16A085'
GRAY   = '#95A5A6'
DARK   = '#2C3E50'
GOLD   = '#F39C12'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'axes.grid.axis': 'x',
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

def hbar(ax, names, means, stds, colors, title, xlabel='F1 Score',
         baseline=None, baseline_label=None, xlim=None, highlight_idx=None):
    y = np.arange(len(names))
    bars = ax.barh(y, means, xerr=stds, color=colors, edgecolor='white',
                   linewidth=0.8, height=0.65, capsize=3,
                   error_kw={'elinewidth': 1.5, 'ecolor': '#555'})

    if baseline is not None:
        ax.axvline(baseline, color=RED, linestyle='--', lw=1.5, alpha=0.7, zorder=5)
        if baseline_label:
            ax.text(baseline + 0.001, len(names) - 0.3, baseline_label,
                    color=RED, fontsize=8, va='top', alpha=0.8)

    # highlight best bar
    if highlight_idx is not None:
        bars[highlight_idx].set_edgecolor(GOLD)
        bars[highlight_idx].set_linewidth(2.5)

    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(m + s + 0.002, i, f'{m:.4f}', va='center', fontsize=9.5, color=DARK)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10.5)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=8)
    if xlim:
        ax.set_xlim(*xlim)
    else:
        xmax = max(m + s for m, s in zip(means, stds))
        ax.set_xlim(min(means) * 0.93, xmax * 1.08)

# ═══════════════════════════════════════════════════════════════════
# Figure A: EXP3 — Encoder Comparison
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 7))
fig.patch.set_facecolor('white')

names_e3 = [
    'RNABERT (frz)', 'RNAErnie (tr)', 'RNABERT (tr)', 'RNA-FM (frz)',
    'RNA-MSM (frz)', 'RNAErnie (frz)', 'RNA-MSM (tr)',
    'Transformer', 'LSTM', 'Hymba', 'Mamba', 'CircMAC',
]
means_e3 = [0.5923, 0.5922, 0.5964, 0.6023, 0.6034, 0.6077, 0.6320,
            0.6060, 0.6486, 0.6961, 0.7065, 0.7380]
stds_e3  = [0.0026, 0.0040, 0.0025, 0.0016, 0.0008, 0.0025, 0.0058,
            0.0075, 0.0026, 0.0050, 0.0045, 0.0025]

# 색: RNA models = 보라 계열 (frz=연, tr=진), general encoders = 파랑, CircMAC = 주황
colors_e3 = [
    '#C39BD3', '#9B59B6', '#A569BD', '#85C1E9',
    '#7FB3D3', '#AED6F1', '#5DADE2',
    '#7DCEA0', '#27AE60', '#1ABC9C', '#2980B9', ORANGE,
]

hbar(ax, names_e3, means_e3, stds_e3, colors_e3,
     title='(A) Encoder Architecture Comparison',
     xlim=(0.55, 0.80), highlight_idx=len(names_e3)-1)

# 범례
legend_patches = [
    mpatches.Patch(color='#9B59B6', label='RNA LMs (trainable)'),
    mpatches.Patch(color='#AED6F1', label='RNA LMs (frozen)'),
    mpatches.Patch(color='#2980B9', label='General encoders'),
    mpatches.Patch(color=ORANGE,    label='CircMAC (ours)'),
]
ax.legend(handles=legend_patches, loc='lower right', fontsize=9.5,
          framealpha=0.9, edgecolor='#ccc')

plt.tight_layout()
plt.savefig(str(Path(__file__).parent / 'results_fig_A_encoder.pdf'), bbox_inches='tight', dpi=300)
plt.savefig(str(Path(__file__).parent / 'results_fig_A_encoder.png'), bbox_inches='tight', dpi=200)
plt.close()
print("Saved: results_fig_A_encoder")

# ═══════════════════════════════════════════════════════════════════
# Figure B: EXP4 — Ablation Study
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 6.5))
fig.patch.set_facecolor('white')

names_e4 = [
    'Attn only', 'CNN only', 'Mamba only',
    'No Mamba', 'No Conv', 'No Attn',
    'No circ bias', 'No circ pad',
    'CircMAC (full)',
]
means_e4 = [0.6755, 0.6667, 0.6967,
            0.6766, 0.6861, 0.7348,
            0.7373, 0.7402,
            0.7397]
stds_e4  = [0.0317, 0.0026, 0.0036,
            0.0024, 0.0090, 0.0024,
            0.0022, 0.0057,
            0.0032]

colors_e4 = [
    '#85C1E9', '#AED6F1', '#A9DFBF',   # single-branch
    '#F0B27A', '#FAD7A0', '#D2B4DE',   # two-branch (remove one)
    '#ABEBC6', '#A9CCE3',              # no circular components
    ORANGE,                            # full
]

hbar(ax, names_e4, means_e4, stds_e4, colors_e4,
     title='(B) CircMAC Ablation Study',
     baseline=0.7397, baseline_label='CircMAC (full)',
     xlim=(0.62, 0.79), highlight_idx=len(names_e4)-1)

# 구분선
ax.axhline(2.5, color='#ccc', lw=1, linestyle=':')
ax.axhline(5.5, color='#ccc', lw=1, linestyle=':')
ax.text(0.621, 1.0, 'Single branch', fontsize=8.5, color='#888', va='center', style='italic')
ax.text(0.621, 4.0, 'Remove one branch', fontsize=8.5, color='#888', va='center', style='italic')
ax.text(0.621, 6.5, 'No circular\ncomponent', fontsize=8.5, color='#888', va='center', style='italic')

plt.tight_layout()
plt.savefig(str(Path(__file__).parent / 'results_fig_B_ablation.pdf'), bbox_inches='tight', dpi=300)
plt.savefig(str(Path(__file__).parent / 'results_fig_B_ablation.png'), bbox_inches='tight', dpi=200)
plt.close()
print("Saved: results_fig_B_ablation")

# ═══════════════════════════════════════════════════════════════════
# Figure C: EXP5 — Interaction Mechanism + EXP6 — Site Head
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
fig.patch.set_facecolor('white')

# EXP5
names_e5  = ['Elementwise', 'Concat', 'Cross-Attn']
means_e5  = [0.7046, 0.7142, 0.7413]
stds_e5   = [0.0021, 0.0025, 0.0034]
colors_e5 = [GRAY, TEAL, ORANGE]
hbar(axes[0], names_e5, means_e5, stds_e5, colors_e5,
     title='(C) Interaction Mechanism', xlim=(0.68, 0.77),
     highlight_idx=2)

# EXP6
names_e6  = ['Linear', 'Conv1D']
means_e6  = [0.7328, 0.7408]
stds_e6   = [0.0027, 0.0046]
colors_e6 = [GRAY, ORANGE]
hbar(axes[1], names_e6, means_e6, stds_e6, colors_e6,
     title='(D) Site Prediction Head', xlim=(0.71, 0.77),
     highlight_idx=1)

plt.tight_layout()
plt.savefig(str(Path(__file__).parent / 'results_fig_CD_interaction_head.pdf'), bbox_inches='tight', dpi=300)
plt.savefig(str(Path(__file__).parent / 'results_fig_CD_interaction_head.png'), bbox_inches='tight', dpi=200)
plt.close()
print("Saved: results_fig_CD_interaction_head")

# ═══════════════════════════════════════════════════════════════════
# Figure E: EXP2 — Pretraining Strategy (exp2v3 임시값)
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5.5))
fig.patch.set_facecolor('white')

# NTP, CPCL, All은 exp2v3 미완료 → 임시로 placeholder (회색, 점선)
names_e2 = ['No PT', 'MLM', 'NTP*', 'SSP', 'Pairing', 'CPCL*', 'MLM+NTP', 'All*']
means_e2 = [0.7525, 0.7517, 0.748, 0.7571, 0.7607, 0.762,  0.7414, 0.765]
stds_e2  = [0.0022, 0.0007, 0.005, 0.0014, 0.0047, 0.005,  0.0023, 0.005]
# *표시 = 미완료/임시값

available = [True, True, False, True, True, False, True, False]
colors_e2 = []
for a in available:
    colors_e2.append(BLUE if a else GRAY)

y = np.arange(len(names_e2))
bars = ax.barh(y, means_e2, color=colors_e2, edgecolor='white',
               linewidth=0.8, height=0.65, zorder=3)
ax.errorbar(means_e2, y, xerr=stds_e2, fmt='none',
            ecolor='#555', capsize=3, elinewidth=1.5, zorder=4)

# 미완료 항목은 점선 테두리
for i, (avail, bar) in enumerate(zip(available, bars)):
    if not avail:
        bar.set_linestyle('--')
        bar.set_edgecolor(GRAY)
        bar.set_linewidth(2)
        bar.set_alpha(0.5)

# baseline (No PT)
ax.axvline(0.7525, color=RED, linestyle='--', lw=1.5, alpha=0.7)
ax.text(0.7525 + 0.001, len(names_e2) - 0.3, 'No PT baseline',
        color=RED, fontsize=8, va='top', alpha=0.8)

for i, (m, s, avail) in enumerate(zip(means_e2, stds_e2, available)):
    txt = f'{m:.4f}' if avail else f'(TBD)'
    color = DARK if avail else GRAY
    ax.text(m + s + 0.002, i, txt, va='center', fontsize=9.5, color=color)

ax.set_yticks(y)
ax.set_yticklabels(names_e2, fontsize=10.5)
ax.set_xlabel('F1 Score', fontsize=11)
ax.set_title('(E) Pretraining Strategy Comparison\n'
             '(* = TBD: pending exp2v4 results)', fontsize=12, fontweight='bold', pad=8)
ax.set_xlim(0.72, 0.79)
ax.grid(True, axis='x', linestyle='--', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

legend_patches = [
    mpatches.Patch(color=BLUE, label='Result available (exp2v3)'),
    mpatches.Patch(color=GRAY, alpha=0.5, label='Pending (exp2v4)'),
]
ax.legend(handles=legend_patches, loc='lower right', fontsize=9.5,
          framealpha=0.9, edgecolor='#ccc')

plt.tight_layout()
plt.savefig(str(Path(__file__).parent / 'results_fig_E_pretraining.pdf'), bbox_inches='tight', dpi=300)
plt.savefig(str(Path(__file__).parent / 'results_fig_E_pretraining.png'), bbox_inches='tight', dpi=200)
plt.close()
print("Saved: results_fig_E_pretraining")

print("\nAll result figures saved to docs/paper_cmi/")
