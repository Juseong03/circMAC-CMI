#!/usr/bin/env python3
"""
Fig 5: Interaction Mechanism & Site Head Ablation (EXP5 + EXP6)

두 행(Interaction · Head) × 세 열(F1 / AUROC / AUPRC) = 6패널.
x-tick 라벨은 모두 검정색·보통체.
"""

import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

ROOT  = Path(__file__).resolve().parents[2]
OUT   = Path(__file__).resolve().parent
DATA_CSV = OUT / 'fig5_ablation_int_head_data.csv'

PROPOSED = '#E05C2A'
BASELINE = '#6B9CC7'

INTERACTION = [
    ('Cross-Attn',  True),
    ('Concat',      False),
    ('Elementwise', False),
]
HEAD = [
    ('Conv1D', True),
    ('Linear', False),
]

METRICS = [
    ('f1_macro', 'F1-macro', {'interaction': (0.65, 0.80), 'head': (0.65, 0.80)}),
    ('roc_auc',  'AUROC',    {'interaction': (0.84, 0.92), 'head': (0.84, 0.92)}),
    ('auprc',    'AUPRC',    {'interaction': (0.38, 0.58), 'head': (0.38, 0.58)}),
]

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'xtick.labelsize':10,'ytick.labelsize':9,
                     'pdf.fonttype':42,'ps.fonttype':42})

# ----------------------- 공통 함수 -----------------------
def load_scores(label, ablation):
    """CSV에서 (model, ablation) 기준으로 metric 점수 로드"""
    df = pd.read_csv(DATA_CSV)
    rows = df[(df['model'] == label) & (df['ablation'] == ablation)]
    return {m: rows[m].dropna().tolist() for m in ('f1_macro','roc_auc','auprc')}

# ----------------------- 패널 -----------------------
def plot_panel(ax, entries, metric, ylim, ablation):
    y0,yr=ylim[0],ylim[1]-ylim[0]
    for i,(lbl,is_best) in enumerate(entries):
        vals=load_scores(lbl, ablation)[metric]
        m,std=np.mean(vals),np.std(vals)
        ax.bar(i,m-y0,bottom=y0,width=0.50,
               color=PROPOSED if is_best else BASELINE,
               alpha=0.85,linewidth=0)
        ax.errorbar(i,m,yerr=std,fmt='none',color='#222',
                    capsize=5,capthick=1.4,elinewidth=1.4)
        ax.text(i,m+std+yr*0.03,f'{m:.3f}',
                ha='center',va='bottom',fontsize=9.5,fontweight='bold',
                color='#222',
                bbox=dict(boxstyle='round,pad=0.15',
                          fc='white',ec='none',alpha=0.85))
    ax.set_xticks(range(len(entries)))
    # x-tick 라벨: 모두 검정·보통체
    ax.set_xticklabels([e[0] for e in entries], color='#000000')  # noqa
    ax.set_ylim(*ylim)
    ax.yaxis.grid(True,linestyle='--',alpha=0.4)
    ax.set_axisbelow(True)

# ----------------------- 메인 -----------------------
def main():
    # Figure
    fig,axes=plt.subplots(2,3,figsize=(12,8.0),
                          gridspec_kw={'hspace':0.55,'wspace':0.30})
    fig.suptitle('Interaction & Head Ablation',
                 fontsize=12,fontweight='bold',y=1.01)

    for r,(group_entries,group_key,title_prefix) in enumerate(
        [(INTERACTION,'interaction','(a) Interaction Mechanism'),
         (HEAD,       'head',       '(b) Site Prediction Head')]):
        for c,(metric,metric_lbl,ylims) in enumerate(METRICS):
            ax=axes[r,c]
            plot_panel(ax,group_entries,metric,ylims['interaction' if r==0 else 'head'],group_key)
            # 패널 제목
            ax.set_title(f'{title_prefix}\n{metric_lbl}' if c==0 else metric_lbl,
                         fontsize=11,fontweight='bold')

    # 범례
    fig.legend(handles=[Patch(facecolor=PROPOSED,label='Proposed (ours)'),
                        Patch(facecolor=BASELINE,label='Baseline')],
               loc='upper center',ncol=2,frameon=False,fontsize=9,
               bbox_to_anchor=(0.5,0.0))

    fig.tight_layout(); fig.subplots_adjust(bottom=0.08)

    for ext in ('pdf','png'):
        fig.savefig(OUT/f'fig5_ablation_int_head.{ext}',
                    dpi=200,bbox_inches='tight')
        print('Saved →', OUT/f'fig5_ablation_int_head.{ext}')
    plt.close(fig)

if __name__=='__main__':
    main()