#!/usr/bin/env python3
"""
Fig 4: CircMAC Module Ablation Study (EXP4)
  - 세 지표: F1-macro / AUROC / AUPRC  (mean ± std)
  - 막대 색상: CircMAC = 주황, Remove-branch = 파랑, Single-branch = 초록
  - x-tick 라벨: 전부 검정, 굵게 처리 없음
"""

import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

ROOT  = Path(__file__).resolve().parents[2]
OUT   = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)
DATA_CSV = OUT / 'fig4_ablation_modules_data.csv'

MODELS = [
    ('CircMAC',    'full'),
    ('w/o Attn',   'remove'),
    ('w/o Conv',   'remove'),
    ('w/o Mamba',  'remove'),
    ('Mamba Only', 'single'),
    ('CNN Only',   'single'),
    ('Attn Only',  'single'),
]

COLORS = {'full':'#E05C2A', 'remove':'#4E9AC7', 'single':'#8DC8A0'}

METRICS = [
    ('f1_macro', '(a) F1-macro', (0.60, 0.78)),
    ('roc_auc',  '(b) AUROC',    (0.78, 0.93)),
    ('auprc',    '(c) AUPRC',    (0.22, 0.57)),
]

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'xtick.labelsize':9,'ytick.labelsize':9,
                     'pdf.fonttype':42,'ps.fonttype':42})

# ------------------- 데이터 로드 -------------------
def load_scores(label):
    """CSV에서 label(model) 기준으로 metric 점수 로드"""
    df = pd.read_csv(DATA_CSV)
    rows = df[df['model'] == label]
    return {m: rows[m].dropna().tolist() for m in ('f1_macro','roc_auc','auprc')}

# ------------------- 패널 그리기 -------------------
def plot_panel(ax, data, metric, title, ylim):
    y0,yr=ylim[0],ylim[1]-ylim[0]; bar_w=0.58
    for i,(lbl,grp,sc) in enumerate(data):
        vals=sc[metric]; m,std=np.mean(vals),np.std(vals)
        ax.bar(i,m-y0,bottom=y0,width=bar_w,
               color=COLORS[grp],alpha=0.85,linewidth=0)
        ax.errorbar(i,m,yerr=std,fmt='none',color='#222',
                    capsize=4,capthick=1.3,elinewidth=1.3)
        ax.text(i,m+std+yr*0.025,f'{m:.3f}',ha='center',va='bottom',
                fontsize=8.5,fontweight='bold',color='#222',
                bbox=dict(boxstyle='round,pad=0.15',
                fc='white',ec='none',alpha=0.85))
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels([d[0] for d in data], rotation=25, ha='right',
                       color='#000000')      # 전부 검정, 굵게 없음
    ax.set_ylim(*ylim)
    ax.yaxis.grid(True,linestyle='--',alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_title(title,fontweight='bold',fontsize=11)

# ------------------- 메인 -------------------
def main():
    data=[(l,g,load_scores(l)) for l,g in MODELS]

    fig,axes=plt.subplots(1,3,figsize=(14,4.4))
    fig.suptitle('CircMAC Module Ablation',
                 fontsize=12,fontweight='bold',y=1.01)
    for ax,(metric,title,ylim) in zip(axes,METRICS):
        plot_panel(ax,data,metric,title,ylim)

    fig.legend(handles=[
        Patch(facecolor=COLORS['full'],  label='CircMAC (full)'),
        Patch(facecolor=COLORS['remove'],label='Remove branch'),
        Patch(facecolor=COLORS['single'],label='Single branch')],
        loc='upper center', ncol=3, frameon=False, fontsize=9,
        bbox_to_anchor=(0.5,0.0))

    fig.tight_layout(); fig.subplots_adjust(bottom=0.20)
    for ext in ('pdf','png'):
        fig.savefig(OUT/f'fig4_ablation_modules.{ext}',
                    dpi=200,bbox_inches='tight')
        print('Saved →', OUT/f'fig4_ablation_modules.{ext}')
    plt.close(fig)

if __name__=='__main__':
    main()