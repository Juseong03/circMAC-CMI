#!/usr/bin/env python3
"""
Create three plot variants for RNA-LM comparison:

 A) toplabel      – group labels above (default)
 B) bottomlabel   – group labels below xticks
 C) finetunedonly – only fine-tuned group + CircMAC
"""

import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- 공통 설정 ----------
ROOT  = Path(__file__).resolve().parents[2]
OUT   = Path(__file__).resolve().parent
DATA_CSV = OUT / 'fig1_rna_lm_data.csv'
(OUT / 'figures_paper/fig1_rna_lm').mkdir(parents=True, exist_ok=True)

RNA_LMS = [
    ('RNABERT',  True),
    ('RNAErnie', True),
    ('RNAMSM',   True),
    ('RNA-FM',   True),
    ('RNABERT',  False),
    ('RNAErnie', False),
    ('RNAMSM',   False),
    ('RNA-FM',   False),
]
LM_COLORS={'RNABERT':'#4878CF','RNAErnie':'#9467BD','RNAMSM':'#2CA02C','RNA-FM':'#17BECF'}
PROP_COLOR, PROP_LABEL = '#E05C2A', 'CircMAC'
FROZEN_HATCH='//'
METRICS=[('f1_macro','F1-macro',(0.46,0.62)),
         ('roc_auc', 'AUROC',   (0.42,0.72)),
         ('auprc',   'AUPRC',   (0.04,0.22))]

import matplotlib as mpl
mpl.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'pdf.fonttype':42,'ps.fonttype':42})

# ---------- 데이터 로딩 ----------
def load_scores_csv(label, frozen):
    """CSV에서 (model, mode) 기준으로 metric 점수 로드"""
    df = pd.read_csv(DATA_CSV)
    mode = 'frozen' if frozen else 'fine-tuned'
    rows = df[(df['model'] == label) & (df['mode'] == mode)]
    return {m: rows[m].dropna().tolist() for m in ('f1_macro','roc_auc','auprc')}

def load_prop_scores():
    df = pd.read_csv(DATA_CSV)
    rows = df[(df['model'] == 'CircMAC') & (df['mode'] == 'proposed')]
    return {m: rows[m].dropna().tolist() for m in ('f1_macro','roc_auc','auprc')}

LM_DATA={(l,f): load_scores_csv(l,f) for l,f in RNA_LMS}
PROP_SCORES=load_prop_scores()

# ---------- 공통 그리기 함수 ----------
def bar_panel(ax,entries,metric,title,ylim):
    y0,yr=ylim[0],ylim[1]-ylim[0]
    for x,lab,col,hatch,vals in entries:
        if not vals: continue
        m,s=np.mean(vals),np.std(vals)
        ax.bar(x,m-y0,bottom=y0,width=0.62,color=col,alpha=0.85,
               hatch=hatch,edgecolor='white' if hatch else col,linewidth=0,zorder=2)
        if len(vals)>1:
            ax.errorbar(x,m,yerr=s,fmt='none',color='#222',capsize=3,capthick=1.3,
                        elinewidth=1.3,zorder=4)
        ax.text(x,m+s+yr*0.022,f'{m:.3f}',ha='center',va='bottom',fontsize=8,
                fontweight='bold',color='#222',
                bbox=dict(boxstyle='round,pad=0.12',fc='white',ec='none',alpha=0.85))
    ax.set_xlim(min(e[0] for e in entries)-0.6,max(e[0] for e in entries)+0.6)
    ax.set_ylim(*ylim)
    ax.set_xticks([e[0] for e in entries])
    ax.set_xticklabels([e[1] for e in entries],rotation=35,ha='right',color='#000')
    ax.set_title(title,fontweight='bold',fontsize=11)
    ax.yaxis.grid(True,linestyle='--',alpha=0.4)
    ax.set_axisbelow(True)

# ---------- 3 버전 생성 ----------
def build_entries(include_frozen, x_start):
    lm_lbls=['RNABERT','RNAErnie','RNAMSM','RNA-FM']
    ent=[]
    x=x_start
    if include_frozen:
        for l in lm_lbls:
            ent.append((x,l,LM_COLORS[l],FROZEN_HATCH,LM_DATA[(l,True)][metric]))
            x+=1
    for l in lm_lbls:
        ent.append((x,l,LM_COLORS[l],None,LM_DATA[(l,False)][metric]))
        x+=1
    ent.append((x,PROP_LABEL,PROP_COLOR,None,PROP_SCORES[metric]))
    return ent

def add_group_labels(ax,top,include_frozen,x_start,ylim):
    txt_kwargs=dict(ha='center',va='top' if top else 'top',fontsize=8.5,color='#555')
    y=ylim[1]-0.01*(ylim[1]-ylim[0]) if top else ylim[0]-0.05
    if include_frozen:
        ax.text(x_start+1.5,y,'Frozen',**txt_kwargs,clip_on=False)
        ax.text(x_start+5.5,y,'Fine-tuned',**txt_kwargs,clip_on=False)
    else:
        ax.text(x_start+1.5,y,'Fine-tuned',**txt_kwargs,clip_on=False)

variants=[
    ('toplabel',      True,  True, 0, True),   # (tag, toplabel?, include_frozen?, x_start, want_label)
    ('bottomlabel',   False, True, 0, True),
    ('finetunedonly', True,  False,0, True),
]

for tag,top,inc_frozen,x0,_ in variants:
    fig,axes=plt.subplots(1,3,figsize=(15,4.6))
    fig.suptitle('RNA Language Model Comparison',fontsize=12,fontweight='bold',y=1.02)
    for ax,(metric,title,ylim) in zip(axes,METRICS):
        entries=build_entries(inc_frozen,x0)
        bar_panel(ax,entries,metric,title,ylim)
        add_group_labels(ax,top,inc_frozen,x0,ylim)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.20)
    for ext in ['pdf','png']:
        out=OUT/f'fig1_rna_lm_{tag}.{ext}'
        fig.savefig(out,dpi=200,bbox_inches='tight'); print('Saved →',out)
    plt.close(fig)