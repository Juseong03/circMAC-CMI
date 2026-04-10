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
import pandas as pd
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

# ── Model Inference ───────────────────────────────────────────────────────────
_trainer_cache = {}   # model_dir → trainer (재사용으로 속도 향상)


def _build_trainer(model_dir):
    """Trainer를 빌드하고 캐싱. 같은 model_dir은 한 번만 로드."""
    if model_dir in _trainer_cache:
        return _trainer_cache[model_dir]

    ROOT_DIR = str(Path(__file__).parent.parent.parent)
    sys.path.insert(0, ROOT_DIR)

    import torch
    from trainer import Trainer
    from data import CircRNABindingSitesDataset
    from utils import get_device
    from utils_config import get_model_config

    model_dir_path = Path(model_dir).resolve()
    seed       = int(model_dir_path.name)
    exp_name   = model_dir_path.parent.name
    model_name = model_dir_path.parent.parent.name

    # vocab_size 확인용 임시 dataset (1행)
    tmp_df = pd.DataFrame([{'circRNA': 'AUCG', 'miRNA': 'AUCG',
                             'sites': [0, 0, 0, 0], 'binding': 0,
                             'length': 4, 'n_binding_site': 0,
                             'ratio_binding_site': 0.0, 'label': 0,
                             'isoform_ID': '', 'miRNA_ID': ''}])
    from data import CircRNABindingSitesDataset
    dataset = CircRNABindingSitesDataset(tmp_df, max_len=1022, k=1, k_target=1)

    device_obj = get_device(0)
    config = get_model_config(model_name, d_model=128, n_layer=6,
                              vocab_size=dataset.vocab_size)
    trainer = Trainer(seed=seed, device=device_obj,
                      experiment_name=exp_name, verbose=False)
    trainer.define_model(model_name=model_name, config=config,
                         is_cross_attention=True,
                         interaction='cross_attention',
                         site_head_type='conv1d')
    trainer.set_pretrained_target(target='mirna', rna_model='rnabert')
    trainer.task = 'sites'
    trainer.rc   = False
    trainer.load_model(epoch=None, pretrain=False, verbose=True)
    trainer.model.eval()

    _trainer_cache[model_dir] = trainer
    return trainer


def get_predictions(row, model_dir=None, device='cuda'):
    """
    model_dir이 있으면 실제 inference, 없으면 GT 기반 pseudo-prediction.
    """
    sites = np.array(row['sites'], dtype=float)
    if model_dir is None:
        from scipy.ndimage import gaussian_filter1d
        noise = np.random.RandomState(42).randn(len(sites)) * 0.08
        pseudo = np.clip(sites * 0.92 + 0.05 + noise, 0, 1)
        return np.clip(gaussian_filter1d(pseudo, sigma=1.5), 0, 1)

    # ROOT_DIR을 sys.path에 먼저 추가해야 data, trainer 등 import 가능
    ROOT_DIR = str(Path(__file__).parent.parent.parent)
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)

    import torch
    from torch.utils.data import DataLoader
    from data import CircRNABindingSitesDataset

    trainer = _build_trainer(model_dir)
    df_single = pd.DataFrame([row])
    dataset   = CircRNABindingSitesDataset(df_single, max_len=1022, k=1, k_target=1)
    loader    = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    with torch.no_grad():
        for data in loader:
            target, target_mask = trainer.forward_target(data)
            emb, mask = trainer.forward(data)
            if trainer.model.is_cross_attention:
                emb, _ = trainer.forward_cross_attention(emb, target, target_mask)
            pred_logits = trainer.forward_task(emb, target, task='sites')
            pred_prob   = torch.softmax(pred_logits, dim=-1)[..., 1]
            L = int(data['length'][0].item())
            return pred_prob[0, :L].cpu().numpy()


def get_all_predictions(row, model_dirs: dict, with_pred=True):
    """
    model_dirs: {'circmac': 'saved_models/circmac/exp/1', 'mamba': ..., ...}
    반환: {'circmac': np.array, 'mamba': np.array, ...}
    """
    results = {}
    for label, mdir in model_dirs.items():
        pred = get_predictions(row, mdir if with_pred else None)
        results[label] = pred
    return results

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
def select_cases(df, circ_id=None, mirna_id=None):
    """
    circ_id / mirna_id가 주어지면 해당 row들을 케이스로 사용.
    없으면 기본 케이스 3개(iloc 기반) 사용.
    """
    if circ_id is not None or mirna_id is not None:
        mask = pd.Series([True] * len(df), index=df.index)
        if circ_id is not None:
            mask &= df['isoform_ID'].astype(str).str.contains(str(circ_id), case=False, regex=False)
        if mirna_id is not None:
            mask &= df['miRNA_ID'].astype(str) == str(mirna_id)
        matched = df[mask]
        if len(matched) == 0:
            raise ValueError(f"No rows found for circ_id={circ_id}, mirna_id={mirna_id}")
        print(f"Found {len(matched)} matching row(s).")
        cases = []
        for i, (_, row) in enumerate(matched.iterrows()):
            iso = str(row.get('isoform_ID', '?'))
            iso_short = iso if len(iso) <= 30 else iso[:27] + '...'
            label = f"isoform: {iso_short}\nmiRNA: {row.get('miRNA_ID', '?')}"
            cases.append((row, label))
        return cases

    # 기본 케이스
    row_a = df.iloc[766]
    row_b = df.iloc[32]
    row_c = df[(df['binding'] == 1) & (df['length'].between(150, 200))].iloc[5]
    return [
        (row_a, "Case A: Binding site at 3' end\n(adjacent to BSJ in circular form)"),
        (row_b, "Case B: Binding site at 5' end\n(adjacent to BSJ in circular form)"),
        (row_c, "Case C: Binding site in the middle\n(not near BSJ)"),
    ]


def parse_model_dirs(model_dirs_arg):
    """
    --model_dirs 파싱: 'label:path label:path ...' 형식
    예) 'circmac:./saved_models/circmac/exp4_full_s3/3 mamba:./saved_models/mamba/exp1_mamba_s3/3'
    """
    model_dirs = {}
    if not model_dirs_arg:
        return model_dirs
    for item in model_dirs_arg:
        if ':' in item:
            label, path = item.split(':', 1)
            model_dirs[label.strip()] = path.strip()
        else:
            # label 없으면 경로에서 자동 추출: saved_models/{model}/{exp}/{seed}
            p = Path(item)
            label = p.parent.parent.name
            model_dirs[label] = item
    return model_dirs


def _sanitize(s):
    import re
    return re.sub(r'[|,\s/\\:*?"<>]', '_', s)


# 모델별 색상
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
DEFAULT_COLORS = ['#E67E22', '#2980B9', '#27AE60', '#8E44AD', '#16A085',
                  '#C0392B', '#F39C12', '#95A5A6']


def main(with_pred=False, model_dirs=None, data_path=None, circ_id=None, mirna_id=None):
    """
    model_dirs: dict {'label': 'path', ...}  — 여러 모델 비교
    """
    np.random.seed(42)
    df = load_data(data_path)
    model_dirs = model_dirs or {}

    cases = select_cases(df, circ_id=circ_id, mirna_id=mirna_id)
    n_cases  = len(cases)
    n_models = max(len(model_dirs), 1)

    # ── Figure 레이아웃 ───────────────────────────────────────────────────────
    # Row 0: 원형 다이어그램 (케이스별)
    # Row 1~N: 모델별 선형 비교 (모델이 여러 개면 행 추가)
    n_cols     = max(n_cases, 3)
    n_pred_rows = n_models if (with_pred and model_dirs) else 1

    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(6 * n_cols, 5 * (1 + n_pred_rows) + 2))
    fig.patch.set_facecolor('white')
    gs = GridSpec(1 + n_pred_rows, n_cols, figure=fig,
                  hspace=0.55, wspace=0.30,
                  top=0.93, bottom=0.07, left=0.05, right=0.97)

    # ── Row 0: Circular diagrams ─────────────────────────────────────────────
    all_preds = []   # cases × models 예측값 캐싱
    for col, (row, ttl) in enumerate(cases):
        ax    = fig.add_subplot(gs[0, col])
        seq   = row['circRNA']
        sites = np.array(row['sites'])

        # 첫 번째 모델(또는 데모) 예측을 원형 링에 표시
        first_dir  = next(iter(model_dirs.values()), None) if model_dirs else None
        pred_circ  = get_predictions(row, first_dir if with_pred else None)
        draw_circular_binding(ax, seq, sites, title=ttl,
                              pred=pred_circ if with_pred else None)

        # 모든 모델 예측 수집
        preds = {}
        if with_pred and model_dirs:
            for label, mdir in model_dirs.items():
                preds[label] = get_predictions(row, mdir)
        all_preds.append(preds)

    # ── Row 1~: 모델별 선형 비교 (케이스 앞 2개) ─────────────────────────────
    if with_pred and model_dirs:
        model_list = list(model_dirs.keys())
        for m_idx, m_label in enumerate(model_list):
            color = MODEL_COLORS.get(m_label,
                                     DEFAULT_COLORS[m_idx % len(DEFAULT_COLORS)])
            for col, (row, ttl) in enumerate(cases[:min(n_cases, n_cols - 1)]):
                ax    = fig.add_subplot(gs[1 + m_idx, col])
                seq   = row['circRNA']
                sites = np.array(row['sites'])
                pred  = all_preds[col].get(m_label,
                            get_predictions(row, model_dirs[m_label]))
                draw_linear_heatmap(ax, seq, sites, pred,
                                    title=f'[{m_label}] {ttl.split(chr(10))[0]}')
    else:
        # 단일 모델 or 데모
        for col, (row, ttl) in enumerate(cases[:min(n_cases, n_cols - 1)]):
            ax    = fig.add_subplot(gs[1, col])
            seq   = row['circRNA']
            sites = np.array(row['sites'])
            pred_c = get_predictions(row, None)
            draw_linear_heatmap(ax, seq, sites, pred_c,
                                title=f'Linear view: {ttl.split(chr(10))[0]}')

    # ── 마지막 열 마지막 행: BSJ statistics ──────────────────────────────────
    ax = fig.add_subplot(gs[-1, n_cols - 1])
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    bsj_wins = [5, 10, 15, 22, 30]
    ratios = []
    pos_df = df[df['binding'] == 1]
    for w in bsj_wins:
        cnt = sum(1 for _, r in pos_df.iterrows()
                  if any(r['sites'][:w]) or any(r['sites'][-w:]))
        ratios.append(cnt / len(pos_df) * 100)
    bars = ax.bar(range(len(bsj_wins)), ratios,
                  color=['#AED6F1', '#5DADE2', '#2980B9', '#1A5276', '#154360'],
                  edgecolor='white', linewidth=1.0)
    for bar, r in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{r:.1f}%', ha='center', fontsize=9, fontweight='bold',
                color='#2C3E50')
    ax.set_xticks(range(len(bsj_wins)))
    ax.set_xticklabels([f'±{w}nt' for w in bsj_wins], fontsize=9)
    ax.set_xlabel('BSJ window size', fontsize=10)
    ax.set_ylabel('Fraction of positive pairs (%)', fontsize=10)
    ax.set_title('Binding sites near BSJ\n(test set positive pairs)',
                 fontsize=10, fontweight='bold', pad=6)
    ax.set_ylim(0, max(ratios) * 1.25)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # ── 범례 ─────────────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(color=BIND_COLOR,   label='Binding site (GT)'),
        mpatches.Patch(color=NOBIND_COLOR, label='Non-binding'),
        mpatches.Patch(color=BSJ_COLOR,    label='BSJ'),
    ]
    if with_pred and model_dirs:
        for i, label in enumerate(model_dirs):
            c = MODEL_COLORS.get(label, DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
            legend_items.append(mpatches.Patch(color=c, label=label))
    fig.legend(handles=legend_items, loc='lower center', ncol=len(legend_items),
               fontsize=9.5, framealpha=0.9, edgecolor='#ccc',
               bbox_to_anchor=(0.5, 0.01))

    model_tag = f' [{", ".join(model_dirs.keys())}]' if model_dirs else ''
    fig.suptitle(f'circRNA–miRNA Binding Site Visualization{model_tag}',
                 fontsize=14, fontweight='bold', y=0.97, color='#2C3E50')

    id_tag = f'_{_sanitize(circ_id)}' if circ_id else ''
    suffix = '_with_pred' if with_pred else '_gt_only'
    out_dir = Path(__file__).parent
    out_pdf = out_dir / f'binding_visualization{id_tag}{suffix}.pdf'
    out_png = out_dir / f'binding_visualization{id_tag}{suffix}.png'
    plt.savefig(out_pdf, bbox_inches='tight', dpi=300)
    plt.savefig(out_png, bbox_inches='tight', dpi=200)
    print(f'Saved: {out_pdf}')
    print(f'Saved: {out_png}')
    plt.close()

    # ── Raw CSV 저장 (모델별 컬럼) ────────────────────────────────────────────
    import csv
    out_csv = out_dir / f'binding_visualization{id_tag}{suffix}.csv'
    model_labels = list(model_dirs.keys()) if model_dirs else ['pred']
    header = ['isoform_ID', 'miRNA_ID', 'length', 'position', 'nucleotide',
              'ground_truth', 'bsj_adjacent'] + [f'pred_{m}' for m in model_labels]
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for (row, ttl), preds in zip(cases, all_preds):
            iso   = row.get('isoform_ID', '')
            mirna = row.get('miRNA_ID', '')
            seq   = row['circRNA']
            sites = np.array(row['sites'])
            L     = len(seq)
            w     = 30
            # 모델 예측값 (없으면 GT pseudo)
            pred_vals = {}
            if with_pred and model_dirs:
                pred_vals = preds
            else:
                pred_vals = {'pred': get_predictions(row, None)}
            for i in range(L):
                bsj_adj = int(i < w or i >= L - w)
                row_data = [iso, mirna, L, i, seq[i], int(sites[i]), bsj_adj]
                for m in model_labels:
                    p = pred_vals.get(m, pred_vals.get('pred',
                            np.zeros(L)))[i] if pred_vals else 0.0
                    row_data.append(round(float(p), 6))
                writer.writerow(row_data)
    print(f'Saved: {out_csv}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
Binding site visualization with multi-model comparison.

Examples:
  # Ground truth only (demo)
  python plot_binding_visualization.py

  # Single model
  python plot_binding_visualization.py --with_pred \\
      --model_dirs circmac:./saved_models/circmac/exp4_full_s3/3

  # Multi-model comparison
  python plot_binding_visualization.py --with_pred \\
      --model_dirs circmac:./saved_models/circmac/exp4_full_s3/3 \\
                   mamba:./saved_models/mamba/exp1_mamba_s3/3 \\
                   lstm:./saved_models/lstm/exp1_lstm_s3/3

  # Specific circRNA
  python plot_binding_visualization.py --with_pred --circ_id "chr4|84678168" \\
      --model_dirs circmac:./saved_models/circmac/exp4_full_s3/3 \\
                   mamba:./saved_models/mamba/exp1_mamba_s3/3
""")
    parser.add_argument('--with_pred', action='store_true',
                        help='Run model inference')
    parser.add_argument('--model_dirs', type=str, nargs='+', default=None,
                        help='Model dirs as "label:path" pairs. '
                             'e.g. circmac:./saved_models/circmac/exp4_full_s3/3')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to df_test_final.pkl')
    parser.add_argument('--circ_id', type=str, default=None,
                        help='isoform_ID substring (e.g. "chr4|84678168")')
    parser.add_argument('--mirna_id', type=str, default=None,
                        help='miRNA_ID to filter (e.g. hsa-miR-21-5p)')
    args = parser.parse_args()

    model_dirs = parse_model_dirs(args.model_dirs)
    main(with_pred=args.with_pred, model_dirs=model_dirs,
         data_path=args.data_path, circ_id=args.circ_id, mirna_id=args.mirna_id)
