"""
BSJ-Proximal vs BSJ-Distal Binding Site Analysis
=================================================

CircMAC의 circular architecture가 실제로 BSJ(Back-Splice Junction) 근처의
binding site 예측에서 얼마나 효과적인지 정량적으로 검증합니다.

[분석 내용]
  Part 1. 데이터 분포 분석 (모델 불필요)
    - test set에서 BSJ-proximal binding site 비율
    - 위치 분포 히스토그램

  Part 2. 모델별 BSJ-Proximal / BSJ-Distal F1 비교 (모델 필요)
    - CircMAC vs Mamba vs Transformer
    - BSJ-proximal에서 CircMAC 우위를 직접 증명

[실행 예시]
  # Part 1만 (모델 없이)
  python docs/paper_cmi/analyze_bsj.py --data_only

  # Part 1 + 2 (모델 있을 때) — seed 3 기준 (models_for_viz)
  python docs/paper_cmi/analyze_bsj.py \\
    --models circmac mamba hymba lstm transformer rnabert rnaernie rnafm rnamsm \\
    --exps exp1_circmac_s3 exp1_mamba_s3 exp1_hymba_s3 exp1_lstm_s3 exp1_transformer_s3 \\
           exp3_rnabert_frozen_s3 exp3_rnaernie_frozen_s3 exp3_rnafm_frozen_s3 exp3_rnamsm_frozen_s3 \\
    --seeds 3 3 3 3 3 3 3 3 3 \\
    --device 0

[BSJ-Proximal 정의]
  circRNA의 길이를 L이라 할 때, 위치 i의 BSJ 거리:
    d_bsj(i) = min(i, L-i)   (0-indexed)
  binding site 위치 i 가 d_bsj(i) < bsj_window 이면 BSJ-proximal
  한 샘플에서 binding site 중 하나라도 proximal이면 BSJ-proximal sample
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless 서버용 (display 없이 파일 저장)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# project root를 sys.path에 추가
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

# ─── 색상 ─────────────────────────────────────────────────────────────────────
COLORS = {
    'circmac':     '#E67E22',   # 주황
    'mamba':       '#2980B9',   # 파랑
    'transformer': '#27AE60',   # 초록
    'hymba':       '#8E44AD',   # 보라
    'lstm':        '#E74C3C',   # 빨강
    'proximal':    '#C0392B',   # 진빨
    'distal':      '#2E86AB',   # 청록
}
MODEL_LABELS = {
    'circmac':     'CircMAC (ours)',
    'mamba':       'Mamba',
    'transformer': 'Transformer',
    'hymba':       'HyMBA',
    'lstm':        'LSTM',
}

# ─── BSJ 거리 계산 ─────────────────────────────────────────────────────────────
def bsj_distance(pos: int, L: int) -> int:
    """circRNA에서 위치 pos의 BSJ 최소 거리 (0-indexed, L=서열 길이)"""
    return min(pos, L - pos)


def label_bsj_proximity(sites: list, bsj_window: int) -> str:
    """
    샘플의 binding site list를 보고 BSJ-proximal / BSJ-distal 판정.
    sites: 0/1 list of length L
    """
    L = len(sites)
    for i, s in enumerate(sites):
        if s == 1 and bsj_distance(i, L) < bsj_window:
            return 'proximal'
    return 'distal'


# ════════════════════════════════════════════════════════════════════════════
# PART 1: 데이터 분포 분석 (모델 불필요)
# ════════════════════════════════════════════════════════════════════════════

def analyze_data_distribution(df_test: pd.DataFrame, bsj_window: int, out_dir: Path):
    """
    test set의 BSJ-proximal binding site 통계 시각화.
    """
    print("\n" + "="*60)
    print("PART 1: Data Distribution Analysis")
    print("="*60)

    # positive samples only
    df_pos = df_test[df_test['binding'] == 1].copy()
    df_pos['bsj_label'] = df_pos['sites'].apply(
        lambda s: label_bsj_proximity(s, bsj_window))
    df_pos['L'] = df_pos['circRNA'].apply(len)

    n_total   = len(df_pos)
    n_prox    = (df_pos['bsj_label'] == 'proximal').sum()
    n_distal  = (df_pos['bsj_label'] == 'distal').sum()
    print(f"Positive test samples: {n_total}")
    print(f"  BSJ-proximal (window={bsj_window}nt): {n_prox} ({n_prox/n_total*100:.1f}%)")
    print(f"  BSJ-distal:                          {n_distal} ({n_distal/n_total*100:.1f}%)")

    # 위치 분포: 모든 binding site position의 BSJ 거리 히스토그램
    all_bsj_dists = []
    for _, row in df_pos.iterrows():
        sites = row['sites']
        L = len(sites)
        for i, s in enumerate(sites):
            if s == 1:
                all_bsj_dists.append(bsj_distance(i, L))

    all_bsj_dists = np.array(all_bsj_dists)
    print(f"\nBinding site BSJ distance stats:")
    print(f"  Total binding positions: {len(all_bsj_dists)}")
    print(f"  Proximal (<{bsj_window}nt):  {(all_bsj_dists < bsj_window).sum()} "
          f"({(all_bsj_dists < bsj_window).mean()*100:.1f}%)")
    print(f"  Median BSJ distance: {np.median(all_bsj_dists):.0f} nt")

    # ── Figure: 3 subplots ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.patch.set_facecolor('white')

    # (a) Proximal vs Distal pie
    ax = axes[0]
    ax.pie([n_prox, n_distal],
           labels=[f'BSJ-proximal\n(n={n_prox})', f'BSJ-distal\n(n={n_distal})'],
           colors=[COLORS['proximal'], COLORS['distal']],
           autopct='%1.1f%%', startangle=90,
           wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    ax.set_title(f'(a) Sample Distribution\n(BSJ window = {bsj_window} nt)', fontsize=12, fontweight='bold')

    # (b) BSJ distance histogram
    ax = axes[1]
    bins = np.arange(0, 520, 20)
    counts, edges = np.histogram(all_bsj_dists, bins=bins)
    bar_colors = [COLORS['proximal'] if (e < bsj_window) else COLORS['distal']
                  for e in edges[:-1]]
    ax.bar(edges[:-1], counts, width=18, color=bar_colors, edgecolor='white', linewidth=0.5)
    ax.axvline(bsj_window, color='black', linestyle='--', lw=1.5, alpha=0.7,
               label=f'BSJ window = {bsj_window} nt')
    ax.set_xlabel('BSJ Distance (nt)', fontsize=11)
    ax.set_ylabel('Number of binding positions', fontsize=11)
    ax.set_title('(b) Binding Site Distance from BSJ', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # (c) Binding site density by position (normalized, 0=BSJ)
    ax = axes[2]
    # Normalize each sample to [0, 1] by position/L  → 0 and 1 are both BSJ
    all_norm_pos = []
    for _, row in df_pos.iterrows():
        sites = row['sites']
        L = len(sites)
        for i, s in enumerate(sites):
            if s == 1:
                norm = i / L   # 0.0 = BSJ, 0.5 = farthest from BSJ
                all_norm_pos.append(norm)

    all_norm_pos = np.array(all_norm_pos)
    ax.hist(all_norm_pos, bins=50, color=COLORS['distal'], edgecolor='white',
            linewidth=0.5, density=True)
    ax.axvspan(0, bsj_window / 250, color=COLORS['proximal'], alpha=0.15,
               label=f'BSJ-proximal region')
    ax.axvspan(1 - bsj_window / 250, 1, color=COLORS['proximal'], alpha=0.15)
    ax.set_xlabel('Normalized Position (0 & 1 = BSJ)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('(c) Binding Site Position Density', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out_path = out_dir / 'bsj_data_distribution.pdf'
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.savefig(out_dir / 'bsj_data_distribution.png', bbox_inches='tight', dpi=200)
    plt.close()
    print(f"\nSaved: {out_path}")

    return df_pos


# ════════════════════════════════════════════════════════════════════════════
# PART 2: 모델 추론 + BSJ-Group F1 비교
# ════════════════════════════════════════════════════════════════════════════

def build_trainer(model_name: str, exp_name: str, seed: int, device_id: int,
                  d_model: int = 128, n_layer: int = 6,
                  max_len: int = 1022, vocab_size: int = 11, verbose: bool = False):
    """
    Trainer를 초기화하고 best checkpoint를 로드한다.
    epoch=None → saved_models/{model}/{exp}/{seed}/train/model.pth 로드
    vocab_size: dataset.vocab_size에서 가져온 값 (default 11 = k=1 KmerTokenizer)
    """
    from trainer import Trainer
    from utils import get_device
    from utils_config import get_model_config

    device = get_device(device_id)
    config = get_model_config(model_name, d_model=d_model, n_layer=n_layer, vocab_size=vocab_size, verbose=verbose)

    trainer = Trainer(seed=seed, device=device, experiment_name=exp_name, verbose=verbose)
    trainer.define_model(
        model_name=model_name,
        config=config,
        # interaction='cross_attention'이면 is_cross_attention도 반드시 True로 전달해야
        # _set_cross_attention()이 호출되어 trainer.model.is_cross_attention = True 설정됨
        is_cross_attention=True,
        interaction='cross_attention',
        site_head_type='conv1d',
    )
    trainer.set_pretrained_target(target='mirna', rna_model='rnabert')
    trainer.task = 'sites'
    trainer.rc = False  # trainer.forward에서 self.rc 참조 필요

    # frozen pretrained 모델(rnabert/rnaernie/rnafm/rnamsm)은 circRNA 인코딩에
    # trainer.model_pt가 별도로 필요 (trainer.forward에서 pre-embedding에 사용)
    if model_name in ['rnabert', 'rnaernie', 'rnafm', 'rnamsm']:
        trainer.define_pretrained_model(model_name)

    # epoch=None → saved_models/{model}/{exp}/{seed}/train/model.pth (최신 best 모델)
    trainer.load_model(epoch=None, pretrain=False, verbose=verbose)
    trainer.model.eval()
    return trainer


def get_best_epoch(model_name: str, exp_name: str, seed: int,
                   saved_models_dir: str = './saved_models') -> int:
    """
    training.json 또는 저장된 epoch 폴더에서 best epoch 자동 탐지.

    training.json 구조:
        {"train": {...}, "valid": {...}, "final": {"145": {...}}, ...}
        → final의 key가 best_epoch
    """
    # 1) training.json 탐색 (logs/ 또는 saved_models/ 하위)
    json_candidates = [
        # logs/{model}/{exp}/{seed}/training.json (실험 서버에서 복사한 경우)
        Path(f'./logs/{model_name}/{exp_name}/{seed}/training.json'),
        # saved_models/{model}/{exp}/{seed}/train/training.json
        Path(saved_models_dir) / model_name / exp_name / str(seed) / 'train' / 'training.json',
    ]
    for jp in json_candidates:
        if jp.exists():
            data = json.loads(jp.read_text())
            # training.json에서 best_epoch 추출: final 딕셔너리의 첫 번째 key
            if 'final' in data and data['final']:
                epoch_key = list(data['final'].keys())[0]
                return int(epoch_key)

    # 2) saved_models 하위 epoch 폴더에서 유일한 epoch 탐지 (fallback)
    epoch_dir = Path(saved_models_dir) / model_name / exp_name / str(seed) / 'train' / 'epoch'
    if epoch_dir.exists():
        epochs = [int(d.name) for d in epoch_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        if epochs:
            best = max(epochs)
            print(f"  [auto] {model_name}/{exp_name}/{seed}: using epoch {best} (latest found)")
            return best

    raise FileNotFoundError(
        f"Cannot determine best epoch for {model_name}/{exp_name}/{seed}.\n"
        f"  Checked:\n"
        + "\n".join(f"    {p}" for p in json_candidates) +
        f"\n  Either copy training.json to ./logs/{model_name}/{exp_name}/{seed}/ "
        f"or pass --best_epochs manually."
    )


@torch.no_grad()
def run_inference(trainer, test_dataset, batch_size: int = 32):
    """
    Test dataset에 대해 inference를 실행하고 per-sample 결과를 반환.

    Returns:
        list of dicts: [{'pred': np.array[L,2], 'label': np.array[L], 'length': int}, ...]
    """
    from torch.utils.data import DataLoader

    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    trainer.model.eval()

    results = []
    for data in loader:
        lengths  = data['length']
        labels   = data['sites']         # [B, max_L]

        target, target_mask = trainer.forward_target(data)
        emb, mask = trainer.forward(data)
        if trainer.model.is_cross_attention:
            emb, _ = trainer.forward_cross_attention(emb, target, target_mask)

        # forward_task returns [B, max_L-1, 2] logits (CLS token removed)
        pred_logits = trainer.forward_task(emb, target, task='sites')  # [B, L, 2]

        # softmax → binding probability per position
        pred_prob = torch.softmax(pred_logits, dim=-1)[..., 1].cpu().numpy()  # [B, L]
        labels_np = labels[:, 1:].cpu().numpy()   # remove CLS position from labels

        for b in range(len(lengths)):
            L = int(lengths[b].item())
            results.append({
                'pred':   pred_prob[b, :L],     # [L]
                'label':  labels_np[b, :L],     # [L]
                'length': L,
            })

    return results


def compute_group_f1(results: list, group_mask: list, threshold: float = 0.5) -> dict:
    """
    results 중 group_mask가 True인 샘플들의 F1 계산.

    Returns:
        {'f1': float, 'precision': float, 'recall': float, 'n_samples': int}
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    all_pred, all_label = [], []
    for r, include in zip(results, group_mask):
        if include:
            all_pred.append(r['pred'])
            all_label.append(r['label'])

    if not all_pred:
        return {'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'n_samples': 0}

    pred_flat  = np.concatenate(all_pred)
    label_flat = np.concatenate(all_label)

    # ignore padding (-100)
    valid = label_flat >= 0
    pred_flat  = pred_flat[valid]
    label_flat = label_flat[valid].astype(int)

    pred_bin = (pred_flat >= threshold).astype(int)

    return {
        'f1':        f1_score(label_flat, pred_bin, average='macro', zero_division=0),
        'precision': precision_score(label_flat, pred_bin, average='macro', zero_division=0),
        'recall':    recall_score(label_flat, pred_bin, average='macro', zero_division=0),
        'n_samples': int(sum(group_mask)),
    }


# ── span 관련 유틸 ─────────────────────────────────────────────────────────────

def extract_spans(seq: np.ndarray) -> list:
    """연속된 1의 구간 → [(start, end), ...] 반환 (0-indexed, end inclusive)"""
    spans, start = [], None
    for i, v in enumerate(seq):
        if v == 1 and start is None:
            start = i
        elif v != 1 and start is not None:
            spans.append((start, i - 1))
            start = None
    if start is not None:
        spans.append((start, len(seq) - 1))
    return spans


def span_is_proximal(span: tuple, L: int, bsj_window: int) -> bool:
    """span (start, end) 내 어느 위치라도 BSJ 거리 < bsj_window 이면 proximal"""
    for pos in range(span[0], span[1] + 1):
        if bsj_distance(pos, L) < bsj_window:
            return True
    return False


def compute_span_metrics(results: list, group_mask: list,
                         bsj_window: int, threshold: float = 0.5) -> dict:
    """
    Span-level (site-level) 평가 — overlap 기준 (strict=False).

    group_mask가 True인 샘플에 대해:
      - 전체 span F1 / precision / recall
      - BSJ-proximal span recall  (GT span 중 proximal한 것만)
      - BSJ-distal   span recall  (GT span 중 distal한 것만)

    Returns dict with keys:
        span_f1, span_precision, span_recall,
        prox_span_recall, dist_span_recall,
        n_gt_spans, n_prox_spans, n_dist_spans, n_samples
    """
    TP = FP = FN = 0
    prox_TP = prox_FN = 0
    dist_TP = dist_FN = 0

    for r, include in zip(results, group_mask):
        if not include:
            continue
        L = r['length']
        label_bin = (r['label'] >= 0.5).astype(int)
        pred_bin  = (r['pred']  >= threshold).astype(int)

        gt_spans   = extract_spans(label_bin)
        pred_spans = extract_spans(pred_bin)

        # overlap matching (strict=False)
        matched_pred = set()
        for gt in gt_spans:
            hit = False
            for j, pred in enumerate(pred_spans):
                if max(gt[0], pred[0]) <= min(gt[1], pred[1]):   # overlap
                    hit = True
                    matched_pred.add(j)
                    break
            if hit:
                TP += 1
            else:
                FN += 1

            # BSJ proximal/distal recall
            if span_is_proximal(gt, L, bsj_window):
                if hit: prox_TP += 1
                else:   prox_FN += 1
            else:
                if hit: dist_TP += 1
                else:   dist_FN += 1

        FP += len(pred_spans) - len(matched_pred)

    precision = TP / (TP + FP + 1e-8)
    recall    = TP / (TP + FN + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    prox_recall = prox_TP / (prox_TP + prox_FN + 1e-8)
    dist_recall = dist_TP / (dist_TP + dist_FN + 1e-8)

    return {
        'span_f1':          f1,
        'span_precision':   precision,
        'span_recall':      recall,
        'prox_span_recall': prox_recall,
        'dist_span_recall': dist_recall,
        'n_gt_spans':   TP + FN,
        'n_prox_spans': prox_TP + prox_FN,
        'n_dist_spans': dist_TP + dist_FN,
        'n_samples':    int(sum(group_mask)),
    }


def find_best_threshold(results: list, group_mask: list) -> float:
    """Validation 없이 test set에서 threshold sweep (분석용)"""
    from sklearn.metrics import f1_score
    all_pred, all_label = [], []
    for r, include in zip(results, group_mask):
        if include:
            all_pred.append(r['pred'])
            all_label.append(r['label'])
    pred_flat  = np.concatenate(all_pred)
    label_flat = np.concatenate(all_label)
    valid = label_flat >= 0
    pred_flat = pred_flat[valid]
    label_flat = label_flat[valid].astype(int)

    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.1, 0.9, 17):
        f1 = f1_score(label_flat, (pred_flat >= t).astype(int), average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def run_model_analysis(df_pos: pd.DataFrame, test_dataset,
                       model_cfgs: list, bsj_window: int,
                       batch_size: int, out_dir: Path,
                       args_bsj_window: int = None):
    """
    여러 모델에 대해 BSJ-proximal / BSJ-distal F1을 계산하고 시각화.

    model_cfgs: [{'name': 'circmac', 'exp': '...', 'seed': 1, 'device': 0}, ...]
    """
    print("\n" + "="*60)
    print("PART 2: Model Inference + BSJ-Group F1")
    print("="*60)

    # BSJ 그룹 레이블 (positive test samples 순서 = df_pos 순서)
    prox_mask  = [row['bsj_label'] == 'proximal' for _, row in df_pos.iterrows()]
    distal_mask = [row['bsj_label'] == 'distal'   for _, row in df_pos.iterrows()]

    if args_bsj_window is None:
        args_bsj_window = bsj_window

    print(f"BSJ-proximal samples: {sum(prox_mask)}")
    print(f"BSJ-distal samples:   {sum(distal_mask)}")

    results_table = []

    # pretrained 모델 max_len (multimolecule 모델별 position embedding 한계)
    PRETRAINED_MAX_LEN = {
        'rnabert': 438, 'rnaernie': 511, 'rnafm': 1022, 'rnamsm': 1022
    }

    for cfg in model_cfgs:
        mname = cfg['name']
        print(f"\n[{mname}] Loading model: {cfg['exp']} seed={cfg['seed']}")
        try:
            # pretrained 모델은 max_len이 다를 수 있어 별도 dataset으로 필터링
            model_max_len = PRETRAINED_MAX_LEN.get(mname, test_dataset.max_len_circrna)
            if model_max_len < test_dataset.max_len_circrna:
                from data import CircRNABindingSitesDataset
                import numpy as np
                # test_dataset.df의 positional index 기반으로 필터링
                keep_mask_arr = test_dataset.df['circRNA'].str.len() <= model_max_len
                keep_indices  = keep_mask_arr[keep_mask_arr].index.tolist()
                df_filtered   = test_dataset.df.loc[keep_indices].reset_index(drop=True)
                cur_dataset   = CircRNABindingSitesDataset(df_filtered, max_len=model_max_len, k=1, k_target=1)
                # prox/distal mask: test_dataset.df 순서 = df_pos 순서 (둘 다 positive only, reset index)
                cur_prox_mask   = [prox_mask[i]  for i in keep_indices]
                cur_distal_mask = [distal_mask[i] for i in keep_indices]
                cur_df_pos = df_pos.iloc[keep_indices].reset_index(drop=True)
                print(f"  (filtered to max_len={model_max_len}: {len(cur_df_pos)} positive samples)")
            else:
                cur_dataset     = test_dataset
                cur_df_pos      = df_pos
                cur_prox_mask   = prox_mask
                cur_distal_mask = distal_mask

            trainer = build_trainer(
                model_name=mname,
                exp_name=cfg['exp'],
                seed=cfg['seed'],
                device_id=cfg['device'],
                vocab_size=cur_dataset.vocab_size,
                verbose=False,
            )
            inference_results = run_inference(trainer, cur_dataset, batch_size=batch_size)
            assert len(inference_results) == len(cur_df_pos), \
                f"Mismatch: {len(inference_results)} predictions vs {len(cur_df_pos)} samples"

            # threshold sweep on all samples
            all_mask_cur = [True] * len(inference_results)
            thresh = find_best_threshold(inference_results, all_mask_cur)
            print(f"  Best threshold: {thresh:.2f}")

            overall  = compute_group_f1(inference_results, all_mask_cur,    threshold=thresh)
            proximal = compute_group_f1(inference_results, cur_prox_mask,   threshold=thresh)
            distal   = compute_group_f1(inference_results, cur_distal_mask, threshold=thresh)

            # ── Span-level metrics ──────────────────────────────────────────
            span_all = compute_span_metrics(
                inference_results, all_mask_cur, args_bsj_window, threshold=thresh)

            results_table.append({
                'model':     mname,
                'label':     MODEL_LABELS.get(mname, mname),
                # nt-level
                'overall':   overall['f1'],
                'proximal':  proximal['f1'],
                'distal':    distal['f1'],
                'prox_n':    proximal['n_samples'],
                'dist_n':    distal['n_samples'],
                # span-level
                'span_f1':          span_all['span_f1'],
                'span_precision':   span_all['span_precision'],
                'span_recall':      span_all['span_recall'],
                'prox_span_recall': span_all['prox_span_recall'],
                'dist_span_recall': span_all['dist_span_recall'],
                'n_gt_spans':   span_all['n_gt_spans'],
                'n_prox_spans': span_all['n_prox_spans'],
                'n_dist_spans': span_all['n_dist_spans'],
            })
            print(f"  Overall F1:          {overall['f1']:.4f}")
            print(f"  BSJ-proximal F1:     {proximal['f1']:.4f} (n={proximal['n_samples']})")
            print(f"  BSJ-distal F1:       {distal['f1']:.4f}  (n={distal['n_samples']})")
            print(f"  Span F1:             {span_all['span_f1']:.4f}  "
                  f"(prec={span_all['span_precision']:.4f}, rec={span_all['span_recall']:.4f})")
            print(f"  Prox span recall:    {span_all['prox_span_recall']:.4f}  "
                  f"(n_spans={span_all['n_prox_spans']})")
            print(f"  Dist span recall:    {span_all['dist_span_recall']:.4f}  "
                  f"(n_spans={span_all['n_dist_spans']})")

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    if not results_table:
        print("No results to plot.")
        return

    # ── 결과 저장 (CSV) ──
    df_results = pd.DataFrame(results_table)
    df_results.to_csv(out_dir / 'bsj_analysis_results.csv', index=False)
    print(f"\nResults saved: {out_dir / 'bsj_analysis_results.csv'}")
    print(df_results.to_string(index=False))

    # ── Figure: BSJ-Proximal vs Distal 비교 (nt-level) ──
    plot_bsj_comparison(df_results, bsj_window, out_dir)

    # ── Figure: Span-level BSJ recall 비교 ──
    plot_span_bsj_comparison(df_results, bsj_window, out_dir)


def plot_bsj_comparison(df_results: pd.DataFrame, bsj_window: int, out_dir: Path):
    """
    그룹별 F1 비교 bar chart 생성.
    Main figure: BSJ-proximal vs BSJ-distal 성능 격차
    """
    n_models = len(df_results)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('white')

    # ── (a) Grouped bar: proximal vs distal per model ──
    ax = axes[0]
    x = np.arange(n_models)
    w = 0.32
    bar_colors = [COLORS.get(m, '#999') for m in df_results['model']]

    bars_prox = ax.bar(x - w/2, df_results['proximal'], width=w,
                       color=bar_colors, alpha=0.95, edgecolor='white', linewidth=0.8,
                       label='BSJ-proximal')
    bars_dist = ax.bar(x + w/2, df_results['distal'],   width=w,
                       color=bar_colors, alpha=0.45, edgecolor='white', linewidth=0.8,
                       label='BSJ-distal')

    # 값 레이블
    for bar in bars_prox:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8.5)
    for bar in bars_dist:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(df_results['label'], fontsize=10)
    ax.set_ylabel('Macro F1 Score', fontsize=11)
    ax.set_title(f'(a) BSJ-Proximal vs Distal F1\n(window = {bsj_window} nt)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, framealpha=0.9)
    ax.set_ylim(0, min(1.0, df_results[['proximal', 'distal']].max().max() * 1.15))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # ── (b) Δ(proximal - distal): circular design 효과 ──
    ax = axes[1]
    delta = df_results['proximal'] - df_results['distal']
    bar_colors2 = [COLORS.get(m, '#999') for m in df_results['model']]
    bars = ax.bar(x, delta, color=bar_colors2, edgecolor='white', linewidth=0.8)

    # CircMAC 강조
    if 'circmac' in df_results['model'].values:
        cm_idx = df_results['model'].tolist().index('circmac')
        bars[cm_idx].set_edgecolor('#F39C12')
        bars[cm_idx].set_linewidth(2.5)

    ax.axhline(0, color='black', linewidth=1.0)
    for i, (bar, d) in enumerate(zip(bars, delta)):
        sign = '+' if d >= 0 else ''
        ax.text(bar.get_x() + bar.get_width()/2,
                d + (0.002 if d >= 0 else -0.005),
                f'{sign}{d:.3f}', ha='center',
                va='bottom' if d >= 0 else 'top', fontsize=9.5)

    ax.set_xticks(x)
    ax.set_xticklabels(df_results['label'], fontsize=10)
    ax.set_ylabel('ΔF1 (Proximal − Distal)', fontsize=11)
    ax.set_title('(b) Advantage at BSJ-Proximal Sites\n(positive = better on BSJ region)',
                 fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    out_pdf = out_dir / f'bsj_analysis_comparison_w{bsj_window}.pdf'
    plt.savefig(out_pdf, bbox_inches='tight', dpi=300)
    plt.savefig(out_dir / f'bsj_analysis_comparison_w{bsj_window}.png', bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Saved: {out_pdf}")

    # ── Figure 2: Overall / Proximal / Distal 3개 나란히 ──
    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor('white')
    x = np.arange(n_models)
    w = 0.25
    bar_colors3 = [COLORS.get(m, '#999') for m in df_results['model']]

    ax.bar(x - w, df_results['overall'],  width=w, color=bar_colors3, alpha=0.6,
           edgecolor='white', label='Overall')
    ax.bar(x,     df_results['proximal'], width=w, color=bar_colors3, alpha=1.0,
           edgecolor='white', label='BSJ-proximal')
    ax.bar(x + w, df_results['distal'],   width=w, color=bar_colors3, alpha=0.35,
           edgecolor='white', label='BSJ-distal')

    ax.set_xticks(x)
    ax.set_xticklabels(df_results['label'], fontsize=10)
    ax.set_ylabel('Macro F1 Score', fontsize=11)
    ax.set_title(f'Overall / BSJ-Proximal / BSJ-Distal F1 (window={bsj_window}nt)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    out_pdf2 = out_dir / f'bsj_analysis_three_groups_w{bsj_window}.pdf'
    plt.savefig(out_pdf2, bbox_inches='tight', dpi=300)
    plt.savefig(out_dir / f'bsj_analysis_three_groups_w{bsj_window}.png', bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Saved: {out_pdf2}")


def plot_span_bsj_comparison(df_results: pd.DataFrame, bsj_window: int, out_dir: Path):
    """
    Site(span)-level BSJ-proximal vs distal recall 비교 figure.

    Panel (a): grouped bar — proximal recall vs distal recall per model
    Panel (b): Δrecall (proximal − distal) — circular architecture 효과
    Panel (c): Span F1 / Precision / Recall overview
    """
    # span columns이 없으면 skip (data_only mode)
    if 'span_f1' not in df_results.columns:
        return

    n_models = len(df_results)
    x = np.arange(n_models)
    bar_colors = [COLORS.get(m, '#999') for m in df_results['model']]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor('white')

    # ── (a) Proximal vs Distal span recall ──
    ax = axes[0]
    w = 0.32
    bars_prox = ax.bar(x - w/2, df_results['prox_span_recall'], width=w,
                       color=bar_colors, alpha=0.95, edgecolor='white', linewidth=0.8,
                       label='BSJ-proximal sites')
    bars_dist = ax.bar(x + w/2, df_results['dist_span_recall'], width=w,
                       color=bar_colors, alpha=0.40, edgecolor='white', linewidth=0.8,
                       label='BSJ-distal sites')
    for bar in bars_prox:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    for bar in bars_dist:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

    n_prox = df_results['n_prox_spans'].iloc[0] if 'n_prox_spans' in df_results.columns else '?'
    n_dist = df_results['n_dist_spans'].iloc[0] if 'n_dist_spans' in df_results.columns else '?'
    ax.set_xticks(x)
    ax.set_xticklabels(df_results['label'], fontsize=10, rotation=15, ha='right')
    ax.set_ylabel('Site Recall (overlap-based)', fontsize=11)
    ax.set_title(f'(a) BSJ-Proximal vs Distal Site Recall\n'
                 f'window={bsj_window}nt  |  prox sites={n_prox}, dist sites={n_dist}',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, framealpha=0.9)
    ax.set_ylim(0, 1.12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # ── (b) Δ recall (proximal − distal) ──
    ax = axes[1]
    delta = df_results['prox_span_recall'] - df_results['dist_span_recall']
    bars = ax.bar(x, delta, color=bar_colors, edgecolor='white', linewidth=0.8)
    if 'circmac' in df_results['model'].values:
        cm_idx = df_results['model'].tolist().index('circmac')
        bars[cm_idx].set_edgecolor('#F39C12')
        bars[cm_idx].set_linewidth(2.5)
    ax.axhline(0, color='black', linewidth=1.0)
    for bar, d in zip(bars, delta):
        sign = '+' if d >= 0 else ''
        ax.text(bar.get_x() + bar.get_width()/2,
                d + (0.005 if d >= 0 else -0.008),
                f'{sign}{d:.3f}', ha='center',
                va='bottom' if d >= 0 else 'top', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(df_results['label'], fontsize=10, rotation=15, ha='right')
    ax.set_ylabel('Δ Site Recall (Proximal − Distal)', fontsize=11)
    ax.set_title('(b) BSJ-Proximal Advantage\n(site-level, positive = better on BSJ)',
                 fontsize=11, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # ── (c) Span F1 / Precision / Recall overview ──
    ax = axes[2]
    w3 = 0.25
    ax.bar(x - w3, df_results['span_f1'],        width=w3, color=bar_colors,
           alpha=1.0, edgecolor='white', label='Span F1')
    ax.bar(x,      df_results['span_precision'],  width=w3, color=bar_colors,
           alpha=0.6, edgecolor='white', label='Precision')
    ax.bar(x + w3, df_results['span_recall'],     width=w3, color=bar_colors,
           alpha=0.35, edgecolor='white', label='Recall')
    ax.set_xticks(x)
    ax.set_xticklabels(df_results['label'], fontsize=10, rotation=15, ha='right')
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('(c) Overall Span-level Metrics\n(overlap-based)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    out_pdf = out_dir / f'bsj_span_comparison_w{bsj_window}.pdf'
    plt.savefig(out_pdf, bbox_inches='tight', dpi=300)
    plt.savefig(out_dir / f'bsj_span_comparison_w{bsj_window}.png', bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Saved: {out_pdf}")


# ════════════════════════════════════════════════════════════════════════════
# PART 3: CNN Circular Padding 효과 분석 (no_circular_cnn_pad ablation 보완)
# ════════════════════════════════════════════════════════════════════════════

def analyze_position_level(df_pos: pd.DataFrame, bsj_window: int, out_dir: Path):
    """
    샘플 레벨이 아닌 position 레벨 분석:
    각 position이 BSJ-proximal인지 여부에 따라 분류.
    (데이터 분포 분석, 모델 불필요)
    """
    print("\n" + "="*60)
    print("PART 3: Position-level BSJ analysis")
    print("="*60)

    rows = []
    for _, row in df_pos.iterrows():
        sites = row['sites']
        L = len(sites)
        for i, s in enumerate(sites):
            d = bsj_distance(i, L)
            rows.append({'label': int(s), 'bsj_dist': d,
                         'is_proximal': d < bsj_window})

    df_pos_level = pd.DataFrame(rows)

    # BSJ 거리별 binding site 비율
    bins = list(range(0, 510, 10))
    df_pos_level['dist_bin'] = pd.cut(df_pos_level['bsj_dist'], bins=bins, right=False)
    site_rate = df_pos_level.groupby('dist_bin')['label'].mean()

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('white')
    bin_centers = [(b.left + b.right) / 2 for b in site_rate.index]
    bar_cols = [COLORS['proximal'] if c < bsj_window else COLORS['distal']
                for c in bin_centers]
    ax.bar(bin_centers, site_rate.values * 100, width=9,
           color=bar_cols, edgecolor='white', linewidth=0.5)
    ax.axvline(bsj_window, color='black', linestyle='--', lw=1.5,
               label=f'BSJ window = {bsj_window} nt')
    ax.set_xlabel('BSJ Distance (nt)', fontsize=11)
    ax.set_ylabel('Binding Site Rate (%)', fontsize=11)
    ax.set_title('Binding Site Rate by BSJ Distance\n'
                 '(higher near BSJ → circular architecture matters more)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    prox_rate  = df_pos_level[df_pos_level['is_proximal']]['label'].mean() * 100
    distal_rate = df_pos_level[~df_pos_level['is_proximal']]['label'].mean() * 100
    ax.text(0.98, 0.95,
            f'Proximal avg: {prox_rate:.1f}%\nDistal avg: {distal_rate:.1f}%',
            transform=ax.transAxes, ha='right', va='top', fontsize=9.5,
            bbox=dict(boxstyle='round', facecolor='#f8f8f8', edgecolor='#ccc', alpha=0.8))

    plt.tight_layout()
    out_path = out_dir / f'bsj_position_level_rate_w{bsj_window}.pdf'
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.savefig(out_dir / f'bsj_position_level_rate_w{bsj_window}.png', bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Proximal binding site rate:  {prox_rate:.1f}%")
    print(f"Distal binding site rate:    {distal_rate:.1f}%")
    print(f"Saved: {out_path}")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='BSJ-Proximal Analysis')
    parser.add_argument('--data_only',  action='store_true',
                        help='Part 1 & 3 only (no model inference)')
    parser.add_argument('--data_path',  type=str,
                        default=str(ROOT / 'data' / 'df_test_final.pkl'),
                        help='Path to test dataframe pickle')
    parser.add_argument('--bsj_window', type=int, default=50,
                        help='BSJ proximity window in nt (default: 50)')
    # 모델 지정 (여러 개)
    parser.add_argument('--models',  nargs='+', default=['circmac', 'mamba', 'transformer'],
                        help='Model names to compare')
    parser.add_argument('--exps',    nargs='+',
                        default=['exp1_circmac_s1', 'exp1_mamba_s1', 'exp1_transformer_s1'],
                        help='Experiment names (same order as --models)')
    parser.add_argument('--seeds',   nargs='+', type=int, default=[1, 1, 1],
                        help='Seeds (same order as --models)')
    parser.add_argument('--device',  type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_len', type=int, default=1022)
    parser.add_argument('--out_dir', type=str,
                        default=str(ROOT / 'docs' / 'paper_cmi'),
                        help='Output directory for figures')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 데이터 로드 ──
    print(f"Loading test data: {args.data_path}")
    df_test = pd.read_pickle(args.data_path)
    df_test['length'] = df_test['circRNA'].apply(len)
    df_test = df_test[df_test['length'] <= args.max_len]
    print(f"Test samples: {len(df_test)} (after length filter ≤ {args.max_len})")

    # ── Part 1: 데이터 분포 ──
    df_pos = analyze_data_distribution(df_test, args.bsj_window, out_dir)

    # ── Part 3: position-level 분석 ──
    analyze_position_level(df_pos, args.bsj_window, out_dir)

    if args.data_only:
        print("\nData-only mode. Skipping model inference.")
        return

    # ── Part 2: 모델 추론 ──
    assert len(args.models) == len(args.exps) == len(args.seeds), \
        "--models, --exps, --seeds must have same length"

    model_cfgs = [
        {'name': m, 'exp': e, 'seed': s, 'device': args.device}
        for m, e, s in zip(args.models, args.exps, args.seeds)
    ]

    # test dataset 준비 (positive only, sites task)
    from utils import prepare_datasets
    from utils import seed_everything
    seed_everything(42)

    df_train_dummy = df_pos.iloc[:10].copy()   # dummy train (not used)
    df_pos_reset = df_pos.reset_index(drop=True)

    from data import CircRNABindingSitesDataset
    test_dataset = CircRNABindingSitesDataset(
        df_pos_reset, max_len=args.max_len + 2, target_type='mirna', k=1)

    run_model_analysis(df_pos, test_dataset, model_cfgs,
                       args.bsj_window, args.batch_size, out_dir,
                       args_bsj_window=args.bsj_window)

    print("\n=== All done! ===")
    print(f"Figures saved to: {out_dir}")


if __name__ == '__main__':
    main()
