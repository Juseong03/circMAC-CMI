"""
Validation set으로 모델별 optimal threshold 계산 및 저장.

학습 시와 동일한 split (split_train_valid, seed=SEED, label_column='label')으로
val set을 재현하여 inference → F1-maximizing global threshold 탐색.

출력: docs/paper_cmi/model_thresholds_s{SEED}.json
  {
    "circmac": 0.712,
    "mamba":   0.834,
    ...
  }

사용법:
  python docs/paper_cmi/compute_thresholds.py --seed 1 --device 0
  python docs/paper_cmi/compute_thresholds.py --seed 1 --device 0 --model_root models_for_viz
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pickle

ROOT_DIR = str(Path(__file__).parent.parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


# ── 모델 경로 헬퍼 ────────────────────────────────────────────────────────────
def get_model_paths(model_root, seed):
    """run_viz_all_models.sh와 동일한 경로 규칙."""
    r = Path(model_root)
    candidates = {
        'circmac':     r / 'circmac'     / f'v2_abl_full_s{seed}'          / str(seed),
        'mamba':       r / 'mamba'        / f'v2_enc_mamba_s{seed}'          / str(seed),
        'lstm':        r / 'lstm'         / f'v2_enc_lstm_s{seed}'           / str(seed),
        'transformer': r / 'transformer'  / f'v2_enc_transformer_s{seed}'    / str(seed),
        'hymba':       r / 'hymba'        / f'v2_enc_hymba_s{seed}'          / str(seed),
        'rnabert':     r / 'rnabert'      / f'exp1_fair_frozen_rnabert_s{seed}' / str(seed),
        'rnaernie':    r / 'rnaernie'     / f'exp1_fair_frozen_rnaernie_s{seed}' / str(seed),
        'rnamsm':      r / 'rnamsm'       / f'exp1_fair_frozen_rnamsm_s{seed}'   / str(seed),
        'rnafm':       r / 'rnafm'        / f'exp1_fair_frozen_rnafm_s{seed}'    / str(seed),
    }
    return {k: str(v) for k, v in candidates.items() if v.is_dir()}


# ── Batch inference ────────────────────────────────────────────────────────────
PRETRAINED_MODELS = ['rnabert', 'rnaernie', 'rnafm', 'rnamsm']
MAX_LEN_MAP = {'rnabert': 438, 'rnaernie': 511, 'rnafm': 1022, 'rnamsm': 1022}


def build_trainer(model_name, model_dir, seed, device_obj):
    import torch
    from trainer import Trainer
    from data import CircRNABindingSitesDataset
    from utils_config import get_model_config

    model_dir_path = Path(model_dir).resolve()
    exp_name = model_dir_path.parent.name

    # vocab_size: checkpoint에서 읽기
    model_pth = model_dir_path / 'train' / 'model.pth'
    if model_pth.exists():
        ckpt = torch.load(str(model_pth), map_location='cpu', weights_only=False)
        emb_key = 'embedding.word_embeddings.weight'
        vocab_size = ckpt[emb_key].shape[0] if emb_key in ckpt else 26
    else:
        vocab_size = 26

    config = get_model_config(model_name, d_model=128, n_layer=6, vocab_size=vocab_size)
    trainer = Trainer(seed=seed, device=device_obj,
                      experiment_name=exp_name, verbose=False)
    trainer.define_model(model_name=model_name, config=config,
                         is_cross_attention=True,
                         interaction='cross_attention',
                         site_head_type='conv1d')
    trainer.set_pretrained_target(target='mirna', rna_model='rnabert')
    trainer.task = 'sites'
    trainer.rc   = False

    if model_name in PRETRAINED_MODELS:
        trainer.define_pretrained_model(model_name)

    if model_pth.exists():
        trainer.load_model_from_path(str(model_pth), verbose=True)
    else:
        trainer.load_model(epoch=None, pretrain=False, verbose=True)
    trainer.model.eval()
    return trainer


def run_inference_on_df(df, model_name, model_dir, seed, device_obj, batch_size=64):
    """val_df 전체에 대해 batch inference → (all_gt, all_pred) 반환."""
    import torch
    from torch.utils.data import DataLoader
    from data import CircRNABindingSitesDataset

    # binding pair만 (gt가 있는 것) → threshold는 positive pair에서 의미 있음
    df_pos = df[df['binding'] == 1].copy()
    if len(df_pos) == 0:
        print(f"  [WARN] No binding pairs in val set for {model_name}")
        return np.array([]), np.array([])

    max_len = MAX_LEN_MAP.get(model_name, 1022)
    trainer = build_trainer(model_name, model_dir, seed, device_obj)

    dataset = CircRNABindingSitesDataset(df_pos, max_len=max_len, k=1, k_target=1)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    all_gt, all_pred = [], []
    total = len(loader)
    with torch.no_grad():
        for i, data in enumerate(loader):
            if (i + 1) % 20 == 0 or (i + 1) == total:
                print(f"    batch {i+1}/{total}", end='\r')
            target, target_mask = trainer.forward_target(data)
            emb, mask = trainer.forward(data)
            if trainer.model.is_cross_attention:
                emb, _ = trainer.forward_cross_attention(emb, target, target_mask)
            pred_logits = trainer.forward_task(emb, target, task='sites')
            pred_prob   = torch.softmax(pred_logits, dim=-1)[..., 1]  # (B, L)

            lengths = data['length']
            labels  = data['sites']  # (B, L) or (B, max_len)

            for b in range(pred_prob.size(0)):
                L = int(lengths[b].item())
                # pred_prob 실제 길이와 맞춤 (max_len 제약으로 짧을 수 있음)
                L = min(L, pred_prob.size(1))
                gt_b   = labels[b, :L].cpu().numpy().astype(int)
                pred_b = pred_prob[b, :L].cpu().numpy()
                # 패딩(-100) 제거
                valid = gt_b >= 0
                all_gt.append(gt_b[valid])
                all_pred.append(pred_b[valid])

    print()
    all_gt   = np.concatenate(all_gt)
    all_pred = np.concatenate(all_pred)
    return all_gt, all_pred


def find_optimal_threshold(gt, pred):
    """F1을 최대화하는 threshold 탐색."""
    from sklearn.metrics import precision_recall_curve
    if gt.sum() == 0 or (1 - gt).sum() == 0:
        return 0.5
    prec, rec, thrs = precision_recall_curve(gt, pred)
    f1s = 2 * prec * rec / (prec + rec + 1e-8)
    best_idx = int(np.argmax(f1s))
    opt_thr  = float(thrs[best_idx]) if best_idx < len(thrs) else 0.5
    best_f1  = float(f1s[best_idx])
    return opt_thr, best_f1


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seed',       type=int, default=1)
    parser.add_argument('--device',     type=int, default=0)
    parser.add_argument('--model_root', type=str, default='models_for_viz')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--out_dir',    type=str, default='docs/paper_cmi')
    args = parser.parse_args()

    from utils import get_device

    device_obj = get_device(args.device)
    print(f"Device: {device_obj}")

    # ── 1. Val set 재현 (학습 시와 동일한 split) ──────────────────────────────
    print(f"\n[Step 1] Loading df_train_final.pkl and splitting (seed={args.seed}) ...")
    df_train_full = pickle.load(open('data/df_train_final.pkl', 'rb'))

    # miRNA 서열 추가 (학습 데이터에는 없을 수 있음)
    seq_csv_path = Path('data/binding_miRNA_seq.csv')
    if seq_csv_path.exists() and 'miRNA' not in df_train_full.columns:
        df_seq = pd.read_csv(seq_csv_path)[['isoform_ID', 'miRNA_ID', 'miRNA']]
        df_train_full = df_train_full.merge(df_seq, on=['isoform_ID', 'miRNA_ID'], how='left')
        df_train_full['miRNA'] = df_train_full['miRNA'].fillna('')

    # split_train_valid: test_size=0.3, 학습 시와 동일
    from sklearn.model_selection import train_test_split
    _, val_df = train_test_split(
        df_train_full,
        test_size=0.3,
        random_state=args.seed,
        stratify=df_train_full['binding']
    )
    print(f"  val set: {len(val_df)} rows  (binding={val_df['binding'].sum()})")

    # ── 2. 모델 경로 확인 ───────────────────────────────────────────────────────
    model_paths = get_model_paths(args.model_root, args.seed)
    print(f"\n[Step 2] Found {len(model_paths)} models: {list(model_paths.keys())}")

    # ── 3. 모델별 inference + threshold 탐색 ──────────────────────────────────
    thresholds = {}
    print()
    for model_name, model_dir in model_paths.items():
        print(f"[{model_name}] inference on val set ({len(val_df)} rows) ...")
        gt, pred = run_inference_on_df(
            val_df, model_name, model_dir,
            seed=args.seed, device_obj=device_obj,
            batch_size=args.batch_size
        )
        if len(gt) == 0:
            print(f"  → skipped (no data)")
            continue

        from sklearn.metrics import roc_auc_score
        auroc = roc_auc_score(gt, pred)
        opt_thr, best_f1 = find_optimal_threshold(gt, pred)
        thresholds[model_name] = round(opt_thr, 4)
        print(f"  → AUROC={auroc:.4f}  opt_thr={opt_thr:.4f}  F1@opt={best_f1:.4f}")

    # ── 4. 저장 ──────────────────────────────────────────────────────────────
    out_path = Path(args.out_dir) / f'model_thresholds_s{args.seed}.json'
    with open(out_path, 'w') as f:
        json.dump(thresholds, f, indent=2)
    print(f"\n[Done] Saved to: {out_path}")
    print(json.dumps(thresholds, indent=2))
