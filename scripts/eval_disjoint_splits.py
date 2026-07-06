#!/usr/bin/env python3
"""
eval_disjoint_splits.py

CircMAC (v2_abl_full) 모델을 세 가지 test split에서 평가:
  1. Standard    : df_test_final.pkl       (원래 pair-disjoint)
  2. Iso-disjoint: df_test_iso_disjoint.pkl (isoform-disjoint)
  3. BSJ-disjoint: df_test_bsj_disjoint.pkl (BSJ-disjoint)

Output:
  eval_results/disjoint_comparison.csv   — seed별 결과
  eval_results/disjoint_summary.csv      — 평균±std 요약
  eval_results/disjoint_comparison.txt   — 논문용 텍스트 테이블

Usage:
    python scripts/eval_disjoint_splits.py --device 0
    python scripts/eval_disjoint_splits.py --device 0 --seeds 1 2 3
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trainer import Trainer
from utils import prepare_datasets
from utils_config import get_model_config

SAVED    = ROOT / "saved_models"
OUT      = ROOT / "eval_results"
OUT.mkdir(parents=True, exist_ok=True)

MAX_LEN  = 1022
D_MODEL  = 128
N_LAYER  = 6
BS       = 32
WORKERS  = 4

MODEL_NAME  = "circmac"
EXP_TPL     = "v2_abl_full"
INTERACTION = "cross_attention"


# ── Metrics ────────────────────────────────────────────────────────────────────
def compute_metrics(labels, probs):
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        f1_score, precision_score, recall_score,
        accuracy_score, matthews_corrcoef,
    )
    n_total  = len(labels)
    n_pos    = int(labels.sum())
    pos_rate = round(n_pos / n_total, 6) if n_total > 0 else float("nan")

    try:
        auroc = float(roc_auc_score(labels, probs))
        auprc = float(average_precision_score(labels, probs))
    except Exception:
        auroc = auprc = float("nan")

    best_t, best_f1mac = 0.5, -1.0
    for t in np.linspace(0.1, 0.9, 17):
        pb = (probs >= t).astype(int)
        fm = float(f1_score(labels, pb, average="macro", zero_division=0))
        if fm > best_f1mac:
            best_f1mac, best_t = fm, t

    pb = (probs >= best_t).astype(int)
    return {
        "n_tokens":  n_total,
        "n_pos":     n_pos,
        "pos_rate":  pos_rate,
        "auroc":     round(auroc, 6),
        "auprc":     round(auprc, 6),
        "threshold": round(float(best_t), 4),
        "acc":       round(float(accuracy_score(labels, pb)), 6),
        "f1_macro":  round(float(f1_score(labels, pb, average="macro",   zero_division=0)), 6),
        "f1_pos":    round(float(f1_score(labels, pb, pos_label=1,       zero_division=0)), 6),
        "prec_pos":  round(float(precision_score(labels, pb, pos_label=1, zero_division=0)), 6),
        "rec_pos":   round(float(recall_score(labels, pb, pos_label=1,   zero_division=0)), 6),
        "mcc":       round(float(matthews_corrcoef(labels, pb)), 6),
    }


# ── Dataset builder ────────────────────────────────────────────────────────────
def build_test_dataset(df_train_raw, df_test_raw, seed):
    """seed 기반 train/val split (비율만), test는 df_test_raw 사용."""
    df_tr = df_train_raw[df_train_raw["length"] <= MAX_LEN].reset_index(drop=True)
    df_te = df_test_raw[df_test_raw["length"]   <= MAX_LEN].reset_index(drop=True)
    _, _, test_ds, _ = prepare_datasets(
        df=df_tr, df_test=df_te,
        max_len=MAX_LEN + 2, target="mirna", seed=seed, kmer=1,
    )
    return test_ds


def _get_ckpt_vocab_size(model_path):
    ckpt  = torch.load(str(model_path), map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    key   = "embedding.word_embeddings.weight"
    if key in state:
        return state[key].shape[0]
    for k, v in state.items():
        if "embedding" in k and "weight" in k and v.ndim == 2 and v.shape[-1] == D_MODEL:
            return v.shape[0]
    return None


def build_trainer(exp, seed, model_path, device):
    trainer = Trainer(seed=seed, device=device,
                      experiment_name=exp, verbose=False)

    ckpt_vocab = _get_ckpt_vocab_size(model_path)
    config = get_model_config(
        model_name=MODEL_NAME, d_model=D_MODEL, n_layer=N_LAYER,
        verbose=False, rc=False,
        **({} if ckpt_vocab is None else {"vocab_size": ckpt_vocab}),
    )
    trainer.define_model(
        config=config, model_name=MODEL_NAME, pretrain=False,
        pooling_mode_target="mean", is_convblock=True,
        is_cross_attention=True, interaction=INTERACTION,
        use_unified_head=False, binding_pooling="mean",
        site_head_type="conv1d",
    )
    trainer.set_pretrained_target(target="mirna", rna_model="rnabert")
    trainer.task = "sites"
    trainer.site_class_weights = None
    trainer.alpha = 0.5
    trainer.beta  = 0.5
    trainer.rc    = False
    trainer.load_model_from_path(str(model_path), verbose=False)
    return trainer


def extract_preds(tensors):
    preds_raw  = tensors["preds_sites"]
    labels_raw = tensors["labels_sites"]
    labels_aligned = labels_raw[:, 1:]
    N, Lm1 = labels_aligned.shape
    if preds_raw.ndim == 3 and preds_raw.shape[-1] == 2:
        probs_2d = torch.softmax(preds_raw.float(), dim=-1)[:, :, 1]
    else:
        probs_2d = preds_raw.float().squeeze(-1)
    labels_flat = labels_aligned.reshape(-1).numpy()
    probs_flat  = probs_2d.reshape(-1).numpy()
    valid = labels_flat != -100
    return labels_flat[valid].astype(np.int8), probs_flat[valid].astype(np.float32)


def evaluate_on_split(trainer, df_train_raw, df_test_raw, seed, split_name, device):
    """df_test_raw로 test_ds 생성 후 평가."""
    test_ds = build_test_dataset(df_train_raw, df_test_raw, seed)
    trainer.set_dataloader(test_ds, part=2, batch_size=BS,
                           num_workers=WORKERS, shuffle=False)
    try:
        _, tensors, _ = trainer.step_loader(
            trainer.test_loader, 0, is_train=False, data_type="Test"
        )
    except Exception as e:
        print(f"    [ERROR] {split_name}: {e}")
        return None

    if not isinstance(tensors.get("preds_sites"), torch.Tensor):
        return None

    labels, probs = extract_preds(tensors)
    return compute_metrics(labels, probs)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=list(range(1, 11)))
    parser.add_argument("--only_positive", action="store_true",
                        help="Evaluate only on binding==1 pairs (site task)")
    args = parser.parse_args()

    # ── 데이터 로드 ────────────────────────────────────────────────────────────
    def load_df(name):
        path = ROOT / "data" / name
        df = pd.read_pickle(path)
        df["length"] = df["circRNA"].apply(len)
        if args.only_positive or True:   # site task: positive pairs only
            df = df[df["binding"] == 1].reset_index(drop=True)
        return df

    print("Loading datasets...")
    df_train_raw = load_df("df_train_final.pkl")
    splits = {
        "standard":     load_df("df_test_final.pkl"),
        "iso_disjoint": load_df("df_test_iso_disjoint.pkl"),
        "bsj_disjoint": load_df("df_test_bsj_disjoint.pkl"),
    }
    for name, df in splits.items():
        print(f"  {name:<16}: {len(df):,} pairs  "
              f"(len<={MAX_LEN}: {(df['length']<=MAX_LEN).sum():,})")

    rows = []

    for seed in args.seeds:
        exp        = f"{EXP_TPL}_s{seed}"
        model_path = SAVED / MODEL_NAME / exp / str(seed) / "train" / "model.pth"

        if not model_path.exists():
            print(f"\n[SKIP] seed={seed} — checkpoint not found: {model_path}")
            continue

        print(f"\n[seed={seed}] loading {exp} ...")
        try:
            trainer = build_trainer(exp, seed, model_path, args.device)
        except Exception as e:
            print(f"  [ERROR] build_trainer: {e}")
            continue

        for split_name, df_test_raw in splits.items():
            print(f"  evaluating on {split_name} ...")
            metrics = evaluate_on_split(
                trainer, df_train_raw, df_test_raw, seed, split_name, args.device
            )
            if metrics is None:
                print(f"    [SKIP] {split_name} — no metrics")
                continue

            print(f"    AUROC={metrics['auroc']:.4f}  AUPRC={metrics['auprc']:.4f}  "
                  f"F1mac={metrics['f1_macro']:.4f}  F1pos={metrics['f1_pos']:.4f}  "
                  f"MCC={metrics['mcc']:.4f}  n={metrics['n_tokens']:,}")

            rows.append({
                "exp_tpl":   EXP_TPL,
                "seed":      seed,
                "split":     split_name,
                **metrics,
            })

        del trainer
        torch.cuda.empty_cache()

    if not rows:
        print("\nNo results to save.")
        return

    # ── 저장 ───────────────────────────────────────────────────────────────────
    df_raw = pd.DataFrame(rows)
    raw_path = OUT / "disjoint_comparison.csv"
    df_raw.to_csv(raw_path, index=False)
    print(f"\nSaved raw results: {raw_path}")

    # 요약 (평균±std per split)
    metric_cols = ["auroc", "auprc", "f1_macro", "f1_pos", "prec_pos",
                   "rec_pos", "mcc", "acc"]
    summary_rows = []
    for split_name, grp in df_raw.groupby("split"):
        row = {"split": split_name, "n_seeds": len(grp)}
        for m in metric_cols:
            row[f"{m}_mean"] = round(grp[m].mean(), 4)
            row[f"{m}_std"]  = round(grp[m].std(),  4)
        summary_rows.append(row)
    df_sum = pd.DataFrame(summary_rows)
    sum_path = OUT / "disjoint_summary.csv"
    df_sum.to_csv(sum_path, index=False)

    # ── 논문용 텍스트 테이블 ────────────────────────────────────────────────────
    lines = []
    def tee(s):
        print(s); lines.append(s)

    tee("\n" + "="*72)
    tee("  CircMAC Disjoint Split Comparison  (v2_abl_full, sites task)")
    tee("="*72)
    tee(f"\n  Seeds evaluated: {sorted(df_raw['seed'].unique().tolist())}")

    headers = ["Split", "AUROC", "AUPRC", "F1(mac)", "F1(pos)", "MCC"]
    tee(f"\n  {headers[0]:<18}  " + "  ".join(f"{h:>10}" for h in headers[1:]))
    tee("  " + "-"*66)

    split_order = ["standard", "iso_disjoint", "bsj_disjoint"]
    for split_name in split_order:
        if split_name not in df_sum["split"].values:
            continue
        r = df_sum[df_sum["split"] == split_name].iloc[0]
        n = int(r["n_seeds"])
        tee(f"  {split_name:<18}  "
            f"  {r['auroc_mean']:.4f}±{r['auroc_std']:.4f}"
            f"  {r['auprc_mean']:.4f}±{r['auprc_std']:.4f}"
            f"  {r['f1_macro_mean']:.4f}±{r['f1_macro_std']:.4f}"
            f"  {r['f1_pos_mean']:.4f}±{r['f1_pos_std']:.4f}"
            f"  {r['mcc_mean']:.4f}±{r['mcc_std']:.4f}"
            f"  (n={n})")

    tee(f"\n  Saved: {raw_path}")
    tee(f"         {sum_path}")

    (OUT / "disjoint_comparison.txt").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
