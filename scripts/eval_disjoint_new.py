#!/usr/bin/env python3
"""
eval_disjoint_new.py

새로 학습된 iso/BSJ-disjoint 모델들을 평가합니다.
(eval_disjoint_splits.py 는 기존 pair-disjoint 모델을 새 test에 적용하는 것,
 이 스크립트는 새 split으로 학습된 모델을 해당 test set에서 평가합니다.)

Models evaluated:
  iso_circmac_s{seed}  → df_test_iso_disjoint.pkl
  iso_hymba_s{seed}    → df_test_iso_disjoint.pkl
  iso_mamba_s{seed}    → df_test_iso_disjoint.pkl
  bsj_circmac_s{seed}  → df_test_bsj_disjoint.pkl
  bsj_hymba_s{seed}    → df_test_bsj_disjoint.pkl
  bsj_mamba_s{seed}    → df_test_bsj_disjoint.pkl

Output:
  eval_results/disjoint_new_raw.csv     — per-seed results
  eval_results/disjoint_new_summary.csv — mean±std
  eval_results/disjoint_new_table.txt   — paper table

Usage:
    python scripts/eval_disjoint_new.py --device 0
    python scripts/eval_disjoint_new.py --device 0 --seeds 1 2 3
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

SAVED   = ROOT / "saved_models"
OUT     = ROOT / "eval_results"
OUT.mkdir(parents=True, exist_ok=True)

MAX_LEN = 1022
D_MODEL = 128
N_LAYER = 6
BS      = 32
WORKERS = 4

# (split_prefix, model_name, exp_prefix, label, test_file)
EXPERIMENTS = [
    # ── Iso-disjoint ──────────────────────────────────────────────────────────
    ("iso", "circmac", "iso_pt_nopt",        "CircMAC (NoPT)",      "df_test_iso_disjoint.pkl"),
    ("iso", "circmac", "iso_pt_mlm",         "CircMAC (MLM)",       "df_test_iso_disjoint.pkl"),
    ("iso", "circmac", "iso_pt_ntp",         "CircMAC (NTP)",       "df_test_iso_disjoint.pkl"),
    ("iso", "circmac", "iso_pt_ssp",         "CircMAC (SSP)",       "df_test_iso_disjoint.pkl"),
    ("iso", "circmac", "iso_pt_pairing",     "CircMAC (Pairing)",   "df_test_iso_disjoint.pkl"),
    ("iso", "circmac", "iso_pt_mlm_ntp",     "CircMAC (MLM+NTP)",   "df_test_iso_disjoint.pkl"),
    ("iso", "circmac", "iso_pt_mlm_ssp",     "CircMAC (MLM+SSP)",   "df_test_iso_disjoint.pkl"),
    ("iso", "circmac", "iso_pt_mlm_pairing", "CircMAC (MLM+Pair)",  "df_test_iso_disjoint.pkl"),
    ("iso", "circmac", "iso_pt_ssp_pairing", "CircMAC (SSP+Pair)",  "df_test_iso_disjoint.pkl"),
    ("iso", "circmac", "iso_pt_mlm_ntp_ssp", "CircMAC (M+N+S)",     "df_test_iso_disjoint.pkl"),
    ("iso", "circmac", "iso_pt_all",         "CircMAC (All)",       "df_test_iso_disjoint.pkl"),
    ("iso", "hymba",       "iso_hymba",       "Hymba",               "df_test_iso_disjoint.pkl"),
    ("iso", "mamba",       "iso_mamba",       "Mamba",               "df_test_iso_disjoint.pkl"),
    ("iso", "lstm",        "iso_lstm",        "LSTM",                "df_test_iso_disjoint.pkl"),
    ("iso", "transformer", "iso_transformer", "Transformer",         "df_test_iso_disjoint.pkl"),
    ("iso", "rnabert",     "iso_rnabert_ft",  "RNABERT (ft)",        "df_test_iso_disjoint.pkl"),
    ("iso", "rnaernie",    "iso_rnaernie_ft", "RNAErnie (ft)",       "df_test_iso_disjoint.pkl"),
    ("iso", "rnamsm",      "iso_rnamsm_ft",   "RNAMSM (ft)",         "df_test_iso_disjoint.pkl"),
    ("iso", "rnafm",       "iso_rnafm_ft",    "RNA-FM (ft)",         "df_test_iso_disjoint.pkl"),
    # ── BSJ-disjoint ──────────────────────────────────────────────────────────
    ("bsj", "circmac", "bsj_pt_nopt",        "CircMAC (NoPT)",      "df_test_bsj_disjoint.pkl"),
    ("bsj", "circmac", "bsj_pt_mlm",         "CircMAC (MLM)",       "df_test_bsj_disjoint.pkl"),
    ("bsj", "circmac", "bsj_pt_ntp",         "CircMAC (NTP)",       "df_test_bsj_disjoint.pkl"),
    ("bsj", "circmac", "bsj_pt_ssp",         "CircMAC (SSP)",       "df_test_bsj_disjoint.pkl"),
    ("bsj", "circmac", "bsj_pt_pairing",     "CircMAC (Pairing)",   "df_test_bsj_disjoint.pkl"),
    ("bsj", "circmac", "bsj_pt_mlm_ntp",     "CircMAC (MLM+NTP)",   "df_test_bsj_disjoint.pkl"),
    ("bsj", "circmac", "bsj_pt_mlm_ssp",     "CircMAC (MLM+SSP)",   "df_test_bsj_disjoint.pkl"),
    ("bsj", "circmac", "bsj_pt_mlm_pairing", "CircMAC (MLM+Pair)",  "df_test_bsj_disjoint.pkl"),
    ("bsj", "circmac", "bsj_pt_ssp_pairing", "CircMAC (SSP+Pair)",  "df_test_bsj_disjoint.pkl"),
    ("bsj", "circmac", "bsj_pt_mlm_ntp_ssp", "CircMAC (M+N+S)",     "df_test_bsj_disjoint.pkl"),
    ("bsj", "circmac", "bsj_pt_all",         "CircMAC (All)",       "df_test_bsj_disjoint.pkl"),
    ("bsj", "hymba",       "bsj_hymba",       "Hymba",               "df_test_bsj_disjoint.pkl"),
    ("bsj", "mamba",       "bsj_mamba",       "Mamba",               "df_test_bsj_disjoint.pkl"),
    ("bsj", "lstm",        "bsj_lstm",        "LSTM",                "df_test_bsj_disjoint.pkl"),
    ("bsj", "transformer", "bsj_transformer", "Transformer",         "df_test_bsj_disjoint.pkl"),
    ("bsj", "rnabert",     "bsj_rnabert_ft",  "RNABERT (ft)",        "df_test_bsj_disjoint.pkl"),
    ("bsj", "rnaernie",    "bsj_rnaernie_ft", "RNAErnie (ft)",       "df_test_bsj_disjoint.pkl"),
    ("bsj", "rnamsm",      "bsj_rnamsm_ft",   "RNAMSM (ft)",         "df_test_bsj_disjoint.pkl"),
    ("bsj", "rnafm",       "bsj_rnafm_ft",    "RNA-FM (ft)",         "df_test_bsj_disjoint.pkl"),
]


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


LM_MODELS = {"rnabert", "rnaernie", "rnamsm", "rnafm"}


def build_trainer(model_name, exp, seed, model_path, device):
    trainer = Trainer(seed=seed, device=device,
                      experiment_name=exp, verbose=False)
    ckpt_vocab = _get_ckpt_vocab_size(model_path)
    config = get_model_config(
        model_name=model_name, d_model=D_MODEL, n_layer=N_LAYER,
        verbose=False, rc=False,
        **({} if ckpt_vocab is None else {"vocab_size": ckpt_vocab}),
    )
    if model_name in LM_MODELS:
        config.trainable = True
    trainer.define_model(
        config=config, model_name=model_name, pretrain=False,
        pooling_mode_target="mean", is_convblock=True,
        is_cross_attention=True, interaction="cross_attention",
        use_unified_head=False, binding_pooling="mean",
        site_head_type="conv1d",
    )
    if model_name in LM_MODELS:
        trainer.define_pretrained_model(model_name=model_name)
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
    if preds_raw.ndim == 3 and preds_raw.shape[-1] == 2:
        probs_2d = torch.softmax(preds_raw.float(), dim=-1)[:, :, 1]
    else:
        probs_2d = preds_raw.float().squeeze(-1)
    labels_flat = labels_aligned.reshape(-1).numpy()
    probs_flat  = probs_2d.reshape(-1).numpy()
    valid = labels_flat != -100
    return labels_flat[valid].astype(np.int8), probs_flat[valid].astype(np.float32)


def evaluate(trainer, df_train_raw, df_test_raw, seed, device, bs=BS):
    df_tr = df_train_raw[df_train_raw["length"] <= MAX_LEN].reset_index(drop=True)
    df_te = df_test_raw[df_test_raw["length"]   <= MAX_LEN].reset_index(drop=True)
    _, _, test_ds, _ = prepare_datasets(
        df=df_tr, df_test=df_te,
        max_len=MAX_LEN + 2, target="mirna", seed=seed, kmer=1,
    )
    trainer.set_dataloader(test_ds, part=2, batch_size=bs,
                           num_workers=WORKERS, shuffle=False)
    try:
        _, tensors, _ = trainer.step_loader(
            trainer.test_loader, 0, is_train=False, data_type="Test"
        )
    except Exception as e:
        print(f"    [ERROR] inference: {e}")
        return None
    if not isinstance(tensors.get("preds_sites"), torch.Tensor):
        return None
    labels, probs = extract_preds(tensors)
    return compute_metrics(labels, probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[1, 2, 3])
    args = parser.parse_args()

    # Load train ref (pair-disjoint train — for dataset split ratio)
    # Note: we use the SAME train file as used during training for each experiment
    train_files = {
        "iso": ROOT / "data/df_train_iso_disjoint.pkl",
        "bsj": ROOT / "data/df_train_bsj_disjoint.pkl",
    }
    test_files = {
        "iso": ROOT / "data/df_test_iso_disjoint.pkl",
        "bsj": ROOT / "data/df_test_bsj_disjoint.pkl",
    }

    # Cache dataframes
    dfs = {}
    for key in ("iso", "bsj"):
        df_train = pd.read_pickle(train_files[key])
        df_test  = pd.read_pickle(test_files[key])
        df_train["length"] = df_train["circRNA"].apply(len)
        df_test["length"]  = df_test["circRNA"].apply(len)
        # sites task: positive pairs only
        df_train = df_train[df_train["binding"] == 1].reset_index(drop=True)
        df_test  = df_test[df_test["binding"]   == 1].reset_index(drop=True)
        dfs[key] = (df_train, df_test)
        print(f"  {key}: train={len(df_train):,}  test={len(df_test):,}")

    rows = []

    LM_MODELS = {"rnabert", "rnaernie", "rnamsm", "rnafm"}

    for split_pfx, model_name, exp_pfx, label, _ in EXPERIMENTS:
        print(f"\n=== {split_pfx.upper()} / {label} ===")
        df_train, df_test = dfs[split_pfx]

        bs = 8 if model_name in LM_MODELS else BS

        for seed in args.seeds:
            exp        = f"{exp_pfx}_s{seed}"
            model_path = SAVED / model_name / exp / str(seed) / "train" / "model.pth"
            if not model_path.exists():
                print(f"  [SKIP] {exp} — not found")
                continue

            print(f"  [EVAL] {exp}")
            try:
                trainer = build_trainer(model_name, exp, seed, model_path, args.device)
            except Exception as e:
                print(f"  [ERROR] {e}")
                continue

            metrics = evaluate(trainer, df_train, df_test, seed, args.device, bs=bs)
            del trainer
            torch.cuda.empty_cache()

            if metrics is None:
                continue

            print(f"    AUROC={metrics['auroc']:.4f}  AUPRC={metrics['auprc']:.4f}  "
                  f"F1pos={metrics['f1_pos']:.4f}  MCC={metrics['mcc']:.4f}")
            rows.append({
                "split":      split_pfx,
                "model_name": model_name,
                "label":      label,
                "exp":        exp,
                "seed":       seed,
                **metrics,
            })

    if not rows:
        print("\nNo results.")
        return

    df_raw = pd.DataFrame(rows)
    raw_path = OUT / "disjoint_new_raw.csv"
    df_raw.to_csv(raw_path, index=False)

    # Summary
    metric_cols = ["auroc", "auprc", "f1_macro", "f1_pos", "prec_pos", "rec_pos", "mcc"]
    summary_rows = []
    for (split, lbl), grp in df_raw.groupby(["split", "label"]):
        row = {"split": split, "label": lbl, "n_seeds": len(grp)}
        for m in metric_cols:
            row[f"{m}_mean"] = round(grp[m].mean(), 4)
            row[f"{m}_std"]  = round(grp[m].std(),  4)
        summary_rows.append(row)
    df_sum = pd.DataFrame(summary_rows)
    sum_path = OUT / "disjoint_new_summary.csv"
    df_sum.to_csv(sum_path, index=False)

    # Paper table
    lines = []
    def tee(s): print(s); lines.append(s)

    tee("\n" + "="*76)
    tee("  Disjoint Split Results  (trained & evaluated on same split)")
    tee("="*76)

    for split_name in ["iso", "bsj"]:
        tee(f"\n  [{split_name.upper()}-DISJOINT]")
        tee(f"  {'Model':<12}  {'AUROC':>12}  {'AUPRC':>12}  {'F1(pos)':>12}  {'MCC':>12}")
        tee("  " + "-"*58)
        grp = df_sum[df_sum["split"] == split_name].sort_values("auroc_mean", ascending=False)
        for _, r in grp.iterrows():
            tee(f"  {r['label']:<12}  "
                f"{r['auroc_mean']:.4f}±{r['auroc_std']:.4f}  "
                f"{r['auprc_mean']:.4f}±{r['auprc_std']:.4f}  "
                f"{r['f1_pos_mean']:.4f}±{r['f1_pos_std']:.4f}  "
                f"{r['mcc_mean']:.4f}±{r['mcc_std']:.4f}")

    tee(f"\n  Saved: {raw_path}")
    (OUT / "disjoint_new_table.txt").write_text("\n".join(lines))
    print(f"  Saved summary: {sum_path}")


if __name__ == "__main__":
    main()
