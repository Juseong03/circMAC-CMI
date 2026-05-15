#!/usr/bin/env python3
"""
eval_all_models.py
saved_models/ 의 모든 실험을 test set으로 재평가하여 올바른 metrics를 기록한다.

- compute_roc_data.py 와 동일한 방식으로 labels[:, 1:] (CLS 제거) 후 계산
- 결과: eval_results/eval_summary.csv

Usage:
    python scripts/eval_all_models.py --device 0
    python scripts/eval_all_models.py --device 0 --group encoder
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trainer import Trainer
from utils import prepare_datasets
from utils_config import get_model_config

SAVED = ROOT / "saved_models"
OUT   = ROOT / "eval_results"
OUT.mkdir(parents=True, exist_ok=True)

MAX_LEN  = 1022
D_MODEL  = 128
N_LAYER  = 6
BS       = 32
WORKERS  = 4
SEEDS    = [1, 2, 3]

LM_MAX_LEN = {
    "rnabert":  438,
    "rnaernie": 511,
    "rnamsm":   1022,
    "rnafm":    1022,
    "circmac":  1022,
}

# ── 실험 목록 ─────────────────────────────────────────────────────────────────
# (group, label, model_name, exp_template, interaction, trainable_pt)
EXPERIMENTS = [
    # Encoder comparison
    ("encoder", "LSTM",        "lstm",        "v2_enc_lstm",        "cross_attention", False),
    ("encoder", "Transformer", "transformer", "v2_enc_transformer", "cross_attention", False),
    ("encoder", "Mamba",       "mamba",       "v2_enc_mamba",       "cross_attention", False),
    ("encoder", "Hymba",       "hymba",       "v2_enc_hymba",       "cross_attention", False),
    ("encoder", "CircMAC",     "circmac",     "v2_abl_full",        "cross_attention", False),

    # Pretrained comparison
    ("pretrained", "RNABERT (frozen)",      "rnabert",  "exp1_fair_frozen_rnabert",     "cross_attention", False),
    ("pretrained", "RNABERT (fine-tuned)",  "rnabert",  "exp1_fair_trainable_rnabert",  "cross_attention", True),
    ("pretrained", "RNAErnie (frozen)",     "rnaernie", "exp1_fair_frozen_rnaernie",    "cross_attention", False),
    ("pretrained", "RNAErnie (fine-tuned)", "rnaernie", "exp1_fair_trainable_rnaernie", "cross_attention", True),
    ("pretrained", "RNAMSM (frozen)",       "rnamsm",   "exp1_fair_frozen_rnamsm",      "cross_attention", False),
    ("pretrained", "RNAMSM (fine-tuned)",   "rnamsm",   "exp1_fair_trainable_rnamsm",   "cross_attention", True),
    ("pretrained", "RNA-FM (frozen)",       "rnafm",    "exp1_fair_frozen_rnafm",       "cross_attention", False),
    ("pretrained", "RNA-FM (fine-tuned)",   "rnafm",    "exp1_fair_trainable_rnafm",    "cross_attention", True),
    ("pretrained", "CircMAC (NoPT)",        "circmac",  "v2_abl_full",                  "cross_attention", False),
    ("pretrained", "CircMAC (Pairing)",     "circmac",  "v2_pt_pairing",                "cross_attention", False),

    # Ablation — modules
    ("ablation", "CircMAC (full)",      "circmac", "v2_abl_full",         "cross_attention", False),
    ("ablation", "Attn only",           "circmac", "v2_abl_attn_only",    "cross_attention", False),
    ("ablation", "Mamba only",          "circmac", "v2_abl_mamba_only",   "cross_attention", False),
    ("ablation", "CNN only",            "circmac", "v2_abl_cnn_only",     "cross_attention", False),
    ("ablation", "No Attn",             "circmac", "v2_abl_no_attn",      "cross_attention", False),
    ("ablation", "No Mamba",            "circmac", "v2_abl_no_mamba",     "cross_attention", False),
    ("ablation", "No Conv",             "circmac", "v2_abl_no_conv",      "cross_attention", False),
    ("ablation", "No CircBias",         "circmac", "v2_abl_no_circ_bias", "cross_attention", False),

    # Interaction mechanism
    ("interaction", "Concat",        "circmac", "v2_int_concat",       "concat",          False),
    ("interaction", "Elementwise",   "circmac", "v2_int_elementwise",  "elementwise",     False),
    ("interaction", "Cross-Attn",    "circmac", "v2_int_cross_attn",   "cross_attention", False),

    # Site head
    ("site_head", "Conv1D head",  "circmac", "v2_head_conv1d", "cross_attention", False),
    ("site_head", "Linear head",  "circmac", "v2_head_linear", "cross_attention", False),

    # Pretraining strategy
    ("pretraining", "NoPT",          "circmac", "v2_pt_nopt",     "cross_attention", False),
    ("pretraining", "MLM",           "circmac", "v2_pt_mlm",      "cross_attention", False),
    ("pretraining", "NTP",           "circmac", "v2_pt_ntp",      "cross_attention", False),
    ("pretraining", "SSP",           "circmac", "v2_pt_ssp",      "cross_attention", False),
    ("pretraining", "CPCL",          "circmac", "v2_pt_cpcl",     "cross_attention", False),
    ("pretraining", "BSJ",           "circmac", "v2_pt_bsj",      "cross_attention", False),
    ("pretraining", "MLM+NTP",       "circmac", "v2_pt_mlm_ntp",  "cross_attention", False),
    ("pretraining", "MLM+SSP",       "circmac", "v2_pt_mlm_ssp",  "cross_attention", False),
    ("pretraining", "MLM+CPCL",      "circmac", "v2_pt_mlm_cpcl", "cross_attention", False),
    ("pretraining", "Pairing",       "circmac", "v2_pt_pairing",  "cross_attention", False),
    ("pretraining", "MLM+CPCL+SSP",  "circmac", "v2_pt_mlm_cpcl_ssp", "cross_attention", False),
    ("pretraining", "All",           "circmac", "v2_pt_all",      "cross_attention", False),
]


def load_test_data():
    import pandas as pd
    df_test = pd.read_pickle(ROOT / "data/df_test_final.pkl")
    df_test["length"] = df_test["circRNA"].apply(len)
    df_test = df_test[df_test["length"] <= MAX_LEN].reset_index(drop=True)
    df_test = df_test[df_test["binding"] == 1].reset_index(drop=True)
    return df_test


def build_test_dataset(df_test, max_len=None):
    import pandas as pd
    if max_len is None:
        max_len = MAX_LEN
    df_dummy = pd.read_pickle(ROOT / "data/df_train_final.pkl")
    df_dummy["length"] = df_dummy["circRNA"].apply(len)
    df_dummy = df_dummy[df_dummy["length"] <= max_len].reset_index(drop=True)
    df_dummy = df_dummy[df_dummy["binding"] == 1].reset_index(drop=True)
    df_test_f = df_test[df_test["length"] <= max_len].reset_index(drop=True)
    _, _, test_ds, _ = prepare_datasets(
        df=df_dummy, df_test=df_test_f,
        max_len=max_len + 2, target="mirna", seed=1, kmer=1,
    )
    return test_ds


def _get_ckpt_vocab_size(model_path, model_name):
    ckpt  = torch.load(str(model_path), map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    if model_name in ["rnabert", "rnaernie", "rnafm", "rnamsm"]:
        return None
    key = "embedding.word_embeddings.weight"
    if key in state:
        return state[key].shape[0]
    for k, v in state.items():
        if "embedding" in k and "weight" in k and v.ndim == 2 and v.shape[-1] == D_MODEL:
            return v.shape[0]
    return None


def run_inference(model_name, exp_template, interaction, trainable_pt,
                  df_test, device, test_ds_cache=None, max_len=None):
    """모든 seed에 대해 inference 실행 → metrics list 반환"""
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

    results = []
    for seed in SEEDS:
        exp        = f"{exp_template}_s{seed}"
        model_path = SAVED / model_name / exp / str(seed) / "train" / "model.pth"
        if not model_path.exists():
            print(f"  [SKIP] {exp} — not found")
            continue

        print(f"  [RUN]  {exp}")
        if test_ds_cache is None:
            test_ds = build_test_dataset(df_test, max_len=max_len)
        else:
            test_ds = test_ds_cache

        trainer = Trainer(seed=seed, device=device,
                          experiment_name=exp, verbose=False)
        trainer.set_dataloader(test_ds, part=2, batch_size=BS, num_workers=WORKERS)

        ckpt_vocab = _get_ckpt_vocab_size(model_path, model_name)
        config = get_model_config(
            model_name=model_name, d_model=D_MODEL, n_layer=N_LAYER,
            verbose=False, rc=False,
            **({} if ckpt_vocab is None else {"vocab_size": ckpt_vocab}),
        )
        if model_name in ["rnabert", "rnaernie", "rnafm", "rnamsm"]:
            config.trainable = trainable_pt

        trainer.define_model(
            config=config, model_name=model_name, pretrain=False,
            pooling_mode_target="mean", is_convblock=True,
            is_cross_attention=(interaction == "cross_attention"),
            interaction=interaction, use_unified_head=False,
            binding_pooling="mean", site_head_type="conv1d",
        )
        if model_name in ["rnabert", "rnaernie", "rnafm", "rnamsm"]:
            if not trainable_pt:
                trainer.define_pretrained_model(model_name=model_name)

        trainer.set_pretrained_target(target="mirna", rna_model="rnabert")
        trainer.task = "sites"
        trainer.site_class_weights = None
        trainer.alpha = 0.5
        trainer.beta  = 0.5
        trainer.rc    = False
        trainer.load_model_from_path(str(model_path), verbose=False)

        try:
            _, tensors, _ = trainer.step_loader(
                trainer.test_loader, 0, is_train=False, data_type="Test"
            )
        except Exception as e:
            print(f"  [ERROR] {exp}: {e}")
            del trainer; torch.cuda.empty_cache()
            continue

        preds_raw  = tensors["preds_sites"]   # (N, L-1, 2)
        labels_raw = tensors["labels_sites"]  # (N, L)

        # CLS 제거 후 올바른 alignment
        labels_aligned = labels_raw[:, 1:]
        labels_flat    = labels_aligned.reshape(-1).numpy()
        valid_mask     = labels_flat != -100

        if preds_raw.ndim == 3:
            preds_flat = preds_raw.reshape(-1, preds_raw.shape[-1])
        else:
            preds_flat = preds_raw

        if preds_flat.shape[-1] == 2:
            preds_prob = torch.softmax(preds_flat.float(), dim=-1)[:, 1].numpy()
        else:
            preds_prob = preds_flat.squeeze(-1).numpy()

        preds_prob  = preds_prob[valid_mask]
        labels_keep = labels_flat[valid_mask]

        try:
            auroc = float(roc_auc_score(labels_keep, preds_prob))
            auprc = float(average_precision_score(labels_keep, preds_prob))
        except Exception:
            auroc, auprc = float("nan"), float("nan")

        preds_bin = (preds_prob >= 0.5).astype(int)
        f1 = float(f1_score(labels_keep, preds_bin, pos_label=1, zero_division=0))

        results.append({"seed": seed, "auroc": auroc, "auprc": auprc, "f1_pos": f1})
        del trainer; torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--group", default="all",
                        help="all / encoder / pretrained / ablation / interaction / site_head / pretraining")
    args = parser.parse_args()

    print("Loading test data...")
    df_test = load_test_data()
    print(f"  {len(df_test)} samples")

    # test dataset cache per max_len
    ds_cache = {}
    rows = []

    groups = [args.group] if args.group != "all" else \
        ["encoder", "pretrained", "ablation", "interaction", "site_head", "pretraining"]

    for group, label, model_name, exp_tpl, interaction, trainable in EXPERIMENTS:
        if group not in groups:
            continue

        ml = LM_MAX_LEN.get(model_name, MAX_LEN)
        if ml not in ds_cache:
            print(f"\nBuilding test dataset (max_len={ml})...")
            ds_cache[ml] = build_test_dataset(df_test, max_len=ml)

        print(f"\n[{group}] {label}")
        results = run_inference(
            model_name, exp_tpl, interaction, trainable,
            df_test, args.device,
            test_ds_cache=ds_cache[ml], max_len=ml,
        )

        if not results:
            print(f"  → No results (all seeds missing)")
            continue

        aurocs = [r["auroc"] for r in results]
        auprcs = [r["auprc"] for r in results]
        f1s    = [r["f1_pos"] for r in results]

        print(f"  AUROC: {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")
        print(f"  AUPRC: {np.mean(auprcs):.4f} ± {np.std(auprcs):.4f}")
        print(f"  F1:    {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

        for r in results:
            rows.append({
                "group":      group,
                "label":      label,
                "model_name": model_name,
                "exp_tpl":    exp_tpl,
                "seed":       r["seed"],
                "auroc":      f"{r['auroc']:.4f}",
                "auprc":      f"{r['auprc']:.4f}",
                "f1_pos":     f"{r['f1_pos']:.4f}",
            })
        # mean row
        rows.append({
            "group":      group,
            "label":      label,
            "model_name": model_name,
            "exp_tpl":    exp_tpl,
            "seed":       "mean",
            "auroc":      f"{np.mean(aurocs):.4f}",
            "auprc":      f"{np.mean(auprcs):.4f}",
            "f1_pos":     f"{np.mean(f1s):.4f}",
        })
        rows.append({
            "group": group, "label": label, "model_name": model_name,
            "exp_tpl": exp_tpl, "seed": "std",
            "auroc": f"{np.std(aurocs):.4f}",
            "auprc": f"{np.std(auprcs):.4f}",
            "f1_pos": f"{np.std(f1s):.4f}",
        })

    # Save CSV
    csv_path = OUT / "eval_summary.csv"
    fieldnames = ["group", "label", "model_name", "exp_tpl", "seed", "auroc", "auprc", "f1_pos"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'='*50}")
    print(f" Saved → {csv_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
