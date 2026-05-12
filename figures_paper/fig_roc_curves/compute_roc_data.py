#!/usr/bin/env python3
"""
compute_roc_data.py — Run inference on test set and cache ROC data

Loads each model checkpoint from models_for_viz/, runs on the test set,
and saves (preds, labels) as a pickle cache for downstream ROC plotting.

Usage:
    python figures_paper/fig_roc_curves/compute_roc_data.py --device 0
    python figures_paper/fig_roc_curves/compute_roc_data.py --device 0 --group encoder
    python figures_paper/fig_roc_curves/compute_roc_data.py --device 0 --group rna_lm

Output:
    figures_paper/fig_roc_curves/roc_cache_encoder.pkl
    figures_paper/fig_roc_curves/roc_cache_rna_lm.pkl
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from trainer import Trainer
from data import prepare_datasets
from utils_config import get_model_config

OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

VIZ  = ROOT / "models_for_viz"   # all checkpoints live here

MAX_LEN  = 1022
D_MODEL  = 128
N_LAYER  = 6
BS       = 128
WORKERS  = 4
TASK     = "sites"
SEEDS    = [1, 2, 3]

# ── Model definitions ─────────────────────────────────────────────────────────
# (label, model_name, exp_template, interaction, trainable_pretrained)
ENCODER_MODELS = [
    ("LSTM",        "lstm",        "v2_enc_lstm",        "cross_attention", False),
    ("Transformer", "transformer", "v2_enc_transformer", "cross_attention", False),
    ("Mamba",       "mamba",       "v2_enc_mamba",       "cross_attention", False),
    ("Hymba",       "hymba",       "v2_enc_hymba",       "cross_attention", False),
    ("CircMAC",     "circmac",     "v2_enc_circmac",     "cross_attention", False),
]

RNA_LM_MODELS = [
    ("RNABERT (frozen)",      "rnabert",  "exp1_fair_frozen_rnabert",     "cross_attention", False),
    ("RNAErnie (frozen)",     "rnaernie", "exp1_fair_frozen_rnaernie",    "cross_attention", False),
    ("RNAMSM (frozen)",       "rnamsm",   "exp1_fair_frozen_rnamsm",      "cross_attention", False),
    ("RNA-FM (frozen)",       "rnafm",    "exp1_fair_frozen_rnafm",       "cross_attention", False),
    ("RNABERT (fine-tuned)",  "rnabert",  "exp1_fair_trainable_rnabert",  "cross_attention", True),
    ("RNAErnie (fine-tuned)", "rnaernie", "exp1_fair_trainable_rnaernie", "cross_attention", True),
    ("RNAMSM (fine-tuned)",   "rnamsm",   "exp1_fair_trainable_rnamsm",   "cross_attention", True),
    ("RNA-FM (fine-tuned)",   "rnafm",    "exp1_fair_trainable_rnafm",    "cross_attention", True),
    ("CircMAC+Pairing",       "circmac",  "v2_pt_pairing",                "cross_attention", False),
]


def load_test_data():
    import pandas as pd
    df_test = pd.read_pickle(ROOT / "data/df_test_final.pkl")
    df_test["length"] = df_test["circRNA"].apply(len)
    df_test = df_test[df_test["length"] <= MAX_LEN].reset_index(drop=True)
    df_test = df_test[df_test["binding"] == 1].reset_index(drop=True)
    return df_test


def run_inference(label, model_name, exp_template, interaction,
                  trainable_pretrained, df_test, device):
    """Run inference for all seeds and return list of {seed, preds, labels} dicts."""
    results = []

    for seed in SEEDS:
        exp        = f"{exp_template}_s{seed}"
        model_path = VIZ / model_name / exp / str(seed) / "train" / "model.pth"

        if not model_path.exists():
            print(f"  [SKIP] {exp} — not found: {model_path}")
            continue

        print(f"  [RUN]  {exp}")

        import pandas as pd
        df_dummy = pd.read_pickle(ROOT / "data/df_train_final.pkl")
        df_dummy["length"] = df_dummy["circRNA"].apply(len)
        df_dummy = df_dummy[df_dummy["length"] <= MAX_LEN].reset_index(drop=True)
        df_dummy = df_dummy[df_dummy["binding"] == 1].reset_index(drop=True)

        _, _, test_dataset, _ = prepare_datasets(
            df=df_dummy,
            df_test=df_test,
            max_len=MAX_LEN + 2,
            target="mirna",
            seed=seed,
            kmer=1,
        )

        trainer = Trainer(seed=seed, device=device,
                          experiment_name=exp, verbose=False)
        trainer.set_dataloader(test_dataset, part=2,
                               batch_size=BS, num_workers=WORKERS)

        config = get_model_config(
            model_name=model_name,
            d_model=D_MODEL,
            n_layer=N_LAYER,
            verbose=False,
            rc=False,
            vocab_size=test_dataset.vocab_size,
        )

        if model_name in ["rnabert", "rnaernie", "rnafm", "rnamsm"]:
            config.trainable = trainable_pretrained

        trainer.define_model(
            config=config,
            model_name=model_name,
            pretrain=False,
            pooling_mode_target="mean",
            is_convblock=True,
            is_cross_attention=(interaction == "cross_attention"),
            interaction=interaction,
            use_unified_head=False,
            binding_pooling="mean",
            site_head_type="conv1d",
        )

        if model_name in ["rnabert", "rnaernie", "rnafm", "rnamsm"]:
            if not trainable_pretrained:
                trainer.define_pretrained_model(model_name=model_name)

        trainer.set_pretrained_target(target="mirna", rna_model=None)
        trainer.task = TASK

        trainer.load_model_from_path(str(model_path), verbose=False)

        _, tensors, _ = trainer.step_loader(
            trainer.test_loader, 0, is_train=False, data_type="Test"
        )

        preds  = tensors["preds_sites"].numpy()
        labels = tensors["labels_sites"].numpy()

        if preds.ndim == 2 and preds.shape[1] == 2:
            preds_prob = torch.softmax(torch.tensor(preds), dim=-1)[:, 1].numpy()
        else:
            preds_prob = preds

        results.append({"seed": seed, "preds": preds_prob, "labels": labels})
        del trainer
        torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--group", choices=["encoder", "rna_lm", "all"],
                        default="all")
    args = parser.parse_args()

    print("Loading test data...")
    df_test = load_test_data()
    print(f"  Test set: {len(df_test)} samples")

    if args.group in ("encoder", "all"):
        print("\n=== Encoder models ===")
        cache = {}
        for label, model_name, exp_tpl, interaction, trainable in ENCODER_MODELS:
            print(f"\n[{label}]")
            cache[label] = run_inference(
                label, model_name, exp_tpl, interaction, trainable, df_test, args.device
            )
        out_path = OUT / "roc_cache_encoder.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(cache, f)
        print(f"\nSaved → {out_path}")

    if args.group in ("rna_lm", "all"):
        print("\n=== RNA-LM models ===")
        cache = {}
        for label, model_name, exp_tpl, interaction, trainable in RNA_LM_MODELS:
            print(f"\n[{label}]")
            cache[label] = run_inference(
                label, model_name, exp_tpl, interaction, trainable, df_test, args.device
            )
        out_path = OUT / "roc_cache_rna_lm.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(cache, f)
        print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
