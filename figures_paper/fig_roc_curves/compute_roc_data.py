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
from utils import prepare_datasets
from utils_config import get_model_config

OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

VIZ  = ROOT / "models_for_viz"   # all checkpoints live here

MAX_LEN  = 1022
D_MODEL  = 128
N_LAYER  = 6
BS       = 32
WORKERS  = 4
TASK     = "sites"
SEEDS    = [1, 2, 3]

# Max sequence length per pretrained model name (position embedding limit - 2)
# RNABERT/RNAErnie: hard architecture limits (position embedding size)
# RNAMSM/RNA-FM: previously capped at 510 to avoid CUDA errors on small GPUs;
#                use 1022 on large-memory GPUs
LM_MAX_LEN = {
    "rnabert":  438,   # hard limit: max_position_embeddings=440
    "rnaernie": 511,   # hard limit: max_position_embeddings=513
    "rnamsm":   1022,  # no architecture limit; use full length on large GPU
    "rnafm":    1022,  # no architecture limit; use full length on large GPU
    "circmac":  1022,
}

# ── Model definitions ─────────────────────────────────────────────────────────
# (label, model_name, exp_template, interaction, trainable_pretrained)
ENCODER_MODELS = [
    ("LSTM",        "lstm",        "v2_enc_lstm",        "cross_attention", False),
    ("Transformer", "transformer", "v2_enc_transformer", "cross_attention", False),
    ("Mamba",       "mamba",       "v2_enc_mamba",       "cross_attention", False),
    ("Hymba",       "hymba",       "v2_enc_hymba",       "cross_attention", False),
    ("CircMAC",     "circmac",     "v2_abl_full",        "cross_attention", False),
]

# Pretrained RNA-LM comparison (fine-tuned) + CircMAC variants
PRETRAINED_MODELS = [
    ("RNABERT",            "rnabert",  "exp1_fair_trainable_rnabert",  "cross_attention", True),
    ("RNAErnie",           "rnaernie", "exp1_fair_trainable_rnaernie", "cross_attention", True),
    ("RNAMSM",             "rnamsm",   "exp1_fair_trainable_rnamsm",   "cross_attention", True),
    ("RNA-FM",             "rnafm",    "exp1_fair_trainable_rnafm",    "cross_attention", True),
    ("CircMAC (Pairing)",  "circmac",  "v2_pt_pairing",                "cross_attention", False),
    ("CircMAC (NoPT)",     "circmac",  "v2_abl_full",                  "cross_attention", False),
]

# Legacy full RNA-LM group (frozen + fine-tuned)
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


def _get_ckpt_vocab_size(model_path, model_name):
    """Read vocab size from checkpoint embedding layer to avoid shape mismatch."""
    ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    # Most models use embedding.word_embeddings.weight; pretrained use their own embedding
    if model_name in ["rnabert", "rnaernie", "rnafm", "rnamsm"]:
        # Pretrained models don't have a trainable embedding — use dataset vocab
        return None
    key = "embedding.word_embeddings.weight"
    if key in state:
        return state[key].shape[0]
    # Fallback: scan for any embedding weight
    for k, v in state.items():
        if "embedding" in k and "weight" in k and v.ndim == 2 and v.shape[-1] == D_MODEL:
            return v.shape[0]
    return None  # Let get_model_config use its default


def build_test_dataset(df_test, max_len=None):
    """Build the test dataset (seed-independent). max_len defaults to MAX_LEN."""
    import pandas as pd
    if max_len is None:
        max_len = MAX_LEN
    df_dummy = pd.read_pickle(ROOT / "data/df_train_final.pkl")
    df_dummy["length"] = df_dummy["circRNA"].apply(len)
    df_dummy = df_dummy[df_dummy["length"] <= max_len].reset_index(drop=True)
    df_dummy = df_dummy[df_dummy["binding"] == 1].reset_index(drop=True)
    # Filter test set to the same max_len
    df_test_filtered = df_test[df_test["length"] <= max_len].reset_index(drop=True)
    _, _, test_dataset, _ = prepare_datasets(
        df=df_dummy, df_test=df_test_filtered,
        max_len=max_len + 2, target="mirna", seed=1, kmer=1,
    )
    return test_dataset


def run_inference(label, model_name, exp_template, interaction,
                  trainable_pretrained, df_test, device,
                  test_dataset_cache=None, max_len=None):
    """Run inference for all seeds and return list of {seed, preds, labels} dicts."""
    if test_dataset_cache is None:
        test_dataset = build_test_dataset(df_test, max_len=max_len)
    else:
        test_dataset = test_dataset_cache

    results = []
    for seed in SEEDS:
        exp        = f"{exp_template}_s{seed}"
        model_path = VIZ / model_name / exp / str(seed) / "train" / "model.pth"

        if not model_path.exists():
            print(f"  [SKIP] {exp} — not found: {model_path}")
            continue

        print(f"  [RUN]  {exp}")

        trainer = Trainer(seed=seed, device=device,
                          experiment_name=exp, verbose=False)
        trainer.set_dataloader(test_dataset, part=2,
                               batch_size=BS, num_workers=WORKERS)

        # Read vocab_size from checkpoint to avoid embedding shape mismatch
        ckpt_vocab = _get_ckpt_vocab_size(model_path, model_name)

        config = get_model_config(
            model_name=model_name,
            d_model=D_MODEL, n_layer=N_LAYER,
            verbose=False, rc=False,
            **({} if ckpt_vocab is None else {"vocab_size": ckpt_vocab}),
        )

        if model_name in ["rnabert", "rnaernie", "rnafm", "rnamsm"]:
            config.trainable = trainable_pretrained

        trainer.define_model(
            config=config, model_name=model_name, pretrain=False,
            pooling_mode_target="mean", is_convblock=True,
            is_cross_attention=(interaction == "cross_attention"),
            interaction=interaction, use_unified_head=False,
            binding_pooling="mean", site_head_type="conv1d",
        )

        if model_name in ["rnabert", "rnaernie", "rnafm", "rnamsm"]:
            if not trainable_pretrained:
                trainer.define_pretrained_model(model_name=model_name)

        # Always call set_pretrained_target to properly initialize proj_target dims
        trainer.set_pretrained_target(target="mirna", rna_model="rnabert")

        trainer.task = TASK
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
            del trainer
            torch.cuda.empty_cache()
            continue

        preds_raw  = tensors["preds_sites"]   # (N, L-1, 2) — no CLS prediction
        labels_raw = tensors["labels_sites"]  # (N, L) with CLS at index 0 + -100 padding

        # Skip CLS token at index 0 to align with preds_raw (which has no CLS prediction)
        labels_aligned = labels_raw[:, 1:]                   # (N, L-1)
        labels_flat    = labels_aligned.reshape(-1).numpy()  # (N*(L-1),)
        valid_mask     = labels_flat != -100

        # Convert logits → class-1 probability; flatten to (N*(L-1), 2) first
        if preds_raw.ndim == 3:
            preds_flat = preds_raw.reshape(-1, preds_raw.shape[-1])  # (N*(L-1), 2)
        else:
            preds_flat = preds_raw

        if preds_flat.shape[-1] == 2:
            preds_prob = torch.softmax(preds_flat.float(), dim=-1)[:, 1].numpy()
        else:
            preds_prob = preds_flat.squeeze(-1).numpy()

        preds_prob  = preds_prob[valid_mask]
        labels_keep = labels_flat[valid_mask]

        results.append({"seed": seed, "preds": preds_prob, "labels": labels_keep})
        del trainer
        torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--group", choices=["encoder", "pretrained", "rna_lm", "all"],
                        default="all")
    args = parser.parse_args()

    print("Loading test data...")
    df_test = load_test_data()
    print(f"  Test set: {len(df_test)} samples")

    print("Building test dataset (shared across all models)...")
    shared_test_ds = build_test_dataset(df_test)
    print("  Done.")

    if args.group in ("encoder", "all"):
        print("\n=== Encoder models ===")
        cache = {}
        for label, model_name, exp_tpl, interaction, trainable in ENCODER_MODELS:
            print(f"\n[{label}]")
            cache[label] = run_inference(
                label, model_name, exp_tpl, interaction, trainable, df_test, args.device,
                test_dataset_cache=shared_test_ds,
            )
        out_path = OUT / "roc_cache_encoder.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(cache, f)
        print(f"\nSaved → {out_path}")

    if args.group in ("pretrained", "all"):
        print("\n=== Pretrained models (fine-tuned RNA-LMs + CircMAC variants) ===")
        cache = {}
        pt_ds_cache = {}
        for label, model_name, exp_tpl, interaction, trainable in PRETRAINED_MODELS:
            ml = LM_MAX_LEN.get(model_name, MAX_LEN)
            if ml not in pt_ds_cache:
                print(f"\nBuilding test dataset for max_len={ml}...")
                pt_ds_cache[ml] = build_test_dataset(df_test, max_len=ml)
                print(f"  Done ({len(pt_ds_cache[ml])} samples)")
            print(f"\n[{label}]  (max_len={ml})")
            cache[label] = run_inference(
                label, model_name, exp_tpl, interaction, trainable, df_test,
                args.device, test_dataset_cache=pt_ds_cache[ml], max_len=ml,
            )
        out_path = OUT / "roc_cache_pretrained.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(cache, f)
        print(f"\nSaved → {out_path}")

    if args.group in ("rna_lm",):
        print("\n=== RNA-LM models (legacy: frozen + fine-tuned) ===")
        cache = {}
        lm_ds_cache = {}
        for label, model_name, exp_tpl, interaction, trainable in RNA_LM_MODELS:
            ml = LM_MAX_LEN.get(model_name, MAX_LEN)
            if ml not in lm_ds_cache:
                print(f"\nBuilding test dataset for max_len={ml}...")
                lm_ds_cache[ml] = build_test_dataset(df_test, max_len=ml)
                print(f"  Done ({len(lm_ds_cache[ml])} samples)")
            print(f"\n[{label}]  (max_len={ml})")
            cache[label] = run_inference(
                label, model_name, exp_tpl, interaction, trainable, df_test,
                args.device, test_dataset_cache=lm_ds_cache[ml], max_len=ml,
            )
        out_path = OUT / "roc_cache_rna_lm.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(cache, f)
        print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
