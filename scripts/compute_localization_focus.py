#!/usr/bin/env python3
"""
compute_localization_focus.py
=============================
Aggregate localization-concentration analysis around annotated binding regions.

This replaces the previous BSJ-only focus score. The ground-truth nucleotide-level
binding mask is used to define the target region for each circRNA-miRNA pair.

For each pair and radius r:

    LPM(r) = sum_{i in G_{+-r}} p_i / sum_i p_i

where G_{+-r} is the union of annotated binding positions expanded by r nt on
both sides. Expansion is circular, so windows can cross the BSJ.

Because a uniform predictor receives non-zero LPM simply from the size of the
window, we also report the expected uniform mass:

    U(r) = |G_{+-r}| / L_valid

and a chance-corrected normalized localization score:

    NLS(r) = (LPM(r) - U(r)) / (1 - U(r))

Interpretation of NLS:
    1.0 : all prediction mass lies inside the annotated-region window
    0.0 : prediction mass is distributed uniformly across valid nucleotides
    <0  : prediction mass is depleted around annotated regions

Notes
-----
* Only pairs with at least one annotated binding position are included.
* For model-specific context limits, sequences longer than the model's supported
  full-sequence length are excluded rather than truncated. This avoids treating
  an unseen/truncated binding site as a localization failure.
* To compare ALL models on exactly the same sequence subset, use a common cap,
  e.g. --max_eval_len 436.

Examples
--------
    python scripts/compute_localization_focus.py --split pair --device 0
    python scripts/compute_localization_focus.py --split pair --radii 0 10 20 --seeds 1 2 3
    python scripts/compute_localization_focus.py --split all --device 0

    # Fair subset for RNABERT/RNAErnie/long-context models together:
    python scripts/compute_localization_focus.py --split pair --max_eval_len 436
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trainer import Trainer
from data import CircRNABindingSitesDataset, KmerTokenizer
from utils import get_device
from utils_config import get_model_config


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
MAX_DATASET_LEN = 1000
MODEL_DATASET_MAX_LEN = {
    "rnabert":     436,
    "rnaernie":    511,
    "rnamsm":      MAX_DATASET_LEN,
    "rnafm":       MAX_DATASET_LEN,
    "lstm":        MAX_DATASET_LEN,
    "transformer": MAX_DATASET_LEN,
    "mamba":       MAX_DATASET_LEN,
    "hymba":       MAX_DATASET_LEN,
    "circmac":     MAX_DATASET_LEN,
}

SPECIAL_TOKEN_ALLOWANCE = 2

# (display label, model_name, experiment prefix, frozen backbone)
MODEL_SPECS = [
    # circMAC
    ("CircMAC (NoPT)",    "circmac",     "v2_abl_full",                 False),
    ("CircMAC (MLM)",     "circmac",     "v2_pt_mlm",                   False),
    ("CircMAC (Pairing)", "circmac",     "v2_pt_pairing",               False),
    # General sequence encoders
    ("LSTM",              "lstm",        "v2_enc_lstm",                 False),
    ("Transformer",       "transformer", "v2_enc_transformer",          False),
    ("Mamba",             "mamba",       "v2_enc_mamba",                False),
    ("Hymba",             "hymba",       "v2_enc_hymba",                False),
    # RNA LMs: frozen
    ("RNABERT (frozen)",  "rnabert",     "exp1_fair_frozen_rnabert",    True),
    ("RNAErnie (frozen)", "rnaernie",    "exp1_fair_frozen_rnaernie",   True),
    ("RNAMSM (frozen)",   "rnamsm",      "exp1_fair_frozen_rnamsm",     True),
    ("RNA-FM (frozen)",   "rnafm",       "exp1_fair_frozen_rnafm",      True),
    # RNA LMs: fine-tuned
    ("RNABERT (ft)",      "rnabert",     "exp1_fair_trainable_rnabert", False),
    ("RNAErnie (ft)",     "rnaernie",    "exp1_fair_trainable_rnaernie",False),
    ("RNAMSM (ft)",       "rnamsm",      "exp1_fair_trainable_rnamsm",  False),
    ("RNA-FM (ft)",       "rnafm",       "exp1_fair_trainable_rnafm",   False),
]

SPLIT_FILES = {
    "pair": ("data/df_train_final.pkl",        "data/df_test_final.pkl"),
    "bsj":  ("data/df_train_bsj_disjoint.pkl", "data/df_test_bsj_disjoint.pkl"),
    "iso":  ("data/df_train_iso_disjoint.pkl", "data/df_test_iso_disjoint.pkl"),
}


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def get_ckpt(model_name: str, exp_name: str, seed: int) -> Optional[Path]:
    candidates = [
        ROOT / "saved_models" / model_name / f"{exp_name}_s{seed}" / str(seed) / "train" / "model.pth",
        ROOT / "saved_models" / model_name / f"{exp_name}_s{seed}" / "train" / "model.pth",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def get_vocab_size(ckpt_path: Optional[Path]) -> Optional[int]:
    if ckpt_path is None:
        return None
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        key = "embedding.word_embeddings.weight"
        if key in state:
            return int(state[key].shape[0])
        for k, v in state.items():
            if "embedding" in k and "weight" in k and hasattr(v, "ndim") and v.ndim == 2 and v.shape[-1] == 128:
                return int(v.shape[0])
    except Exception:
        pass
    return None


def make_dataset(df: pd.DataFrame, max_len: int, vocab_size_override: Optional[int] = None):
    ds = CircRNABindingSitesDataset(
        df.reset_index(drop=True),
        max_len=max_len,
        target_type="mirna",
        k=1,
        k_target=1,
    )
    if vocab_size_override == 11 and ds.vocab_size != 11:
        ds.circrna_tokenizer = KmerTokenizer(1)
        ds.vocab_size = 11
    return ds


def build_trainer(model_name, exp_name, seed, device, frozen, vocab_size):
    exp_folder = f"{exp_name}_s{seed}"
    tr = Trainer(
        seed=seed, device=device,
        dir_save=str(ROOT / "saved_models"),
        experiment_name=exp_folder,
        verbose=False,
    )
    tr.task = "sites"
    tr.rc   = False
    tr.use_unified_head = False
    tr.interaction = "cross_attention"

    cfg = get_model_config(model_name=model_name, d_model=128, n_layer=6, vocab_size=vocab_size)
    if model_name in ["rnabert", "rnaernie", "rnafm", "rnamsm"]:
        cfg.trainable = not frozen

    tr.define_model(cfg, model_name=model_name, pretrain=False,
                    is_cross_attention=True, interaction="cross_attention",
                    site_head_type="conv1d")

    if model_name in ["rnabert", "rnaernie", "rnafm", "rnamsm"] and frozen:
        tr.define_pretrained_model(model_name=model_name)

    tr.set_pretrained_target(target="mirna", rna_model="rnabert")

    ckpt = get_ckpt(model_name, exp_name, seed)
    if ckpt is not None:
        tr.load_model_from_path(str(ckpt), verbose=False)
    else:
        tr.load_model(epoch=None, pretrain=False, verbose=False)

    tr.model.eval()
    return tr


def logits_or_probs_to_probs(preds: torch.Tensor) -> torch.Tensor:
    if preds.ndim == 3 and preds.shape[-1] == 2:
        return torch.softmax(preds.float(), dim=-1)[..., 1]
    scores = preds.float().squeeze(-1)
    if scores.numel() == 0:
        return scores
    s_min = float(scores.detach().min().cpu())
    s_max = float(scores.detach().max().cpu())
    if s_min < 0.0 or s_max > 1.0:
        scores = torch.sigmoid(scores)
    return scores


def align_sites_to_predictions(site_tensor: torch.Tensor, n_pred: int) -> np.ndarray:
    labels = site_tensor.detach().cpu().numpy().reshape(-1)
    if len(labels) >= n_pred + 1 and labels[0] == -100:
        labels = labels[1: 1 + n_pred]
    elif len(labels) >= n_pred:
        labels = labels[:n_pred]
    else:
        raise ValueError(f"Site label array shorter than predictions: labels={len(labels)}, n_pred={n_pred}")
    return labels.astype(np.int16, copy=False)


def circular_expand_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Expand a boolean mask by +/-radius nt with circular wrap-around."""
    mask = np.asarray(mask, dtype=bool)
    if radius <= 0 or mask.size == 0:
        return mask.copy()
    out = mask.copy()
    for shift in range(1, radius + 1):
        out |= np.roll(mask, shift)
        out |= np.roll(mask, -shift)
    return out


def compute_localization_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    radius: int,
) -> Optional[Dict[str, float]]:
    """Compute LPM, uniform expectation U, and NLS for one pair."""
    labels = np.asarray(labels)
    probs  = np.asarray(probs, dtype=np.float64)

    valid = labels != -100
    if not np.any(valid):
        return None

    labels_valid = labels[valid]
    probs_valid  = probs[valid]
    positive_valid = labels_valid == 1

    if not np.any(positive_valid):
        return None  # no annotated binding site → metric undefined

    target_window = circular_expand_mask(positive_valid, radius)

    total_mass = float(probs_valid.sum())
    if total_mass <= 1e-12:
        return None

    lpm = float(probs_valid[target_window].sum()) / total_mass

    n_valid  = int(len(labels_valid))
    n_window = int(target_window.sum())
    uniform_mass = n_window / n_valid

    if uniform_mass >= 1.0 - 1e-12:
        nls = np.nan
    else:
        nls = (lpm - uniform_mass) / (1.0 - uniform_mass)

    return {
        "lpm":           float(lpm),
        "uniform_mass":  float(uniform_mass),
        "normalized_score": float(nls),
        "n_valid":       n_valid,
        "n_positive":    int(positive_valid.sum()),
        "n_window":      n_window,
        "total_pred_mass": total_mass,
    }


def resolve_exp_candidates(exp_prefix: str, split: str) -> List[str]:
    candidates = [
        exp_prefix,
        exp_prefix.replace("v2_", f"{split}_").replace("exp1_fair_", f"{split}_"),
    ]
    if split != "pair":
        short = (exp_prefix
                 .replace("v2_abl_full",               f"{split}_circmac_nopt")
                 .replace("v2_pt_mlm",                 f"{split}_circmac_mlm")
                 .replace("v2_pt_pairing",             f"{split}_circmac_pairing")
                 .replace("v2_enc_lstm",               f"{split}_enc_lstm")
                 .replace("v2_enc_transformer",        f"{split}_enc_transformer")
                 .replace("v2_enc_mamba",              f"{split}_enc_mamba")
                 .replace("v2_enc_hymba",              f"{split}_enc_hymba")
                 .replace("exp1_fair_frozen_rnabert",  f"{split}_rnabert_frozen")
                 .replace("exp1_fair_frozen_rnaernie", f"{split}_rnaernie_frozen")
                 .replace("exp1_fair_frozen_rnamsm",   f"{split}_rnamsm_frozen")
                 .replace("exp1_fair_frozen_rnafm",    f"{split}_rnafm_frozen")
                 .replace("exp1_fair_trainable_rnabert",  f"{split}_rnabert_ft")
                 .replace("exp1_fair_trainable_rnaernie", f"{split}_rnaernie_ft")
                 .replace("exp1_fair_trainable_rnamsm",   f"{split}_rnamsm_ft")
                 .replace("exp1_fair_trainable_rnafm",    f"{split}_rnafm_ft"))
        candidates.append(short)
    return list(dict.fromkeys(candidates))


# -----------------------------------------------------------------------------
# Inference and score aggregation
# -----------------------------------------------------------------------------
def run_localization(trainer, dataset, radii, metadata_df=None) -> pd.DataFrame:
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    rows = []

    with torch.no_grad():
        for sample_idx, data in enumerate(loader):
            target, target_mask = trainer.forward_target(data)
            emb, _  = trainer.forward(data)
            emb, _  = trainer.forward_cross_attention(emb, target, target_mask)
            logits  = trainer.forward_task(emb, target, task="sites")
            prob_t  = logits_or_probs_to_probs(logits)

            if prob_t.ndim != 2 or prob_t.shape[0] != 1:
                raise ValueError(f"Expected [1, L] site probs, got {tuple(prob_t.shape)}")

            probs  = prob_t[0].detach().cpu().numpy().astype(np.float64)
            labels = align_sites_to_predictions(data["sites"][0], len(probs))

            base = {"sample_idx": sample_idx}
            if metadata_df is not None and sample_idx < len(metadata_df):
                row_meta = metadata_df.iloc[sample_idx]
                for col in ["isoform_ID", "circRNA_ID", "miRNA_ID", "BSJ_ID", "length"]:
                    if col in metadata_df.columns:
                        base[col] = row_meta[col]

            for radius in radii:
                m = compute_localization_metrics(labels, probs, int(radius))
                if m is not None:
                    rows.append({**base, "radius": int(radius), **m})

    return pd.DataFrame(rows)


def prepare_test_dataframe(df_test: pd.DataFrame) -> pd.DataFrame:
    df = df_test.copy()
    if "length" not in df.columns:
        if "circRNA" not in df.columns:
            raise KeyError("Test dataframe needs 'length' or 'circRNA' column")
        df["length"] = df["circRNA"].astype(str).str.len()

    if "binding" in df.columns:
        before = len(df)
        df = df[df["binding"] == 1].copy()
        if len(df) != before:
            print(f"  Positive pairs only: {before:,} -> {len(df):,}")

    mirna_csv = ROOT / "data" / "binding_miRNA_seq.csv"
    if "miRNA" not in df.columns:
        if mirna_csv.exists():
            df_mirna  = pd.read_csv(mirna_csv)
            mirna_map = df_mirna.groupby("miRNA_ID")["miRNA"].first().to_dict()
            df["miRNA"] = df["miRNA_ID"].map(mirna_map).fillna("")
        else:
            df["miRNA"] = ""

    return df.reset_index(drop=True)


def summarize_seed_scores(pair_scores: pd.DataFrame) -> pd.DataFrame:
    if pair_scores.empty:
        return pd.DataFrame()
    return (pair_scores.groupby("radius", as_index=False)
            .agg(n_pairs          =("sample_idx",        "nunique"),
                 lpm_mean         =("lpm",               "mean"),
                 lpm_sd_pairs     =("lpm",               "std"),
                 uniform_mean     =("uniform_mass",      "mean"),
                 normalized_mean  =("normalized_score",  "mean"),
                 normalized_sd_pairs=("normalized_score","std")))


def run_split(split, seeds, device, radii, max_eval_len, out_dir) -> Optional[pd.DataFrame]:
    _, test_file = SPLIT_FILES[split]
    test_path = ROOT / test_file
    if not test_path.exists():
        print(f"  [SKIP] {test_path} not found")
        return None

    df_test = prepare_test_dataframe(pd.read_pickle(test_path))

    print(f"\n{'='*78}")
    print(f"  Split: {split} | annotated pairs={len(df_test):,} | radii={radii} | max_eval_len={max_eval_len}")
    print(f"{'='*78}")

    seed_summary_rows, all_pair_rows = [], []

    for label, model_name, exp_prefix, frozen in MODEL_SPECS:
        native_limit    = MODEL_DATASET_MAX_LEN.get(model_name, MAX_DATASET_LEN)
        effective_limit = min(int(max_eval_len), int(native_limit), MAX_DATASET_LEN)

        df_model = df_test[df_test["length"] <= effective_limit].reset_index(drop=True)
        if df_model.empty:
            print(f"  {label:<24} no eligible sequences (limit={effective_limit})")
            continue

        dataset_max_len = effective_limit + SPECIAL_TOKEN_ALLOWANCE
        print(f"\n  {label} | limit={effective_limit} | eligible pairs={len(df_model):,}")

        model_seed_summaries = []

        for seed in seeds:
            exp_name = None
            for cand in resolve_exp_candidates(exp_prefix, split):
                if get_ckpt(model_name, cand, seed) is not None:
                    exp_name = cand
                    break
            if exp_name is None:
                print(f"    [seed {seed}] NO CKPT FOUND")
                continue

            try:
                ckpt       = get_ckpt(model_name, exp_name, seed)
                vocab_size = get_vocab_size(ckpt)
                ds         = make_dataset(df_model, dataset_max_len, vocab_size)
                tr         = build_trainer(model_name, exp_name, seed, device, frozen, ds.vocab_size)

                pair_scores = run_localization(tr, ds, radii, metadata_df=df_model)
                if pair_scores.empty:
                    print(f"    [seed {seed}] no valid annotated-site pairs")
                else:
                    pair_scores.insert(0, "seed",  seed)
                    pair_scores.insert(0, "model", label)
                    pair_scores.insert(0, "split", split)
                    pair_scores["effective_max_len"] = effective_limit
                    all_pair_rows.append(pair_scores)

                    seed_summary = summarize_seed_scores(pair_scores)
                    seed_summary.insert(0, "seed",  seed)
                    seed_summary.insert(0, "model", label)
                    seed_summary.insert(0, "split", split)
                    seed_summary["effective_max_len"] = effective_limit
                    model_seed_summaries.append(seed_summary)

                    for _, r in seed_summary.iterrows():
                        print(f"    s{seed} r={int(r['radius']):>2}: "
                              f"LPM={r['lpm_mean']:.4f} | "
                              f"uniform={r['uniform_mean']:.4f} | "
                              f"NLS={r['normalized_mean']:.4f} | "
                              f"n={int(r['n_pairs']):,}")

                del tr
                torch.cuda.empty_cache()

            except Exception as exc:
                print(f"    [{label} s{seed}] ERROR: {exc}")

        if model_seed_summaries:
            seed_df = pd.concat(model_seed_summaries, ignore_index=True)
            for radius, group in seed_df.groupby("radius"):
                seed_summary_rows.append({
                    "split":              split,
                    "model":              label,
                    "radius":             int(radius),
                    "effective_max_len":  effective_limit,
                    "n_seeds":            int(group["seed"].nunique()),
                    "n_pairs_per_seed":   int(group["n_pairs"].min()),
                    "lpm_mean":           float(group["lpm_mean"].mean()),
                    "lpm_std_seed":       float(group["lpm_mean"].std(ddof=0)),
                    "uniform_mean":       float(group["uniform_mean"].mean()),
                    "normalized_mean":    float(group["normalized_mean"].mean()),
                    "normalized_std_seed":float(group["normalized_mean"].std(ddof=0)),
                })

    if not seed_summary_rows:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)

    summary_df  = pd.DataFrame(seed_summary_rows)
    summary_path = out_dir / f"localization_focus_{split}_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  Saved summary -> {summary_path}")

    if all_pair_rows:
        pair_df   = pd.concat(all_pair_rows, ignore_index=True)
        pair_path = out_dir / f"localization_focus_{split}_per_pair.csv"
        pair_df.to_csv(pair_path, index=False)
        print(f"  Saved per-pair  -> {pair_path}")

    return summary_df


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",   type=str, default="pair",
                        choices=["pair", "bsj", "iso", "all"])
    parser.add_argument("--device",  type=int, default=0)
    parser.add_argument("--radii",   type=int, nargs="+", default=[0, 10, 20],
                        help="Expansion radii in nt around annotated binding positions (default: 0 10 20)")
    parser.add_argument("--seeds",   type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--max_eval_len", type=int, default=1000,
                        help="Global sequence-length cap; model-native limits also enforced (default: 1000)")
    args = parser.parse_args()

    if any(r < 0 for r in args.radii):
        raise ValueError("All radii must be >= 0")

    device  = get_device(args.device)
    out_dir = ROOT / "figures_paper" / "fig_localization_focus"
    splits  = ["pair", "bsj", "iso"] if args.split == "all" else [args.split]

    all_summaries = []
    for split in splits:
        df = run_split(split=split, seeds=args.seeds, device=device,
                       radii=args.radii, max_eval_len=args.max_eval_len,
                       out_dir=out_dir)
        if df is not None:
            all_summaries.append(df)

    if all_summaries and len(splits) > 1:
        combined = pd.concat(all_summaries, ignore_index=True)
        combined.to_csv(out_dir / "localization_focus_all_summary.csv", index=False)
        print(f"\nSaved combined -> {out_dir / 'localization_focus_all_summary.csv'}")


if __name__ == "__main__":
    main()
