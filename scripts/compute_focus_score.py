#!/usr/bin/env python3
"""
compute_focus_score.py
======================
Compute the BSJ Focus Score for each model on the test set.

Definition (per circRNA–miRNA pair):

    Focus score = sum_{i in B_{±k}} ŷ_i  /  sum_{i=1}^{L} ŷ_i

where B_{±k} = first k and last k nucleotide positions (BSJ-adjacent window).

The score is averaged across all pairs in the test set.
A high focus score means the model concentrates predictions near the BSJ.

Usage:
    python scripts/compute_focus_score.py --split pair  --device 0
    python scripts/compute_focus_score.py --split bsj   --device 0
    python scripts/compute_focus_score.py --split iso   --device 0
    python scripts/compute_focus_score.py --split all   --device 0  # all three splits
    python scripts/compute_focus_score.py --split pair --bsj_win 40 --seeds 1 2 3
"""

import argparse
import sys
from pathlib import Path

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

# ── Constants ──────────────────────────────────────────────────────────────────
MAX_LEN  = 1022
BSJ_WIN  = 40

MODEL_MAX_LEN = {
    'rnabert':  436,
    'rnaernie': 511,
}

# (label, model_name, exp_prefix, frozen)
MODEL_SPECS = [
    # CircMAC
    ("CircMAC (NoPT)",     "circmac",     "v2_abl_full",                 False),
    ("CircMAC (MLM)",      "circmac",     "v2_pt_mlm",                   False),
    ("CircMAC (Pairing)",  "circmac",     "v2_pt_pairing",               False),
    # Encoders
    ("LSTM",               "lstm",        "v2_enc_lstm",                 False),
    ("Transformer",        "transformer", "v2_enc_transformer",          False),
    ("Mamba",              "mamba",       "v2_enc_mamba",                False),
    ("Hymba",              "hymba",       "v2_enc_hymba",                False),
    # RNA-LMs (frozen)
    ("RNABERT (frozen)",   "rnabert",     "exp1_fair_frozen_rnabert",    True),
    ("RNAErnie (frozen)",  "rnaernie",    "exp1_fair_frozen_rnaernie",   True),
    ("RNAMSM (frozen)",    "rnamsm",      "exp1_fair_frozen_rnamsm",     True),
    ("RNA-FM (frozen)",    "rnafm",       "exp1_fair_frozen_rnafm",      True),
    # RNA-LMs (fine-tuned)
    ("RNABERT (ft)",       "rnabert",     "exp1_fair_trainable_rnabert", False),
    ("RNAErnie (ft)",      "rnaernie",    "exp1_fair_trainable_rnaernie",False),
    ("RNAMSM (ft)",        "rnamsm",      "exp1_fair_trainable_rnamsm",  False),
    ("RNA-FM (ft)",        "rnafm",       "exp1_fair_trainable_rnafm",   False),
]

SPLIT_FILES = {
    "pair": ("data/df_train_final.pkl",        "data/df_test_final.pkl"),
    "bsj":  ("data/df_train_bsj_disjoint.pkl", "data/df_test_bsj_disjoint.pkl"),
    "iso":  ("data/df_train_iso_disjoint.pkl", "data/df_test_iso_disjoint.pkl"),
}

# exp_prefix suffix per split (how saved_models are named on server)
EXP_SUFFIX = {
    "pair": "",         # e.g. v2_abl_full_s1
    "bsj":  "_bsj",    # e.g. bsj_circmac_nopt_s1  ← adjust if naming differs
    "iso":  "_iso",
}


def bsj_flag(length: int, win: int) -> np.ndarray:
    f = np.zeros(length, dtype=bool)
    f[:win] = True
    f[max(0, length - win):] = True
    return f


def get_ckpt(model_name: str, exp_name: str, seed: int) -> Path:
    """Try common checkpoint locations."""
    candidates = [
        ROOT / "saved_models" / model_name / f"{exp_name}_s{seed}" / str(seed) / "train" / "model.pth",
        ROOT / "saved_models" / model_name / f"{exp_name}_s{seed}" / "train" / "model.pth",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def get_vocab_size(ckpt_path: Path):
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        key = "embedding.word_embeddings.weight"
        if key in ckpt:
            return ckpt[key].shape[0]
    except Exception:
        pass
    return None


def make_dataset(df, max_len, vocab_size_override=None):
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

    cfg = get_model_config(model_name=model_name, d_model=128, n_layer=6,
                           vocab_size=vocab_size)
    if model_name in ["rnabert", "rnaernie", "rnafm", "rnamsm"]:
        cfg.trainable = not frozen

    tr.define_model(cfg, model_name=model_name, pretrain=False,
                    is_cross_attention=True, interaction="cross_attention",
                    site_head_type="conv1d")

    if model_name in ["rnabert", "rnaernie", "rnafm", "rnamsm"] and frozen:
        tr.define_pretrained_model(model_name=model_name)

    tr.set_pretrained_target(target="mirna", rna_model="rnabert")

    ckpt = get_ckpt(model_name, exp_name, seed)
    if ckpt:
        tr.load_model_from_path(str(ckpt), verbose=False)
    else:
        tr.load_model(epoch=None, pretrain=False, verbose=False)
    tr.model.eval()
    return tr


def run_focus(trainer, dataset, bsj_win: int):
    """Return list of (focus_score, total_pred_sum) per pair."""
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    scores = []
    with torch.no_grad():
        for data in loader:
            target, target_mask = trainer.forward_target(data)
            emb, mask = trainer.forward(data)
            emb, _   = trainer.forward_cross_attention(emb, target, target_mask)
            logits   = trainer.forward_task(emb, target, task="sites")
            prob     = torch.softmax(logits, dim=-1)[..., 1]

            L      = int(data["length"][0].item())
            n_pred = prob.shape[1]
            pred   = np.full(L, np.nan)
            pred[:min(L, n_pred)] = prob[0, :min(L, n_pred)].cpu().numpy()

            # ignore NaN positions (beyond model max_len)
            valid = ~np.isnan(pred)
            pred_valid = pred.copy()
            pred_valid[~valid] = 0.0

            flag  = bsj_flag(L, bsj_win)
            bsj_sum   = pred_valid[flag].sum()
            total_sum = pred_valid.sum()

            focus = bsj_sum / total_sum if total_sum > 1e-9 else np.nan
            scores.append(focus)
    return scores


def run_split(split: str, seeds: list, device, bsj_win: int, out_dir: Path):
    _, test_file = SPLIT_FILES[split]
    test_path = ROOT / test_file
    if not test_path.exists():
        print(f"  [SKIP] {test_path} not found")
        return None

    df_test = pd.read_pickle(test_path)

    # Add miRNA sequences
    mirna_csv = ROOT / "data" / "binding_miRNA_seq.csv"
    if mirna_csv.exists():
        df_mirna = pd.read_csv(mirna_csv)
        mirna_map = df_mirna.groupby("miRNA_ID")["miRNA"].first().to_dict()
        df_test["miRNA"] = df_test["miRNA_ID"].map(mirna_map).fillna("")
    else:
        df_test["miRNA"] = ""

    print(f"\n{'='*60}")
    print(f"  Split: {split}  |  N={len(df_test)}  |  BSJ_WIN={bsj_win}")
    print(f"{'='*60}")

    rows = []
    for label, model_name, exp_prefix, frozen in MODEL_SPECS:
        model_max_len = MODEL_MAX_LEN.get(model_name, MAX_LEN) + 2

        seed_scores = []
        for seed in seeds:
            # Build exp_name based on split suffix conventions
            # Try multiple naming conventions
            exp_candidates = [
                exp_prefix,                                      # pair split: same prefix
                exp_prefix.replace("v2_", f"{split}_").replace("exp1_fair_", f"{split}_"),
            ]
            # For bsj/iso splits, also try direct naming like bsj_circmac_nopt
            if split != "pair":
                short = (exp_prefix
                         .replace("v2_abl_full", f"{split}_circmac_nopt")
                         .replace("v2_pt_mlm",   f"{split}_circmac_mlm")
                         .replace("v2_pt_pairing", f"{split}_circmac_pairing")
                         .replace("v2_enc_lstm",  f"{split}_enc_lstm")
                         .replace("v2_enc_transformer", f"{split}_enc_transformer")
                         .replace("v2_enc_mamba", f"{split}_enc_mamba")
                         .replace("v2_enc_hymba", f"{split}_enc_hymba")
                         .replace("exp1_fair_frozen_rnabert",     f"{split}_rnabert_frozen")
                         .replace("exp1_fair_frozen_rnaernie",    f"{split}_rnaernie_frozen")
                         .replace("exp1_fair_frozen_rnamsm",      f"{split}_rnamsm_frozen")
                         .replace("exp1_fair_frozen_rnafm",       f"{split}_rnafm_frozen")
                         .replace("exp1_fair_trainable_rnabert",  f"{split}_rnabert_ft")
                         .replace("exp1_fair_trainable_rnaernie", f"{split}_rnaernie_ft")
                         .replace("exp1_fair_trainable_rnamsm",   f"{split}_rnamsm_ft")
                         .replace("exp1_fair_trainable_rnafm",    f"{split}_rnafm_ft"))
                exp_candidates.append(short)

            # Find working exp_name
            exp_name = None
            for cand in exp_candidates:
                if get_ckpt(model_name, cand, seed):
                    exp_name = cand
                    break
            if exp_name is None:
                continue

            try:
                ckpt  = get_ckpt(model_name, exp_name, seed)
                vs    = get_vocab_size(ckpt)
                ds    = make_dataset(df_test, model_max_len, vs)
                tr    = build_trainer(model_name, exp_name, seed, device, frozen, ds.vocab_size)
                sc    = run_focus(tr, ds, bsj_win)
                valid = [s for s in sc if not np.isnan(s)]
                if valid:
                    seed_scores.append(np.mean(valid))
                del tr
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"    [{label} s{seed}] ERROR: {e}")

        if seed_scores:
            mean_fs = np.mean(seed_scores)
            std_fs  = np.std(seed_scores)
            print(f"  {label:<22}  focus={mean_fs:.4f} ± {std_fs:.4f}  (n_seeds={len(seed_scores)})")
            rows.append(dict(split=split, model=label, bsj_win=bsj_win,
                             focus_mean=round(mean_fs, 6),
                             focus_std=round(std_fs, 6),
                             n_seeds=len(seed_scores)))
        else:
            print(f"  {label:<22}  NO CKPT FOUND")

    if rows:
        df_out = pd.DataFrame(rows)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / f"focus_score_{split}.csv"
        df_out.to_csv(out_csv, index=False)
        print(f"\n  Saved → {out_csv}")
        return df_out
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",   type=str, default="pair",
                        choices=["pair", "bsj", "iso", "all"])
    parser.add_argument("--device",  type=int, default=0)
    parser.add_argument("--bsj_win", type=int, default=BSJ_WIN,
                        help=f"BSJ window size k (default={BSJ_WIN})")
    parser.add_argument("--seeds",   type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()

    device  = get_device(args.device)
    out_dir = ROOT / "figures_paper" / "fig_focus_score"
    splits  = ["pair", "bsj", "iso"] if args.split == "all" else [args.split]

    all_rows = []
    for sp in splits:
        df = run_split(sp, args.seeds, device, args.bsj_win, out_dir)
        if df is not None:
            all_rows.append(df)

    if all_rows and len(splits) > 1:
        pd.concat(all_rows).to_csv(out_dir / "focus_score_all.csv", index=False)
        print(f"\nSaved combined → {out_dir / 'focus_score_all.csv'}")


if __name__ == "__main__":
    main()
