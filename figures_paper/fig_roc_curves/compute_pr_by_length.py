#!/usr/bin/env python3
"""
compute_pr_by_length.py
=======================
Run inference on length-stratified test subsets and compute PR curves.

Length groups (based on RNA-LM architecture limits):
  sub436 : L ≤ 436   — all models applicable (RNABERT native limit)
  sub511 : L ≤ 511   — RNABERT excluded      (RNAErnie native limit)
  full   : L ≤ 1022  — only CircMAC/RNAMSM/RNA-FM (no LM length limit)

Each group uses the SAME test sequences for all models → fair comparison.

Output:
    figures_paper/fig_roc_curves/pr_cache_by_length.pkl
    figures_paper/fig_roc_curves/fig_pr_by_length.{pdf,png}

Usage:
    python figures_paper/fig_roc_curves/compute_pr_by_length.py --device 0
    python figures_paper/fig_roc_curves/compute_pr_by_length.py --device 0 --no_plot
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from trainer import Trainer
from utils import prepare_datasets
from utils_config import get_model_config

OUT  = Path(__file__).resolve().parent
VIZ  = ROOT / "models_for_viz"

D_MODEL = 128
N_LAYER = 6
BS      = 32
WORKERS = 4
SEEDS   = [1, 2, 3]

# Length groups: (label, max_len_filter)
LENGTH_GROUPS = [
    ("≤436 nt",  436),
    ("≤511 nt",  511),
    ("≤1022 nt", 1022),
]

# (label, model_name, exp_template, trainable_pretrained, max_len_for_model)
#   max_len_for_model: model's architecture limit (tokenizer truncation)
ALL_MODELS = [
    ("CircMAC (NoPT)",    "circmac",  "v2_abl_full",                False, 1022),
    ("CircMAC (Pairing)", "circmac",  "v2_pt_pairing",              False, 1022),
    ("RNABERT",           "rnabert",  "exp1_fair_trainable_rnabert",True,  436),
    ("RNAErnie",          "rnaernie", "exp1_fair_trainable_rnaernie",True,  511),
    ("RNAMSM",            "rnamsm",   "exp1_fair_trainable_rnamsm", True,  1022),
    ("RNA-FM",            "rnafm",    "exp1_fair_trainable_rnafm",  True,  1022),
]

COLORS = {
    "CircMAC (NoPT)":    "#BCBD22",
    "CircMAC (Pairing)": "#E05C2A",
    "RNABERT":           "#4878CF",
    "RNAErnie":          "#9467BD",
    "RNAMSM":            "#2CA02C",
    "RNA-FM":            "#17BECF",
}

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         9,
    "axes.linewidth":    0.9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
})


# ── Data ──────────────────────────────────────────────────────────────────────
def load_test_df():
    df = pd.read_pickle(ROOT / "data/df_test_final.pkl")
    df["length"] = df["circRNA"].apply(len)
    df = df[df["binding"] == 1].reset_index(drop=True)
    return df


def build_dataset(df_test, df_train, max_len_filter, model_max_len):
    """Build test dataset filtered to max_len_filter, tokenized at model_max_len."""
    tok_len = model_max_len + 2  # +2 for CLS/SEP

    df_tr = df_train[df_train["length"] <= max_len_filter].reset_index(drop=True)
    df_te = df_test[df_test["length"]  <= max_len_filter].reset_index(drop=True)

    _, _, test_ds, _ = prepare_datasets(
        df=df_tr, df_test=df_te,
        max_len=tok_len, target="mirna", seed=1, kmer=1,
    )
    return test_ds, df_te


# ── Inference ─────────────────────────────────────────────────────────────────
def _vocab_size(model_path, model_name):
    try:
        ckpt  = torch.load(str(model_path), map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        if model_name in ["rnabert", "rnaernie", "rnafm", "rnamsm"]:
            return None
        key = "embedding.word_embeddings.weight"
        if key in state:
            return state[key].shape[0]
    except Exception:
        pass
    return None


def run_inference_one(model_name, exp_template, trainable_pretrained,
                      model_max_len, test_dataset, device):
    """Run all seeds; return list of {seed, preds, labels}."""
    results = []
    tok_len = model_max_len + 2

    for seed in SEEDS:
        exp        = f"{exp_template}_s{seed}"
        model_path = VIZ / model_name / exp / str(seed) / "train" / "model.pth"
        if not model_path.exists():
            print(f"    [SKIP] {exp}")
            continue

        vs  = _vocab_size(model_path, model_name)
        cfg = get_model_config(model_name=model_name, d_model=D_MODEL, n_layer=N_LAYER,
                               verbose=False, rc=False,
                               **({} if vs is None else {"vocab_size": vs}))
        if model_name in ["rnabert", "rnaernie", "rnafm", "rnamsm"]:
            cfg.trainable = trainable_pretrained

        trainer = Trainer(seed=seed, device=device,
                          experiment_name=exp, verbose=False)
        trainer.set_dataloader(test_dataset, part=2, batch_size=BS, num_workers=WORKERS)
        trainer.task = "sites"
        trainer.rc   = False

        trainer.define_model(config=cfg, model_name=model_name, pretrain=False,
                             is_cross_attention=True, interaction="cross_attention",
                             site_head_type="conv1d")
        if model_name in ["rnabert", "rnaernie", "rnafm", "rnamsm"] and not trainable_pretrained:
            trainer.define_pretrained_model(model_name=model_name)
        trainer.set_pretrained_target(target="mirna", rna_model="rnabert")
        trainer.load_model_from_path(str(model_path), verbose=False)
        trainer.model.eval()

        preds, labels = [], []
        with torch.no_grad():
            for batch in trainer.test_loader:
                tgt, tgt_mask = trainer.forward_target(batch)
                emb, _        = trainer.forward(batch)
                emb, _        = trainer.forward_cross_attention(emb, tgt, tgt_mask)
                logits        = trainer.forward_task(emb, tgt, task="sites")
                prob          = torch.softmax(logits, dim=-1)[..., 1]

                site_labels = batch["sites"]       # (B, L)
                lengths     = batch["length"]      # (B,)
                mask_valid  = site_labels != -100  # ignore pad

                for b in range(prob.shape[0]):
                    L = int(lengths[b].item())
                    preds.append(prob[b, :L].cpu().numpy())
                    labels.append(site_labels[b, :L].cpu().numpy())

        results.append({
            "seed":   seed,
            "preds":  np.concatenate(preds),
            "labels": np.concatenate(labels).astype(int),
        })
        del trainer
        torch.cuda.empty_cache()
        print(f"    [OK]   {exp}  n={len(np.concatenate(preds))}")

    return results


# ── Plot ──────────────────────────────────────────────────────────────────────
def mean_pr(seed_results):
    base_rec = np.linspace(0, 1, 300)
    precs, aps, prevs = [], [], []
    for r in seed_results:
        mask   = r["labels"] != -100
        labels = r["labels"][mask].astype(int)
        preds  = r["preds"][mask]
        p, rc, _ = precision_recall_curve(labels, preds)
        p, rc = p[::-1], rc[::-1]
        precs.append(np.interp(base_rec, rc, p))
        aps.append(auc(rc[::-1], p[::-1]))
        prevs.append(labels.mean())
    return (base_rec, np.mean(precs, 0), np.std(precs, 0),
            np.mean(aps), np.std(aps), np.mean(prevs))


def plot_panel(ax, cache, group_label, max_len_filter):
    """Plot one length-group panel."""
    baseline_drawn = False
    for label, model_name, _, _, model_max_len in ALL_MODELS:
        # skip models that can't handle this length group
        if model_max_len < max_len_filter:
            continue
        key = (label, max_len_filter)
        if key not in cache or not cache[key]:
            continue

        rec, mp, sp, map_, std_ap, prev = mean_pr(cache[key])
        color = COLORS.get(label, "#888888")
        lw    = 2.2 if "CircMAC" in label and "Pairing" in label else 1.4
        ls    = "--" if "NoPT" in label else "-"

        ax.plot(rec, mp, color=color, lw=lw, ls=ls, zorder=3,
                label=f"{label}  (AP={map_:.3f}±{std_ap:.3f})")
        ax.fill_between(rec,
                        np.clip(mp - sp, 0, 1),
                        np.clip(mp + sp, 0, 1),
                        color=color, alpha=0.10, zorder=2)

        if not baseline_drawn:
            ax.axhline(prev, color="gray", lw=0.8, ls=":", alpha=0.55,
                       label=f"Random  ({prev:.3f})")
            baseline_drawn = True

    n_pairs = cache.get(("__n_pairs__", max_len_filter), "?")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Recall", fontsize=9)
    ax.set_ylabel("Precision", fontsize=9)
    ax.set_title(f"Length {group_label}\n(N={n_pairs} pairs)", fontsize=10,
                 fontweight="bold", pad=5)
    ax.grid(True, linestyle="--", alpha=0.3, zorder=0)
    ax.legend(loc="upper right", fontsize=7.5, frameon=True,
              framealpha=0.92, edgecolor="#cccccc", handlelength=2.2)


def make_figure(cache):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2),
                              gridspec_kw={"wspace": 0.30})
    fig.suptitle("Precision-Recall Curves by Sequence Length (Pair Split)",
                 fontsize=12, fontweight="bold", y=1.02)

    for ax, (group_label, max_len) in zip(axes, LENGTH_GROUPS):
        plot_panel(ax, cache, group_label, max_len)

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        p = OUT / f"fig_pr_by_length.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"Saved → {p}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",   type=int, default=0)
    parser.add_argument("--no_plot",  action="store_true")
    parser.add_argument("--no_cache", action="store_true",
                        help="Re-run inference even if cache exists")
    args = parser.parse_args()

    from utils import get_device
    device = get_device(args.device)

    cache_path = OUT / "pr_cache_by_length.pkl"

    # ── Load or build cache ───────────────────────────────────────────────────
    if cache_path.exists() and not args.no_cache:
        print(f"Loading cache: {cache_path}")
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
    else:
        df_test  = load_test_df()
        df_train = pd.read_pickle(ROOT / "data/df_train_final.pkl")
        df_train["length"] = df_train["circRNA"].apply(len)
        df_train = df_train[df_train["binding"] == 1].reset_index(drop=True)

        cache = {}
        for group_label, max_len_filter in LENGTH_GROUPS:
            print(f"\n{'='*60}")
            print(f"  Length group: {group_label}  (max_len={max_len_filter})")

            df_te_grp = df_test[df_test["length"] <= max_len_filter]
            n_pairs   = len(df_te_grp)
            cache[("__n_pairs__", max_len_filter)] = n_pairs
            print(f"  Test pairs in group: {n_pairs}")

            for label, model_name, exp_tmpl, trainable, model_max_len in ALL_MODELS:
                if model_max_len < max_len_filter:
                    print(f"  [{label}] SKIP (model_max={model_max_len} < {max_len_filter})")
                    continue

                print(f"\n  [{label}]")
                test_ds, _ = build_dataset(df_test, df_train,
                                           max_len_filter, model_max_len)
                results = run_inference_one(model_name, exp_tmpl, trainable,
                                            model_max_len, test_ds, device)
                cache[(label, max_len_filter)] = results

        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)
        print(f"\nCache saved → {cache_path}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    if not args.no_plot:
        make_figure(cache)


if __name__ == "__main__":
    main()
