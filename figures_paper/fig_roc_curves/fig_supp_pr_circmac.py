#!/usr/bin/env python3
"""
fig_supp_pr_circmac.py
======================
Supplementary Figure: CircMAC-BPP PR curves
(A) Pair-disjoint  (B) Isoform-disjoint  (C) BSJ-disjoint

AP values computed with the exact same valid-position evaluation protocol
as the AUPRC values reported in the main tables.

Usage:
    python figures_paper/fig_roc_curves/fig_supp_pr_circmac.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

ROOT = Path(__file__).resolve().parents[2]
OUT  = Path(__file__).resolve().parent
LOGS = ROOT / "logs"
PRED = ROOT / "eval_results" / "preds"

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         9,
    "axes.linewidth":    0.9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
})

# CircMAC-BPP experiment names per split
# logs/{model}/{exp_name}_s{seed}/{seed}/best_preds/test_preds.pkl
SPLIT_CONFIG = {
    "pair": {
        "title":    "(A) Pair-disjoint",
        "exp_name": "max_circmac_pairing",
        "model":    "circmac",
    },
    "iso": {
        "title":    "(B) Isoform-disjoint",
        "exp_name": "bsj_circmac_pairing",   # adjust if naming differs on server
        "model":    "circmac",
    },
    "bsj": {
        "title":    "(C) BSJ-disjoint",
        "exp_name": "bsj_circmac_pairing",   # adjust if naming differs on server
        "model":    "circmac",
    },
}

SEEDS = [1, 2, 3]
COLOR = "#E05C2A"


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_logs_pkl(model, exp_name, seed):
    """logs/{model}/{exp_name}_s{seed}/{seed}/best_preds/test_preds.pkl"""
    p = LOGS / model / f"{exp_name}_s{seed}" / str(seed) / "best_preds" / "test_preds.pkl"
    if not p.exists():
        return None, None, str(p)
    with open(p, "rb") as f:
        d = pickle.load(f)
    labels = d["labels_sites"].reshape(-1).numpy()
    probs  = d["probs_sites"].reshape(-1).numpy()
    valid  = labels != -100
    return labels[valid].astype(int), probs[valid].astype(float), str(p)


def load_eval_pkl(exp_name, seed):
    """eval_results/preds/{exp_name}_s{seed}/test_preds.pkl  (DataFrame format)"""
    p = PRED / f"{exp_name}_s{seed}" / "test_preds.pkl"
    if not p.exists():
        return None, None, str(p)
    df = pd.read_pickle(p)
    return df["label"].values.astype(int), df["prob"].values.astype(float), str(p)


def load_preds_for_split(split_cfg, seeds):
    """Try logs/ first, then eval_results/."""
    results = []
    exp_name = split_cfg["exp_name"]
    model    = split_cfg["model"]
    for seed in seeds:
        y, p, src = load_logs_pkl(model, exp_name, seed)
        if y is None:
            y, p, src = load_eval_pkl(exp_name, seed)
        if y is None:
            print(f"    [SKIP] seed={seed}  not found at {src}")
            continue
        ap = average_precision_score(y, p)
        print(f"    [OK] seed={seed}  n={len(y):,}  pos={y.mean():.4f}  AP={ap:.4f}  src={src}")
        results.append({"seed": seed, "labels": y, "preds": p})
    return results


# ── PR utils ──────────────────────────────────────────────────────────────────

def mean_pr(results):
    base_rec = np.linspace(0, 1, 500)
    precs, aps, prevs = [], [], []
    for r in results:
        y, p = r["labels"], r["preds"]
        prec, rec, _ = precision_recall_curve(y, p)
        prec, rec = prec[::-1], rec[::-1]  # ascending recall
        precs.append(np.interp(base_rec, rec, prec))
        aps.append(average_precision_score(y, p))
        prevs.append(y.mean())
    return (base_rec,
            np.mean(precs, 0), np.std(precs, 0),
            np.mean(aps), np.std(aps),
            np.mean(prevs))


# ── Figure ────────────────────────────────────────────────────────────────────

def main():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5),
                             gridspec_kw={"wspace": 0.30})

    any_data = False
    for ax, split in zip(axes, ["pair", "iso", "bsj"]):
        cfg = SPLIT_CONFIG[split]
        print(f"\n=== {cfg['title']} ({split}) ===")
        results = load_preds_for_split(cfg, SEEDS)

        ax.set_title(cfg["title"], fontsize=10, fontweight="bold", pad=6)
        ax.set_xlabel("Recall", fontsize=9)
        ax.set_ylabel("Precision", fontsize=9)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, linestyle="--", alpha=0.3, zorder=0)

        if not results:
            ax.text(0.5, 0.5, "No predictions found\n(upload from server)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="gray")
            continue

        any_data = True
        rec, mp, sp, map_, std_ap, prev = mean_pr(results)

        # Random baseline
        ax.axhline(prev, color="gray", lw=0.9, ls=":", alpha=0.6,
                   label=f"Random  (AP={prev:.3f})", zorder=2)

        # CircMAC-BPP
        ax.plot(rec, mp, color=COLOR, lw=2.2, zorder=3,
                label=f"CircMAC-BPP  (AP={map_:.3f}±{std_ap:.3f})")
        ax.fill_between(rec,
                        np.clip(mp - sp, 0, 1),
                        np.clip(mp + sp, 0, 1),
                        color=COLOR, alpha=0.15, zorder=2)

        ax.legend(loc="upper right", fontsize=7.5, frameon=True,
                  framealpha=0.92, edgecolor="#cccccc", handlelength=2.2)

    fig.suptitle("Supplementary Figure: CircMAC-BPP Precision–Recall Curves",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()

    for ext in ["pdf", "png"]:
        p = OUT / f"fig_supp_pr_circmac.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"\nSaved → {p}")
    plt.close(fig)

    if not any_data:
        print("\n[!] No data found for any split.")
        print("    Upload from server:")
        for split, cfg in SPLIT_CONFIG.items():
            for seed in SEEDS:
                print(f"    logs/circmac/{cfg['exp_name']}_s{seed}/{seed}/best_preds/test_preds.pkl")


if __name__ == "__main__":
    main()
