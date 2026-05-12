#!/usr/bin/env python3
"""
fig_roc_curves.py — ROC Curve figures for Fig1 (encoder) and Fig2 (RNA-LM)

Reads cached prediction data from compute_roc_data.py output.
Plots mean ROC curve ± std band across 3 seeds.

Output:
    fig_roc_encoder.{pdf,png}   — Fig1: encoder model comparison
    fig_roc_rna_lm.{pdf,png}    — Fig2: RNA-LM frozen vs fine-tuned vs circMAC
"""

import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc

OUT   = Path(__file__).resolve().parent
CACHE = OUT  # same directory

# ── Colors ───────────────────────────────────────────────────────────────────
MODEL_COLORS = {
    "circMAC (ours)":        "#FF7F0E",
    "LSTM":                  "#E377C2",
    "Transformer":           "#8C564B",
    "Mamba":                 "#D62728",
    "Hymba":                 "#BCBD22",
    "RNABERT (frozen)":      "#1F77B4",
    "RNAErnie (frozen)":     "#9467BD",
    "RNAMSM (frozen)":       "#2CA02C",
    "RNA-FM (frozen)":       "#17BECF",
    "RNABERT (fine-tuned)":  "#1F77B4",
    "RNAErnie (fine-tuned)": "#9467BD",
    "RNAMSM (fine-tuned)":   "#2CA02C",
    "RNA-FM (fine-tuned)":   "#17BECF",
}

# Line style: frozen = dashed, fine-tuned / encoder = solid
def get_ls(label):
    return "--" if "frozen" in label else "-"

def get_lw(label):
    return 2.4 if "circMAC" in label else 1.4

plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "font.size":    9,
    "axes.linewidth": 0.9,
    "pdf.fonttype": 42,
    "ps.fonttype":  42,
})


# ── Core: compute mean ROC across seeds ──────────────────────────────────────
def mean_roc(seed_results):
    """
    Interpolate per-seed ROC onto a common FPR grid and return
    mean_tpr, std_tpr, mean_auc.
    """
    base_fpr = np.linspace(0, 1, 300)
    tprs, aucs = [], []

    for r in seed_results:
        fpr, tpr, _ = roc_curve(r["labels"], r["preds"])
        tprs.append(np.interp(base_fpr, fpr, tpr))
        aucs.append(auc(fpr, tpr))

    mean_tpr = np.mean(tprs, axis=0)
    std_tpr  = np.std(tprs, axis=0)
    mean_auc = np.mean(aucs)
    std_auc  = np.std(aucs)

    return base_fpr, mean_tpr, std_tpr, mean_auc, std_auc


# ── Plot helpers ─────────────────────────────────────────────────────────────
def plot_roc_panel(ax, cache, model_order, title):
    """Draw ROC curves for models in model_order onto ax."""
    for label in model_order:
        if label not in cache or not cache[label]:
            print(f"  [WARN] No data for: {label}")
            continue

        base_fpr, mean_tpr, std_tpr, mean_auc, std_auc = mean_roc(cache[label])

        color = MODEL_COLORS.get(label, "#888888")
        ls    = get_ls(label)
        lw    = get_lw(label)
        alpha_band = 0.12 if "circMAC" in label else 0.08

        ax.plot(base_fpr, mean_tpr,
                color=color, lw=lw, ls=ls, zorder=3,
                label=f"{label}  (AUC={mean_auc:.3f}±{std_auc:.3f})")

        ax.fill_between(base_fpr,
                        np.clip(mean_tpr - std_tpr, 0, 1),
                        np.clip(mean_tpr + std_tpr, 0, 1),
                        color=color, alpha=alpha_band, zorder=2)

    # Diagonal
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4, zorder=1)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("False Positive Rate", fontsize=9)
    ax.set_ylabel("True Positive Rate", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=7)
    ax.grid(True, linestyle="--", alpha=0.3, zorder=0)

    legend = ax.legend(
        loc="lower right",
        fontsize=7.5,
        frameon=True,
        framealpha=0.92,
        edgecolor="#cccccc",
        handlelength=2.2,
    )
    return legend


# ── Fig1: Encoder ROC ─────────────────────────────────────────────────────────
def make_encoder_roc():
    cache_path = CACHE / "roc_cache_encoder.pkl"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Cache not found: {cache_path}\n"
            "Run compute_roc_data.py --group encoder first."
        )
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    model_order = ["LSTM", "Transformer", "Mamba", "Hymba", "circMAC (ours)"]

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    plot_roc_panel(ax, cache, model_order,
                   "ROC Curve — General Encoder Models")

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        p = OUT / f"fig_roc_encoder.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"Saved → {p}")
    plt.close(fig)


# ── Fig2: RNA-LM ROC (frozen vs fine-tuned, two panels) ──────────────────────
def make_rna_lm_roc():
    cache_path = CACHE / "roc_cache_rna_lm.pkl"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Cache not found: {cache_path}\n"
            "Run compute_roc_data.py --group rna_lm first."
        )
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    frozen_order    = ["RNABERT (frozen)", "RNAErnie (frozen)",
                       "RNAMSM (frozen)", "RNA-FM (frozen)", "circMAC (ours)"]
    finetune_order  = ["RNABERT (fine-tuned)", "RNAErnie (fine-tuned)",
                       "RNAMSM (fine-tuned)", "RNA-FM (fine-tuned)", "circMAC (ours)"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0),
                             gridspec_kw={"wspace": 0.28})

    plot_roc_panel(axes[0], cache, frozen_order,
                   "(a) Frozen RNA language models")
    plot_roc_panel(axes[1], cache, finetune_order,
                   "(b) Fine-tuned RNA language models")

    fig.suptitle("ROC Curves — RNA Language Model Comparison",
                 fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()

    for ext in ["pdf", "png"]:
        p = OUT / f"fig_roc_rna_lm.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"Saved → {p}")
    plt.close(fig)


def main():
    print("=== Encoder ROC ===")
    try:
        make_encoder_roc()
    except FileNotFoundError as e:
        print(f"  Skip: {e}")

    print("\n=== RNA-LM ROC ===")
    try:
        make_rna_lm_roc()
    except FileNotFoundError as e:
        print(f"  Skip: {e}")


if __name__ == "__main__":
    main()
