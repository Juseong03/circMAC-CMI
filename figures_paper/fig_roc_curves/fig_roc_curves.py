#!/usr/bin/env python3
"""
fig_roc_curves.py — ROC Curve figures

fig1b_roc_rna_lm.{pdf,png}  — Fig 1b (main): fine-tuned RNA-LMs vs CircMAC+Pairing
fig1b_roc_rna_lm_full.{pdf,png} — Fig 1b extended: frozen + fine-tuned + CircMAC+Pairing (2 panels)
fig2b_roc_encoder.{pdf,png}  — Fig 2b (ablation): encoder architecture ROC curves

Reads cached prediction data produced by compute_roc_data.py.

Usage:
    python figures_paper/fig_roc_curves/fig_roc_curves.py
"""

import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc

OUT   = Path(__file__).resolve().parent
CACHE = OUT

# ── Colors ────────────────────────────────────────────────────────────────────
LM_COLORS = {
    "RNABERT":  "#4878CF",
    "RNAErnie": "#9467BD",
    "RNAMSM":   "#2CA02C",
    "RNA-FM":   "#17BECF",
}
PROPOSED_COLOR  = "#E05C2A"
NOPT_COLOR      = "#BCBD22"
ENCODER_COLORS = {
    "LSTM":        "#E377C2",
    "Transformer": "#8C564B",
    "Mamba":       "#D62728",
    "Hymba":       "#BCBD22",
    "CircMAC":     "#E05C2A",
}
PRETRAINED_COLORS = {
    "RNABERT":           "#4878CF",
    "RNAErnie":          "#9467BD",
    "RNAMSM":            "#2CA02C",
    "RNA-FM":            "#17BECF",
    "CircMAC (Pairing)": "#E05C2A",
    "CircMAC (NoPT)":    "#BCBD22",
}

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      9,
    "axes.linewidth": 0.9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "pdf.fonttype":   42,
    "ps.fonttype":    42,
})


def mean_roc(seed_results):
    """Interpolate per-seed ROC onto common FPR grid → mean ± std."""
    base_fpr = np.linspace(0, 1, 300)
    tprs, aucs = [], []
    for r in seed_results:
        fpr, tpr, _ = roc_curve(r["labels"], r["preds"])
        tprs.append(np.interp(base_fpr, fpr, tpr))
        aucs.append(auc(fpr, tpr))
    mean_tpr = np.mean(tprs, axis=0)
    std_tpr  = np.std(tprs, axis=0)
    return base_fpr, mean_tpr, std_tpr, np.mean(aucs), np.std(aucs)


def plot_roc_panel(ax, cache, entries, title):
    """
    entries: list of (label_in_cache, display_label, color, linestyle, linewidth)
    """
    for cache_key, disp_label, color, ls, lw in entries:
        if cache_key not in cache or not cache[cache_key]:
            print(f"  [WARN] No data: {cache_key}")
            continue

        base_fpr, mean_tpr, std_tpr, mean_auc, std_auc = mean_roc(cache[cache_key])
        alpha_band = 0.14 if lw > 1.8 else 0.08

        ax.plot(base_fpr, mean_tpr,
                color=color, lw=lw, ls=ls, zorder=3,
                label=f"{disp_label}  ({mean_auc:.3f}±{std_auc:.3f})")
        ax.fill_between(base_fpr,
                        np.clip(mean_tpr - std_tpr, 0, 1),
                        np.clip(mean_tpr + std_tpr, 0, 1),
                        color=color, alpha=alpha_band, zorder=2)

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.35, zorder=1)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("False Positive Rate", fontsize=9)
    ax.set_ylabel("True Positive Rate", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.grid(True, linestyle="--", alpha=0.3, zorder=0)
    ax.legend(loc="lower right", fontsize=7.5, frameon=True,
              framealpha=0.92, edgecolor="#cccccc", handlelength=2.2)


# ── Fig 1b: RNA-LM ROC (main, fine-tuned only + CircMAC+Pairing) ─────────────
def make_fig1b_rna_lm():
    cache_path = CACHE / "roc_cache_rna_lm.pkl"
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache not found: {cache_path}\nRun compute_roc_data.py first.")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    entries = [
        ("RNABERT (fine-tuned)",  "RNABERT",      LM_COLORS["RNABERT"],  "-",  1.3),
        ("RNAErnie (fine-tuned)", "RNAErnie",      LM_COLORS["RNAErnie"], "-",  1.3),
        ("RNAMSM (fine-tuned)",   "RNAMSM",        LM_COLORS["RNAMSM"],  "-",  1.3),
        ("RNA-FM (fine-tuned)",   "RNA-FM",        LM_COLORS["RNA-FM"],  "-",  1.3),
        ("CircMAC+Pairing",       "CircMAC+Pairing (Ours)", PROPOSED_COLOR, "-", 2.4),
    ]

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    plot_roc_panel(ax, cache, entries,
                   "ROC Curves — RNA-LM Comparison\n(Fine-tuned encoders)")
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        p = OUT / f"fig1b_roc_rna_lm.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"Saved → {p}")
    plt.close(fig)


# ── Fig 1b (full): frozen + fine-tuned + CircMAC+Pairing (2 panels) ──────────
def make_fig1b_rna_lm_full():
    cache_path = CACHE / "roc_cache_rna_lm.pkl"
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache not found: {cache_path}")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    lm_names = ["RNABERT", "RNAErnie", "RNAMSM", "RNA-FM"]

    entries_frozen = [
        (f"{n} (frozen)", n, LM_COLORS[n], "--", 1.3) for n in lm_names
    ] + [("CircMAC+Pairing", "CircMAC+Pairing (Ours)", PROPOSED_COLOR, "-", 2.4)]

    entries_ft = [
        (f"{n} (fine-tuned)", n, LM_COLORS[n], "-", 1.3) for n in lm_names
    ] + [("CircMAC+Pairing", "CircMAC+Pairing (Ours)", PROPOSED_COLOR, "-", 2.4)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0),
                              gridspec_kw={"wspace": 0.28})
    fig.suptitle("ROC Curves — RNA Language Model Comparison",
                 fontsize=11, fontweight="bold", y=1.01)

    plot_roc_panel(axes[0], cache, entries_frozen,  "(a) Frozen encoders")
    plot_roc_panel(axes[1], cache, entries_ft,       "(b) Fine-tuned encoders")

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        p = OUT / f"fig1b_roc_rna_lm_full.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"Saved → {p}")
    plt.close(fig)


# ── Fig 2b: Encoder ROC ───────────────────────────────────────────────────────
def make_fig2b_encoder():
    cache_path = CACHE / "roc_cache_encoder.pkl"
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache not found: {cache_path}")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    entries = [
        ("LSTM",        "LSTM",        ENCODER_COLORS["LSTM"],        "-", 1.3),
        ("Transformer", "Transformer", ENCODER_COLORS["Transformer"], "-", 1.3),
        ("Mamba",       "Mamba",       ENCODER_COLORS["Mamba"],       "-", 1.3),
        ("Hymba",       "Hymba",       ENCODER_COLORS["Hymba"],       "-", 1.3),
        ("CircMAC",     "CircMAC (Ours)", ENCODER_COLORS["CircMAC"],  "-", 2.4),
    ]

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    plot_roc_panel(ax, cache, entries, "ROC Curves — Encoder Architecture")
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        p = OUT / f"fig2b_roc_encoder.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"Saved → {p}")
    plt.close(fig)


# ── Fig: Pretrained comparison (fine-tuned RNA-LMs + CircMAC Pairing + NoPT) ──
def make_fig_pretrained():
    cache_path = CACHE / "roc_cache_pretrained.pkl"
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache not found: {cache_path}\nRun compute_roc_data.py --group pretrained first.")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    entries = []
    model_order = [
        ("RNABERT",            LM_COLORS["RNABERT"],            "-",  1.3),
        ("RNAErnie",           LM_COLORS["RNAErnie"],           "-",  1.3),
        ("RNAMSM",             LM_COLORS["RNAMSM"],             "-",  1.3),
        ("RNA-FM",             LM_COLORS["RNA-FM"],             "-",  1.3),
        ("CircMAC (NoPT)",     NOPT_COLOR,                      "--", 1.6),
        ("CircMAC (Pairing)",  PROPOSED_COLOR,                  "-",  2.4),
    ]
    for key, color, ls, lw in model_order:
        entries.append((key, key, color, ls, lw))

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    plot_roc_panel(ax, cache, entries,
                   "ROC Curves — Pretrained Model Comparison\n(Fine-tuned RNA-LMs vs CircMAC)")
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        p = OUT / f"fig_roc_pretrained.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"Saved → {p}")
    plt.close(fig)


def main():
    print("=== Fig: Pretrained model comparison ===")
    try:
        make_fig_pretrained()
    except FileNotFoundError as e:
        print(f"  Skip: {e}")

    print("\n=== Fig 1b: RNA-LM ROC ===")
    try:
        make_fig1b_rna_lm()
        make_fig1b_rna_lm_full()
    except FileNotFoundError as e:
        print(f"  Skip: {e}")

    print("\n=== Fig 2b: Encoder ROC ===")
    try:
        make_fig2b_encoder()
    except FileNotFoundError as e:
        print(f"  Skip: {e}")


if __name__ == "__main__":
    main()
