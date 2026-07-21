#!/usr/bin/env python3
"""
fig_pr_curves.py — Precision-Recall Curve figures (pair split)

Reads the same cache files as fig_roc_curves.py (produced by compute_roc_data.py).

Output:
    fig_pr_encoder.{pdf,png}    — encoder architecture comparison
    fig_pr_pretrained.{pdf,png} — fine-tuned RNA-LMs vs CircMAC

Usage:
    python figures_paper/fig_roc_curves/fig_pr_curves.py
"""

import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import precision_recall_curve, average_precision_score

OUT   = Path(__file__).resolve().parent
CACHE = OUT

# ── Colors (same as fig_roc_curves.py) ───────────────────────────────────────
LM_COLORS = {
    "RNABERT":  "#4878CF",
    "RNAErnie": "#9467BD",
    "RNAMSM":   "#2CA02C",
    "RNA-FM":   "#17BECF",
}
PROPOSED_COLOR = "#E05C2A"
NOPT_COLOR     = "#BCBD22"
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
    "font.family":       "DejaVu Sans",
    "font.size":         9,
    "axes.linewidth":    0.9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
})


def mean_pr(seed_results):
    """Interpolate per-seed PR curves onto common recall grid → mean ± std."""
    base_recall = np.linspace(0, 1, 300)
    precs, aps, prevalences = [], [], []
    for r in seed_results:
        prec, rec, _ = precision_recall_curve(r["labels"], r["preds"])
        # precision_recall_curve returns decreasing recall — reverse for interp
        prec, rec = prec[::-1], rec[::-1]
        precs.append(np.interp(base_recall, rec, prec))
        aps.append(average_precision_score(r["labels"], r["preds"]))
        prevalences.append(r["labels"].mean())
    mean_prec = np.mean(precs, axis=0)
    std_prec  = np.std(precs,  axis=0)
    return base_recall, mean_prec, std_prec, np.mean(aps), np.std(aps), np.mean(prevalences)


def plot_pr_panel(ax, cache, entries, title):
    """
    entries: list of (cache_key, display_label, color, linestyle, linewidth)
    """
    baseline_drawn = False
    for cache_key, disp_label, color, ls, lw in entries:
        if cache_key not in cache or not cache[cache_key]:
            print(f"  [WARN] No data: {cache_key}")
            continue

        rec, mean_prec, std_prec, mean_ap, std_ap, prevalence = mean_pr(cache[cache_key])
        alpha_band = 0.14 if lw > 1.8 else 0.08

        ax.plot(rec, mean_prec,
                color=color, lw=lw, ls=ls, zorder=3,
                label=f"{disp_label}  (AP={mean_ap:.3f}±{std_ap:.3f})")
        ax.fill_between(rec,
                        np.clip(mean_prec - std_prec, 0, 1),
                        np.clip(mean_prec + std_prec, 0, 1),
                        color=color, alpha=alpha_band, zorder=2)

        if not baseline_drawn:
            ax.axhline(prevalence, color="gray", lw=0.9, ls=":", alpha=0.6,
                       label=f"Random  ({prevalence:.3f})")
            baseline_drawn = True

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Recall", fontsize=9)
    ax.set_ylabel("Precision", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.grid(True, linestyle="--", alpha=0.3, zorder=0)
    ax.legend(loc="upper right", fontsize=7.5, frameon=True,
              framealpha=0.92, edgecolor="#cccccc", handlelength=2.2)


# ── Encoder PR ────────────────────────────────────────────────────────────────
def make_pr_encoder():
    cache_path = CACHE / "roc_cache_encoder.pkl"
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache not found: {cache_path}")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    entries = [
        ("LSTM",        "LSTM",           ENCODER_COLORS["LSTM"],        "-", 1.3),
        ("Transformer", "Transformer",    ENCODER_COLORS["Transformer"], "-", 1.3),
        ("Mamba",       "Mamba",          ENCODER_COLORS["Mamba"],       "-", 1.3),
        ("Hymba",       "Hymba",          ENCODER_COLORS["Hymba"],       "-", 1.3),
        ("CircMAC",     "CircMAC (Ours)", ENCODER_COLORS["CircMAC"],     "-", 2.4),
    ]

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    plot_pr_panel(ax, cache, entries, "PR Curves — Encoder Architecture")
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        p = OUT / f"fig_pr_encoder.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"Saved → {p}")
    plt.close(fig)


# ── Pretrained PR ─────────────────────────────────────────────────────────────
def make_pr_pretrained():
    cache_path = CACHE / "roc_cache_pretrained.pkl"
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache not found: {cache_path}")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    model_order = [
        ("RNABERT",           LM_COLORS["RNABERT"],  "-",  1.3),
        ("RNAErnie",          LM_COLORS["RNAErnie"], "-",  1.3),
        ("RNAMSM",            LM_COLORS["RNAMSM"],   "-",  1.3),
        ("RNA-FM",            LM_COLORS["RNA-FM"],   "-",  1.3),
        ("CircMAC (NoPT)",    NOPT_COLOR,             "--", 1.6),
        ("CircMAC (Pairing)", PROPOSED_COLOR,         "-",  2.4),
    ]
    entries = [(k, k, c, ls, lw) for k, c, ls, lw in model_order]

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    plot_pr_panel(ax, cache, entries,
                  "PR Curves — Pretrained Model Comparison\n(Fine-tuned RNA-LMs vs CircMAC)")
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        p = OUT / f"fig_pr_pretrained.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"Saved → {p}")
    plt.close(fig)


# ── Combined 2-panel ──────────────────────────────────────────────────────────
def make_pr_combined():
    enc_path  = CACHE / "roc_cache_encoder.pkl"
    pre_path  = CACHE / "roc_cache_pretrained.pkl"
    if not enc_path.exists() or not pre_path.exists():
        print("  Skip combined: missing cache")
        return

    with open(enc_path, "rb") as f:
        enc_cache = pickle.load(f)
    with open(pre_path, "rb") as f:
        pre_cache = pickle.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0),
                              gridspec_kw={"wspace": 0.28})
    fig.suptitle("Precision-Recall Curves (Pair Split)",
                 fontsize=11, fontweight="bold", y=1.01)

    enc_entries = [
        ("LSTM",        "LSTM",           ENCODER_COLORS["LSTM"],        "-", 1.3),
        ("Transformer", "Transformer",    ENCODER_COLORS["Transformer"], "-", 1.3),
        ("Mamba",       "Mamba",          ENCODER_COLORS["Mamba"],       "-", 1.3),
        ("Hymba",       "Hymba",          ENCODER_COLORS["Hymba"],       "-", 1.3),
        ("CircMAC",     "CircMAC (Ours)", ENCODER_COLORS["CircMAC"],     "-", 2.4),
    ]
    pre_entries = [
        ("RNABERT",           "RNABERT",           LM_COLORS["RNABERT"],  "-",  1.3),
        ("RNAErnie",          "RNAErnie",           LM_COLORS["RNAErnie"], "-",  1.3),
        ("RNAMSM",            "RNAMSM",             LM_COLORS["RNAMSM"],   "-",  1.3),
        ("RNA-FM",            "RNA-FM",             LM_COLORS["RNA-FM"],   "-",  1.3),
        ("CircMAC (NoPT)",    "CircMAC (NoPT)",     NOPT_COLOR,             "--", 1.6),
        ("CircMAC (Pairing)", "CircMAC (Pairing)",  PROPOSED_COLOR,         "-",  2.4),
    ]

    plot_pr_panel(axes[0], enc_cache, enc_entries, "(A) Encoder Architecture")
    plot_pr_panel(axes[1], pre_cache, pre_entries, "(B) RNA-LM Comparison")

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        p = OUT / f"fig_pr_combined.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"Saved → {p}")
    plt.close(fig)


def main():
    print("=== PR: Encoder ===")
    try:
        make_pr_encoder()
    except FileNotFoundError as e:
        print(f"  Skip: {e}")

    print("\n=== PR: Pretrained ===")
    try:
        make_pr_pretrained()
    except FileNotFoundError as e:
        print(f"  Skip: {e}")

    print("\n=== PR: Combined ===")
    make_pr_combined()


if __name__ == "__main__":
    main()
