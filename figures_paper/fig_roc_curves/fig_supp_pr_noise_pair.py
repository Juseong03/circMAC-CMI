#!/usr/bin/env python3
"""
fig_supp_pr_noise_pair.py
=========================
Supplementary Figure (pair split): 2-panel per model
  (A) Precision–Recall curve — model + random baseline
  (B) AUPRC under simulated missing-positive annotation rates

Generates two separate figures:
    fig_supp_pr_noise_pair_pairing.{png,eps}   — CircMAC-BPP
    fig_supp_pr_noise_pair_nopt.{png,eps}       — CircMAC (NoPT)

Usage:
    python figures_paper/fig_roc_curves/fig_supp_pr_noise_pair.py
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

MODELS = {
    "pairing": {
        "exp_name":  "max_circmac_pairing",
        "model_dir": "circmac",
        "label":     "CircMAC-BPP",
        "color":     "#E05C2A",
        "suptitle":  "Supplementary Figure: CircMAC-BPP on Pair-Disjoint Split",
    },
    "nopt": {
        "exp_name":  "max_circmac_nopt",
        "model_dir": "circmac",
        "label":     "CircMAC (NoPT)",
        "color":     "#BCBD22",
        "suptitle":  "Supplementary Figure: CircMAC (NoPT) on Pair-Disjoint Split",
    },
}

SEEDS      = [1, 2, 3]
MASK_RATES = [0.0, 0.1, 0.2, 0.3, 0.5]
N_SIM      = 10


# ── Loader ────────────────────────────────────────────────────────────────────

def load_preds(exp_name, model_dir, seed):
    """Try logs/ first, then eval_results/."""
    p = LOGS / model_dir / f"{exp_name}_s{seed}" / str(seed) / "best_preds" / "test_preds.pkl"
    if p.exists():
        with open(p, "rb") as f:
            d = pickle.load(f)
        labels = d["labels_sites"].reshape(-1).numpy()
        probs  = d["probs_sites"].reshape(-1).numpy()
        valid  = labels != -100
        return labels[valid].astype(int), probs[valid].astype(float)

    p2 = PRED / f"{exp_name}_s{seed}" / "test_preds.pkl"
    if p2.exists():
        df = pd.read_pickle(p2)
        return df["label"].values.astype(int), df["prob"].values.astype(float)

    return None, None


# ── Panel A: PR curve ─────────────────────────────────────────────────────────

def plot_pr(ax, all_labels, all_probs, label, color):
    base_rec = np.linspace(0, 1, 500)
    precs, aps = [], []
    for y, p in zip(all_labels, all_probs):
        prec, rec, _ = precision_recall_curve(y, p)
        prec, rec = prec[::-1], rec[::-1]
        precs.append(np.interp(base_rec, rec, prec))
        aps.append(average_precision_score(y, p))

    mp     = np.mean(precs, 0)
    sp     = np.std(precs, 0)
    map_   = np.mean(aps)
    std_ap = np.std(aps)
    prev   = np.mean([y.mean() for y in all_labels])

    ax.axhline(prev, color="gray", lw=0.9, ls=":", alpha=0.6,
               label=f"Random  (AP={prev:.3f})", zorder=2)
    ax.plot(base_rec, mp, color=color, lw=2.2, zorder=3,
            label=f"{label}  (AP={map_:.3f}±{std_ap:.3f})")
    ax.fill_between(base_rec,
                    np.clip(mp - sp, 0, 1),
                    np.clip(mp + sp, 0, 1),
                    color=color, alpha=0.15, zorder=2)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Recall", fontsize=9)
    ax.set_ylabel("Precision", fontsize=9)
    ax.set_title("(A) Precision–Recall Curve", fontsize=10, fontweight="bold", pad=6)
    ax.grid(True, linestyle="--", alpha=0.3, zorder=0)
    ax.legend(loc="upper right", fontsize=7.5, frameon=True,
              framealpha=0.92, edgecolor="#cccccc", handlelength=2.2)

    print(f"  PR curve:  AP={map_:.4f}±{std_ap:.4f}  (seeds={len(aps)})")


# ── Panel B: label-noise ──────────────────────────────────────────────────────

def simulate_missing(labels, preds, mask_rate, rng):
    labels = labels.copy()
    pos_idx = np.where(labels == 1)[0]
    n_mask  = int(len(pos_idx) * mask_rate)
    if n_mask > 0:
        labels[rng.choice(pos_idx, size=n_mask, replace=False)] = 0
    return average_precision_score(labels, preds)


def plot_noise(ax, all_labels, all_probs, color, csv_path):
    rng = np.random.default_rng(42)
    rate_aps = {r: [] for r in MASK_RATES}

    for y, p in zip(all_labels, all_probs):
        for rate in MASK_RATES:
            if rate == 0.0:
                rate_aps[rate].append(average_precision_score(y, p))
            else:
                rate_aps[rate].extend(
                    [simulate_missing(y, p, rate, rng) for _ in range(N_SIM)]
                )

    rates_pct = [r * 100 for r in MASK_RATES]
    means = [np.mean(rate_aps[r]) for r in MASK_RATES]
    stds  = [np.std(rate_aps[r])  for r in MASK_RATES]

    ax.plot(rates_pct, means, color=color, lw=2.0, marker="o", ms=5, zorder=3)
    ax.fill_between(rates_pct,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    color=color, alpha=0.15, zorder=2)

    for x, m in zip(rates_pct, means):
        ax.annotate(f"{m:.3f}", (x, m), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=7.5, color=color)

    ax.set_xticks(rates_pct)
    ax.set_xlabel("Masked positive rate (%)", fontsize=9)
    ax.set_ylabel("AUPRC", fontsize=9)
    ax.set_title("(B) Label-Noise Sensitivity", fontsize=10, fontweight="bold", pad=6)
    ax.grid(True, linestyle="--", alpha=0.3, zorder=0)

    print(f"  Noise sim: " +
          "  ".join(f"{r*100:.0f}%→{np.mean(rate_aps[r]):.3f}" for r in MASK_RATES))

    rows = [{"mask_rate_pct": r * 100, "auprc_mean": np.mean(rate_aps[r]),
             "auprc_std": np.std(rate_aps[r])} for r in MASK_RATES]
    pd.DataFrame(rows).to_csv(csv_path, index=False)


# ── Per-model figure ──────────────────────────────────────────────────────────

def make_figure(key, cfg):
    exp_name  = cfg["exp_name"]
    model_dir = cfg["model_dir"]
    label     = cfg["label"]
    color     = cfg["color"]

    print(f"\n=== {label} ({exp_name}) ===")
    all_labels, all_probs = [], []
    for seed in SEEDS:
        y, p = load_preds(exp_name, model_dir, seed)
        if y is None:
            print(f"  [SKIP] seed={seed} — not found")
            continue
        all_labels.append(y)
        all_probs.append(p)
        print(f"  [OK]   seed={seed}  n={len(y):,}  pos={y.mean():.4f}")

    if not all_labels:
        print(f"  [ERROR] No predictions found for {label}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4),
                             gridspec_kw={"wspace": 0.32})

    plot_pr(axes[0], all_labels, all_probs, label, color)
    plot_noise(axes[1], all_labels, all_probs, color,
               csv_path=OUT / f"fig_supp_pr_noise_pair_{key}.csv")

    fig.suptitle(cfg["suptitle"], fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()

    for ext in ["png", "eps"]:
        p = OUT / f"fig_supp_pr_noise_pair_{key}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight",
                    format=ext if ext == "eps" else None)
        print(f"  Saved → {p}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    for key, cfg in MODELS.items():
        make_figure(key, cfg)


if __name__ == "__main__":
    main()
