#!/usr/bin/env python3
"""
fig_supp_label_noise.py
=======================
Supplementary Figure: AUPRC under simulated missing-positive annotation rates.

For each masked_rate in {0%, 10%, 20%, 30%, 50%}:
  - Randomly set that fraction of positive labels to 0 (missing annotation)
  - Re-compute AUPRC using the same valid-position protocol as main tables
  - Repeat over seeds → mean ± std

Caption note:
  "This analysis does not estimate a definitive biological upper bound, but
   illustrates how incomplete CLIP-derived annotations can deflate
   precision-based metrics."

Usage:
    python figures_paper/fig_roc_curves/fig_supp_label_noise.py
    python figures_paper/fig_roc_curves/fig_supp_label_noise.py --splits pair iso bsj
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score

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

MASK_RATES = [0.0, 0.1, 0.2, 0.3, 0.5]
SEEDS      = [1, 2, 3]
COLOR      = "#E05C2A"

SPLIT_CONFIG = {
    "pair": {"exp_name": "max_circmac_pairing", "model": "circmac",
             "label": "Pair-disjoint"},
    "iso":  {"exp_name": "bsj_circmac_pairing", "model": "circmac",
             "label": "Isoform-disjoint"},
    "bsj":  {"exp_name": "bsj_circmac_pairing", "model": "circmac",
             "label": "BSJ-disjoint"},
}


# ── Loaders (same as fig_supp_pr_circmac) ────────────────────────────────────

def load_logs_pkl(model, exp_name, seed):
    p = LOGS / model / f"{exp_name}_s{seed}" / str(seed) / "best_preds" / "test_preds.pkl"
    if not p.exists():
        return None, None
    with open(p, "rb") as f:
        d = pickle.load(f)
    labels = d["labels_sites"].reshape(-1).numpy()
    probs  = d["probs_sites"].reshape(-1).numpy()
    valid  = labels != -100
    return labels[valid].astype(int), probs[valid].astype(float)


def load_eval_pkl(exp_name, seed):
    p = PRED / f"{exp_name}_s{seed}" / "test_preds.pkl"
    if not p.exists():
        return None, None
    df = pd.read_pickle(p)
    return df["label"].values.astype(int), df["prob"].values.astype(float)


def load_preds(split_cfg, seed):
    y, p = load_logs_pkl(split_cfg["model"], split_cfg["exp_name"], seed)
    if y is None:
        y, p = load_eval_pkl(split_cfg["exp_name"], seed)
    return y, p


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate_missing(labels, preds, mask_rate, rng):
    """Randomly flip mask_rate fraction of positive labels → 0."""
    labels = labels.copy()
    pos_idx = np.where(labels == 1)[0]
    n_mask  = int(len(pos_idx) * mask_rate)
    if n_mask > 0:
        chosen = rng.choice(pos_idx, size=n_mask, replace=False)
        labels[chosen] = 0
    return average_precision_score(labels, preds)


def run_split(split, split_cfg, n_sim=10):
    """Return dict: mask_rate → list of AUPRC values across seeds × simulations."""
    print(f"\n  [{split_cfg['label']}]")
    rate_aps = {r: [] for r in MASK_RATES}
    rng = np.random.default_rng(42)

    for seed in SEEDS:
        y, p = load_preds(split_cfg, seed)
        if y is None:
            print(f"    [SKIP] seed={seed}")
            continue
        print(f"    seed={seed}  n={len(y):,}  pos={y.mean():.4f}")
        for rate in MASK_RATES:
            if rate == 0.0:
                rate_aps[rate].append(average_precision_score(y, p))
            else:
                # Average over n_sim random masks per seed
                sims = [simulate_missing(y, p, rate, rng) for _ in range(n_sim)]
                rate_aps[rate].extend(sims)

    return rate_aps


# ── Figure ────────────────────────────────────────────────────────────────────

def plot_noise_panel(ax, rate_aps, title, color=COLOR):
    rates_pct = [r * 100 for r in MASK_RATES]
    means = [np.mean(rate_aps[r]) if rate_aps[r] else np.nan for r in MASK_RATES]
    stds  = [np.std(rate_aps[r])  if rate_aps[r] else np.nan for r in MASK_RATES]

    ax.plot(rates_pct, means, color=color, lw=2.0, marker="o", ms=5, zorder=3)
    ax.fill_between(rates_pct,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    color=color, alpha=0.15, zorder=2)

    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.set_xlabel("Masked positive rate (%)", fontsize=9)
    ax.set_ylabel("AUPRC", fontsize=9)
    ax.set_xticks([r * 100 for r in MASK_RATES])
    ax.grid(True, linestyle="--", alpha=0.3, zorder=0)

    # Annotate values
    for x, m, s in zip(rates_pct, means, stds):
        if not np.isnan(m):
            ax.annotate(f"{m:.3f}", (x, m), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=7.5, color=color)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["pair", "iso", "bsj"])
    parser.add_argument("--n_sim", type=int, default=10,
                        help="Number of random mask simulations per seed")
    args = parser.parse_args()

    splits = [s for s in args.splits if s in SPLIT_CONFIG]
    n_panels = len(splits)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4.5),
                             gridspec_kw={"wspace": 0.32})
    if n_panels == 1:
        axes = [axes]

    print("=== Label-noise sensitivity (missing-positive simulation) ===")
    all_data = {}
    for ax, split in zip(axes, splits):
        cfg = SPLIT_CONFIG[split]
        rate_aps = run_split(split, cfg, n_sim=args.n_sim)
        all_data[split] = rate_aps
        plot_noise_panel(ax, rate_aps, cfg["label"])

    fig.suptitle("Supplementary Figure: AUPRC under Simulated Missing-Positive Annotation",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()

    for ext in ["pdf", "png"]:
        p = OUT / f"fig_supp_label_noise.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"\nSaved → {p}")
    plt.close(fig)

    # Save CSV summary
    rows = []
    for split, rate_aps in all_data.items():
        for rate, aps in rate_aps.items():
            if aps:
                rows.append({"split": split, "mask_rate": rate,
                             "auprc_mean": np.mean(aps), "auprc_std": np.std(aps),
                             "n": len(aps)})
    if rows:
        df = pd.DataFrame(rows)
        csv_p = OUT / "fig_supp_label_noise.csv"
        df.to_csv(csv_p, index=False)
        print(f"Saved CSV → {csv_p}")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
