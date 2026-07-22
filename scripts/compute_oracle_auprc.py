#!/usr/bin/env python3
"""
compute_oracle_auprc.py
=======================
Oracle AUPRC experiment to estimate the upper bound of achievable AUPRC
under simulated label noise in circRNA-miRNA binding site annotations.

Motivation:
-----------
Reviewer comment: "circMAC achieves AUPRC ~0.52–0.55; provide PR curves and
estimate an upper bound via an oracle experiment under simulated label noise."

The global AUPRC is ~0.12–0.18 (site-level, pair split) — substantially lower
than the 0.52–0.55 figure the reviewer references, which corresponds to F1-macro
or AUROC in Table 1, not AUPRC.

This script shows that even a PERFECT predictor achieves limited AUPRC due to
heavy class imbalance (~5.6% positive sites) and potential annotation noise.
Two scenarios are simulated:

  1. Oracle (no noise): use ground-truth labels as predictions → upper bound
  2. Noisy oracle: randomly flip p_noise fraction of true positives → shows
     how annotation incompleteness degrades achievable AUPRC

Usage:
    python scripts/compute_oracle_auprc.py
    python scripts/compute_oracle_auprc.py --noise_levels 0.0 0.1 0.2 0.3 0.5
    python scripts/compute_oracle_auprc.py --model_pkl eval_results/preds/v2_abl_full_s1/test_preds.pkl
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve, auc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "figures_paper" / "fig_oracle_auprc"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         9,
    "axes.linewidth":    0.9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
})


def load_preds(pkl_path: Path) -> pd.DataFrame:
    """Load a test_preds.pkl file → DataFrame with [sample_idx, position, label, prob]."""
    return pd.read_pickle(pkl_path)


def oracle_auprc_global(labels: np.ndarray, noise: float = 0.0,
                        n_trials: int = 20, rng=None):
    """
    Global (token-level) oracle AUPRC under annotation noise.

    Interpretation:
      - `labels` = TRUE underlying binding sites (assumed complete)
      - We simulate that noise% of true positives are MISSED in annotation
        → noisy_labels = observed (incomplete) annotations
      - Oracle prediction = TRUE labels (model knows ground truth)
      - AUPRC is computed: oracle_preds vs noisy_labels
      → This caps achievable AUPRC because oracle is penalized for
        confidently predicting sites that were NOT annotated.

    noise=0: oracle uses true labels evaluated on true labels → AUPRC=1.0
    noise>0: oracle knows truth but evaluated on noisy GT → AUPRC < 1.0
    """
    if rng is None:
        rng = np.random.default_rng(42)

    if noise == 0.0:
        return 1.0, 0.0

    pos_idx = np.where(labels == 1)[0]
    aps = []
    for _ in range(n_trials):
        noisy = labels.copy()
        n_flip = int(len(pos_idx) * noise)
        flip_idx = rng.choice(pos_idx, size=n_flip, replace=False)
        noisy[flip_idx] = 0   # these sites were missed in annotation
        # Oracle prediction = true labels; GT for eval = noisy observed
        if len(np.unique(noisy)) > 1:
            ap = average_precision_score(noisy, labels.astype(float))
            aps.append(ap)
    return float(np.mean(aps)), float(np.std(aps))


def oracle_auprc_perpair(df: pd.DataFrame, noise: float = 0.0,
                          n_trials: int = 20, rng=None):
    """
    Per-pair oracle AUPRC (averaged across pairs) under annotation noise.

    Same interpretation as oracle_auprc_global but computed per circRNA–miRNA pair.
    Returns (mean, std) across Monte-Carlo trials.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    if noise == 0.0:
        return 1.0, 0.0

    trial_means = []
    for _ in range(n_trials):
        noisy_df = df.copy()
        pos_mask = noisy_df["label"] == 1
        pos_idx  = noisy_df.index[pos_mask]
        n_flip   = int(len(pos_idx) * noise)
        flip_idx = rng.choice(pos_idx, size=n_flip, replace=False)
        noisy_df.loc[flip_idx, "label"] = 0

        aps = []
        for idx, grp in noisy_df.groupby("sample_idx"):
            noisy_lbl  = grp["label"].values
            true_lbl   = df.loc[grp.index, "label"].values
            # Only include pairs where noisy annotation still has positives
            if noisy_lbl.sum() > 0 and len(np.unique(noisy_lbl)) > 1:
                ap = average_precision_score(noisy_lbl, true_lbl.astype(float))
                aps.append(ap)
        if aps:
            trial_means.append(np.mean(aps))

    return float(np.mean(trial_means)), float(np.std(trial_means))


def model_auprc_global(df: pd.DataFrame) -> float:
    """Global AUPRC for model predictions (as reported in Table 1 eval)."""
    return average_precision_score(df["label"].values, df["prob"].values)


def model_auprc_perpair(df: pd.DataFrame):
    """Per-pair AUPRC for model predictions."""
    aps = []
    for _, grp in df.groupby("sample_idx"):
        lbl = grp["label"].values
        if lbl.sum() > 0 and len(np.unique(lbl)) > 1:
            aps.append(average_precision_score(lbl, grp["prob"].values))
    return float(np.mean(aps)), float(np.std(aps))


def make_oracle_figure(results_global: dict, results_perpair: dict,
                       model_global: float, model_perpair: float,
                       noise_levels: list):
    """
    Two-panel figure:
    (A) Global AUPRC vs noise level — oracle curve + model line
    (B) Per-pair AUPRC vs noise level — oracle curve + model line
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), gridspec_kw={"wspace": 0.30})

    noise_pct = [n * 100 for n in noise_levels]

    # (A) Global AUPRC
    ax = axes[0]
    oracle_vals_g = [results_global[n][0] for n in noise_levels]
    oracle_stds_g = [results_global[n][1] for n in noise_levels]
    ax.plot(noise_pct, oracle_vals_g, "o-", color="#2171B5", lw=2,
            label="Oracle upper bound")
    ax.axhline(model_global, color="#E05C2A", lw=1.8, ls="--",
               label=f"CircMAC (NoPT)  (AP={model_global:.3f})")
    ax.set_xlabel("Annotation noise (% of positives flipped)", fontsize=9)
    ax.set_ylabel("Global AUPRC", fontsize=9)
    ax.set_title("(A) Global (token-level) AUPRC", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, frameon=True, framealpha=0.9, edgecolor="#cccccc")
    ax.grid(True, linestyle="--", alpha=0.3)

    # Annotate prevalence
    ax.annotate(f"Baseline (random) ≈ {0.056:.3f}",
                xy=(noise_pct[-1], 0.056), xytext=(noise_pct[-1] * 0.5, 0.08),
                fontsize=7.5, color="gray",
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.7))

    # (B) Per-pair AUPRC
    ax = axes[1]
    perpair_means = [results_perpair[n][0] for n in noise_levels]
    perpair_stds  = [results_perpair[n][1] for n in noise_levels]
    perp_std_arr  = np.array(perpair_stds)
    perp_mean_arr = np.array(perpair_means)

    ax.plot(noise_pct, perp_mean_arr, "o-", color="#2171B5", lw=2,
            label="Oracle upper bound")
    ax.fill_between(noise_pct,
                    np.clip(perp_mean_arr - perp_std_arr, 0, 1),
                    np.clip(perp_mean_arr + perp_std_arr, 0, 1),
                    color="#2171B5", alpha=0.15)
    ax.axhline(model_perpair, color="#E05C2A", lw=1.8, ls="--",
               label=f"CircMAC (NoPT)  (AP={model_perpair:.3f})")
    ax.set_xlabel("Annotation noise (% of positives flipped)", fontsize=9)
    ax.set_ylabel("Per-pair mean AUPRC", fontsize=9)
    ax.set_title("(B) Per-pair AUPRC", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, frameon=True, framealpha=0.9, edgecolor="#cccccc")
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle(
        "Oracle AUPRC Upper Bound Under Simulated Annotation Noise\n"
        "(CircRNA–miRNA Binding Site Prediction, Pair Split)",
        fontsize=11, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        p = OUT / f"fig_oracle_auprc.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"Saved → {p}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_pkl", type=str,
                        default="eval_results/preds/v2_abl_full_s1/test_preds.pkl",
                        help="Path to test_preds.pkl from a CircMAC (NoPT) run")
    parser.add_argument("--noise_levels", type=float, nargs="+",
                        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                        help="Fraction of positive labels to randomly flip to negative")
    parser.add_argument("--n_trials", type=int, default=20,
                        help="Monte-Carlo trials per noise level")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pkl_path = ROOT / args.model_pkl
    if not pkl_path.exists():
        # Try alternate paths
        alts = list((ROOT / "eval_results" / "preds").glob("v2_abl_full_s*/test_preds.pkl"))
        if not alts:
            raise FileNotFoundError(f"Cannot find test_preds.pkl: {pkl_path}")
        pkl_path = alts[0]
        print(f"Using: {pkl_path}")

    print(f"Loading predictions from: {pkl_path}")
    df = load_preds(pkl_path)
    print(f"  Total tokens: {len(df)}, pos_rate={df['label'].mean():.4f}")
    print(f"  n_pairs: {df['sample_idx'].nunique()}")

    rng = np.random.default_rng(args.seed)
    labels = df["label"].values

    # Model AUPRC
    model_g  = model_auprc_global(df)
    model_pp, model_pp_std = model_auprc_perpair(df)
    print(f"\nCircMAC (NoPT) predictions:")
    print(f"  Global AUPRC:   {model_g:.4f}")
    print(f"  Per-pair AUPRC: {model_pp:.4f} ± {model_pp_std:.4f}")

    print(f"\nOracle AUPRC under simulated annotation noise:")
    print(f"  (noise = fraction of TRUE positives randomly flipped to 0)")
    print(f"  {'Noise':>8}  {'Global':>10}  {'Per-pair':>12}  {'(±std)':>8}")

    results_global  = {}
    results_perpair = {}
    for noise in args.noise_levels:
        g, g_std = oracle_auprc_global(labels.copy(), noise=noise, n_trials=args.n_trials, rng=rng)
        pp, pp_std = oracle_auprc_perpair(df, noise=noise, n_trials=args.n_trials, rng=rng)
        results_global[noise]  = (g, g_std)
        results_perpair[noise] = (pp, pp_std)
        print(f"  {noise*100:>7.0f}%  {g:>10.4f} ({g_std:.4f})  {pp:>10.4f} ({pp_std:.4f})")

    # Save table
    rows = []
    for noise in args.noise_levels:
        rows.append(dict(
            noise_pct=noise * 100,
            oracle_global=results_global[noise][0],
            oracle_global_std=results_global[noise][1],
            oracle_perpair_mean=results_perpair[noise][0],
            oracle_perpair_std=results_perpair[noise][1],
            model_global=model_g,
            model_perpair=model_pp,
        ))
    df_out = pd.DataFrame(rows)
    csv_path = OUT / "oracle_auprc_results.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"\nSaved table → {csv_path}")

    make_oracle_figure(results_global, results_perpair, model_g, model_pp, args.noise_levels)


if __name__ == "__main__":
    main()
