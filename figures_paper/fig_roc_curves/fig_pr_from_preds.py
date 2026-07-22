#!/usr/bin/env python3
"""
fig_pr_from_preds.py
====================
Generate PR curves directly from eval_results/preds/*/test_preds.pkl.
No model inference needed — just reads saved predictions.

Output:
    figures_paper/fig_roc_curves/fig_pr_pretrained_preds.{pdf,png}
    figures_paper/fig_roc_curves/fig_pr_encoder_preds.{pdf,png}

Usage:
    python figures_paper/fig_roc_curves/fig_pr_from_preds.py
    python figures_paper/fig_roc_curves/fig_pr_from_preds.py --seeds 1 2 3
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

ROOT = Path(__file__).resolve().parents[2]
OUT  = Path(__file__).resolve().parent
PRED = ROOT / "eval_results" / "preds"   # eval_full.py 결과 (DataFrame format)
LOGS = ROOT / "logs"                      # trainer best_preds (dict format)

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         9,
    "axes.linewidth":    0.9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
})

# ── Model definitions ─────────────────────────────────────────────────────────
# (display_label, exp_template, model_name, color, linestyle, linewidth)
PRETRAINED_MODELS = [
    ("RNABERT (ft)",      "exp1_fair_trainable_rnabert",  "rnabert",  "#4878CF", "-",  1.4),
    ("RNAErnie (ft)",     "exp1_fair_trainable_rnaernie", "rnaernie", "#9467BD", "-",  1.4),
    ("RNAMSM (ft)",       "exp1_fair_trainable_rnamsm",   "rnamsm",   "#2CA02C", "-",  1.4),
    ("RNA-FM (ft)",       "exp1_fair_trainable_rnafm",    "rnafm",    "#17BECF", "-",  1.4),
    ("CircMAC (NoPT)",    "max_circmac_nopt",              "circmac",  "#BCBD22", "--", 1.6),
    ("CircMAC (Pairing)", "max_circmac_pairing",          "circmac",  "#E05C2A", "-",  2.4),
]

ENCODER_MODELS = [
    ("LSTM",        "v2_enc_lstm",        "lstm",        "#E377C2", "-", 1.4),
    ("Transformer", "v2_enc_transformer", "transformer", "#8C564B", "-", 1.4),
    ("Mamba",       "v2_enc_mamba",       "mamba",       "#D62728", "-", 1.4),
    ("Hymba",       "v2_enc_hymba",       "hymba",       "#BCBD22", "-", 1.4),
    ("CircMAC",     "max_circmac_nopt",    "circmac",     "#E05C2A", "-", 2.4),
]


# ── Data loading ──────────────────────────────────────────────────────────────
def _load_logs_preds(pkl_path: Path):
    """
    logs/{model}/{exp}/{seed}/best_preds/test_preds.pkl 형식 읽기.
    format: dict with {probs_sites, labels_sites, lengths_sites}
    훈련 중 AUPRC 계산에 쓰인 exact same data → training log 수치와 일치.
    """
    import pickle
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    probs_t  = d["probs_sites"]   # [N, L-1]
    labels_t = d["labels_sites"]  # [N, L-1], -100 for padding

    labels_flat = labels_t.reshape(-1).numpy()
    probs_flat  = probs_t.reshape(-1).numpy()
    valid = labels_flat != -100
    return labels_flat[valid].astype(int), probs_flat[valid].astype(float)


def _load_eval_preds(pkl_path: Path):
    """
    eval_results/preds/{exp}/test_preds.pkl 형식 읽기.
    format: DataFrame with [sample_idx, position, label, prob]
    """
    df = pd.read_pickle(pkl_path)
    return df["label"].values.astype(int), df["prob"].values.astype(float)


def load_preds(exp_template: str, seeds: list[int], model_name: str = None):
    """
    Best preds를 우선순위에 따라 로드:
      1) logs/{model}/{exp}/{seed}/best_preds/test_preds.pkl  (훈련 로그와 동일)
      2) eval_results/preds/{exp}/test_preds.pkl              (eval_full.py 결과)
    """
    results = []
    for seed in seeds:
        exp = f"{exp_template}_s{seed}"
        labels, preds = None, None

        # 1) logs/ best_preds 우선
        if model_name:
            logs_pkl = LOGS / model_name / exp / str(seed) / "best_preds" / "test_preds.pkl"
            if logs_pkl.exists():
                labels, preds = _load_logs_preds(logs_pkl)
                src = f"logs/{model_name}"

        # 2) eval_results/ fallback
        if labels is None:
            eval_pkl = PRED / exp / "test_preds.pkl"
            if eval_pkl.exists():
                labels, preds = _load_eval_preds(eval_pkl)
                src = "eval_results"

        if labels is None:
            print(f"  [SKIP] {exp}  (no preds found)")
            continue

        results.append({"seed": seed, "labels": labels, "preds": preds})
        print(f"  [OK]   {exp}  n={len(labels):,}  pos={labels.mean():.4f}  src={src}")
    return results


# ── PR curve utils ────────────────────────────────────────────────────────────
def mean_pr(seed_results):
    """Interpolate per-seed PR curves → mean ± std across seeds."""
    base_rec = np.linspace(0, 1, 500)
    precs, aps, prevs = [], [], []
    for r in seed_results:
        labels, preds = r["labels"], r["preds"]
        p, rc, _ = precision_recall_curve(labels, preds)
        p, rc = p[::-1], rc[::-1]          # ascending recall
        precs.append(np.interp(base_rec, rc, p))
        aps.append(average_precision_score(labels, preds))
        prevs.append(labels.mean())
    return (base_rec,
            np.mean(precs, 0), np.std(precs, 0),
            np.mean(aps), np.std(aps),
            np.mean(prevs))


def plot_pr_panel(ax, model_defs, seeds, title):
    baseline_drawn = False
    for label, exp_tmpl, model_name, color, ls, lw in model_defs:
        print(f"\n  {label}")
        results = load_preds(exp_tmpl, seeds, model_name)
        if not results:
            print(f"    → no preds found")
            continue

        rec, mp, sp, map_, std_ap, prev = mean_pr(results)

        ax.plot(rec, mp, color=color, lw=lw, ls=ls, zorder=3,
                label=f"{label}  (AP={map_:.3f}±{std_ap:.3f})")
        ax.fill_between(rec,
                        np.clip(mp - sp, 0, 1),
                        np.clip(mp + sp, 0, 1),
                        color=color, alpha=0.12 if lw > 1.8 else 0.07, zorder=2)

        if not baseline_drawn:
            ax.axhline(prev, color="gray", lw=0.8, ls=":", alpha=0.55,
                       label=f"Random  ({prev:.3f})")
            baseline_drawn = True

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Recall", fontsize=9)
    ax.set_ylabel("Precision", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.grid(True, linestyle="--", alpha=0.3, zorder=0)
    ax.legend(loc="upper right", fontsize=7.5, frameon=True,
              framealpha=0.92, edgecolor="#cccccc", handlelength=2.4)


# ── Figure builders ───────────────────────────────────────────────────────────
def make_pr_pretrained(seeds):
    print("\n=== PR: Pretrained model comparison ===")
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    plot_pr_panel(ax, PRETRAINED_MODELS, seeds,
                  "PR Curve — RNA-LM vs CircMAC\n(Pair Split, Site-level)")
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        p = OUT / f"fig_pr_pretrained_preds.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"Saved → {p}")
    plt.close(fig)


def make_pr_encoder(seeds):
    print("\n=== PR: Encoder comparison ===")
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    plot_pr_panel(ax, ENCODER_MODELS, seeds,
                  "PR Curve — Encoder Architecture\n(Pair Split, Site-level)")
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        p = OUT / f"fig_pr_encoder_preds.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"Saved → {p}")
    plt.close(fig)


def make_pr_combined(seeds):
    print("\n=== PR: Combined (2-panel) ===")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2),
                              gridspec_kw={"wspace": 0.28})
    fig.suptitle("Precision-Recall Curves (Pair Split, Site-level)",
                 fontsize=11, fontweight="bold", y=1.01)
    plot_pr_panel(axes[0], ENCODER_MODELS, seeds,
                  "(A) Encoder Architecture")
    plot_pr_panel(axes[1], PRETRAINED_MODELS, seeds,
                  "(B) RNA-LM vs CircMAC")
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        p = OUT / f"fig_pr_combined_preds.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"Saved → {p}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--only", choices=["pretrained", "encoder", "combined", "all"],
                        default="all")
    args = parser.parse_args()

    seeds = args.seeds
    print(f"Seeds: {seeds}")
    print(f"Preds dir: {PRED}")

    if args.only in ("pretrained", "all"):
        make_pr_pretrained(seeds)
    if args.only in ("encoder", "all"):
        make_pr_encoder(seeds)
    if args.only in ("combined", "all"):
        make_pr_combined(seeds)


if __name__ == "__main__":
    main()
