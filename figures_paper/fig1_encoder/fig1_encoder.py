#!/usr/bin/env python3
"""
Fig 1: Encoder Architecture Comparison (EXP3)
  - Panel A: F1-macro (mean ± std)
  - Panel B: AUROC    (mean ± std)
  - Panel C: AUPRC    (mean ± std)

Output (this folder):
  fig1_encoder.pdf / fig1_encoder.png
  fig1_encoder_data.csv
  fig1_encoder_summary.csv

Model order:
  LSTM → Transformer → Mamba → Hymba → circMAC
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / "logs_0510"
OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [1, 2, 3]


# ── Model order: baselines first, circMAC last ───────────────────────────────
MODELS = [
    ("LSTM",        "lstm",        "v2_enc_lstm"),
    ("Transformer", "transformer", "v2_enc_transformer"),
    ("Mamba",       "mamba",       "v2_enc_mamba"),
    ("Hymba",       "hymba",       "v2_enc_hymba"),
    ("circMAC",     "circmac",     "v2_enc_circmac"),
]


# Same circMAC color as Fig. 2
PROPOSED = "#FF7F0E"
BASELINE = "#6B9CC7"
BAR_ALPHA = 0.82


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


METRICS = [
    ("f1_macro", "(a) F1-macro", (0.55, 0.85)),
    ("roc_auc",  "(b) AUROC",    (0.70, 0.97)),
    ("auprc",    "(c) AUPRC",    (0.15, 0.65)),
]


def load_scores(model_dir, exp_base):
    scores = {
        "f1_macro": [],
        "roc_auc": [],
        "auprc": [],
    }

    for seed in SEEDS:
        path = LOGS / model_dir / f"{exp_base}_s{seed}" / str(seed) / "training.json"

        if not path.exists():
            print(f"[Warning] Missing file: {path}")
            continue

        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            print(f"[Warning] Failed to parse JSON: {path}")
            continue

        final = data.get("final", {})
        if not final:
            print(f"[Warning] Missing final results: {path}")
            continue

        try:
            site_scores = list(final.values())[0]["scores"]["sites"]
        except KeyError:
            print(f"[Warning] Missing site scores: {path}")
            continue

        for metric in scores:
            if metric in site_scores:
                scores[metric].append(site_scores[metric])
            else:
                print(f"[Warning] Missing metric '{metric}' in {path}")

    return scores


def plot_panel(ax, data, metric, title, ylim):
    """
    Plot one metric panel.

    data:
      list of (label, color, vals)
    """
    ylo = ylim[0]
    y_range = ylim[1] - ylim[0]

    for i, (label, color, vals) in enumerate(data):
        if not vals:
            print(f"[Warning] No values for {label} / {metric}")
            continue

        vals = np.asarray(vals, dtype=float)
        mean = np.mean(vals)
        std = np.std(vals)

        # Bar
        ax.bar(
            i,
            mean - ylo,
            bottom=ylo,
            width=0.55,
            color=color,
            alpha=BAR_ALPHA,
            zorder=2,
            linewidth=0,
        )

        # Error bar
        ax.errorbar(
            i,
            mean,
            yerr=std,
            fmt="none",
            color="#222222",
            capsize=4,
            capthick=1.4,
            elinewidth=1.4,
            zorder=4,
        )

        # Mean annotation
        ax.text(
            i,
            mean + std + y_range * 0.025,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#222222",
            zorder=6,
            bbox=dict(
                boxstyle="round,pad=0.15",
                fc="white",
                ec="none",
                alpha=0.85,
            ),
        )

    ax.set_xticks(range(len(data)))
    ax.set_xticklabels([d[0] for d in data], rotation=20, ha="right")
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)


def save_tables(rows):
    df = pd.DataFrame(rows)

    summary = (
        df.groupby("model")
        .agg(
            f1_mean=("f1_macro", "mean"),
            f1_std=("f1_macro", "std"),
            roc_mean=("roc_auc", "mean"),
            roc_std=("roc_auc", "std"),
            auprc_mean=("auprc", "mean"),
            auprc_std=("auprc", "std"),
        )
        .round(4)
    )

    data_path = OUT / "fig1_encoder_data.csv"
    summary_path = OUT / "fig1_encoder_summary.csv"

    df.to_csv(data_path, index=False)
    summary.to_csv(summary_path)

    print(f"Saved → {data_path}")
    print(f"Saved → {summary_path}")
    print(summary.to_string())


def main():
    all_data = []
    rows = []

    for label, model_dir, exp_base in MODELS:
        color = PROPOSED if label == "circMAC" else BASELINE
        scores = load_scores(model_dir, exp_base)

        all_data.append((label, color, scores))

        max_len = max(
            len(scores["f1_macro"]),
            len(scores["roc_auc"]),
            len(scores["auprc"]),
        )

        for i in range(max_len):
            seed = SEEDS[i] if i < len(SEEDS) else i + 1

            rows.append({
                "model": label,
                "seed": seed,
                "f1_macro": scores["f1_macro"][i] if i < len(scores["f1_macro"]) else np.nan,
                "roc_auc": scores["roc_auc"][i] if i < len(scores["roc_auc"]) else np.nan,
                "auprc": scores["auprc"][i] if i < len(scores["auprc"]) else np.nan,
            })

    save_tables(rows)

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.0))
    fig.suptitle(
        "Encoder Architecture Comparison",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )

    for ax, (metric, title, ylim) in zip(axes, METRICS):
        panel_data = [
            (label, color, scores[metric])
            for label, color, scores in all_data
        ]

        plot_panel(
            ax=ax,
            data=panel_data,
            metric=metric,
            title=title,
            ylim=ylim,
        )

    # Highlight circMAC x-axis label
    for ax in axes:
        for tick in ax.get_xticklabels():
            if tick.get_text() == "circMAC":
                tick.set_fontweight("bold")
                tick.set_color(PROPOSED)

    fig.tight_layout()

    for ext in ["pdf", "png"]:
        path = OUT / f"fig1_encoder.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved → {path}")

    plt.close(fig)


if __name__ == "__main__":
    main()