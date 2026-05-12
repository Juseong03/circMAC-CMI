#!/usr/bin/env python3
"""
Fig 2: RNA Language Model Comparison — Frozen vs Fine-tuned (EXP1)

Two separate figures:
  fig2a_frozen.{pdf,png}     — Frozen RNA-LMs vs circMAC
  fig2b_finetune.{pdf,png}   — Fine-tuned RNA-LMs vs circMAC

Shared:
  fig2_rna_lm_data.csv
  fig2_rna_lm_summary.csv

Model order in plots:
  RNABERT → RNAErnie → RNAMSM → RNA-FM → circMAC (ours)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / "logs_0512"
OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [1, 2, 3]


# ── Model definitions ────────────────────────────────────────────────────────
# (display_label, model_dir, exp_base, is_frozen)
RNA_LMS = [
    ("RNABERT",  "rnabert",  "exp1_fair_frozen_rnabert",     True),
    ("RNABERT",  "rnabert",  "exp1_fair_trainable_rnabert",  False),

    ("RNAErnie", "rnaernie", "exp1_fair_frozen_rnaernie",    True),
    ("RNAErnie", "rnaernie", "exp1_fair_trainable_rnaernie", False),

    ("RNAMSM",   "rnamsm",   "exp1_fair_frozen_rnamsm",      True),
    ("RNAMSM",   "rnamsm",   "exp1_fair_trainable_rnamsm",   False),

    ("RNA-FM",   "rnafm",    "exp1_fair_frozen_rnafm",       True),
    ("RNA-FM",   "rnafm",    "exp1_fair_trainable_rnafm",    False),
]

# circMAC shown once at the far right
CIRCMAC = ("circMAC", "circmac", "v2_enc_circmac", False)


# ── Colors ───────────────────────────────────────────────────────────────────
RNA_LM_COLORS = {
    "RNABERT":  "#1F77B4",
    "RNAErnie": "#9467BD",
    "RNAMSM":   "#2CA02C",
    "RNA-FM":   "#17BECF",
}

CIRCMAC_COLOR = "#FF7F0E"
FROZEN_HATCH = "//"


# ── Matplotlib style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ── Metrics ──────────────────────────────────────────────────────────────────
METRICS = [
    ("f1_macro", "(a) F1-macro", (0.55, 0.82)),
    ("roc_auc",  "(b) AUROC",    (0.65, 0.97)),
    ("auprc",    "(c) AUPRC",    (0.15, 0.65)),
]


def load_scores(model_dir, exp_base):
    """
    Load scores from training.json files across seeds.
    """
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


def build_entries_single(scores_cm, lm_scores, metric, is_frozen):
    """
    Build plotting entries for one mode, with circMAC at the far right.

    Layout:
      x=0: RNABERT
      x=1: RNAErnie
      x=2: RNAMSM
      x=3: RNA-FM
      x=5: circMAC

    The empty gap between x=3 and x=5 visually separates RNA-LMs from ours.
    """
    entries = []
    hatch = FROZEN_HATCH if is_frozen else None

    lm_order = ["RNABERT", "RNAErnie", "RNAMSM", "RNA-FM"]

    for x, lm_name in enumerate(lm_order):
        color = RNA_LM_COLORS[lm_name]
        scores = lm_scores.get((lm_name, is_frozen), None)

        if scores is None:
            print(f"[Warning] Missing loaded scores for {lm_name}, frozen={is_frozen}")
            vals = []
        else:
            vals = scores[metric]

        entries.append((x, lm_name, color, hatch, vals))

    # circMAC last
    entries.append((4, "circMAC", CIRCMAC_COLOR, None, scores_cm[metric]))

    return entries


def plot_panel(ax, entries, metric, title, ylim):
    """
    Plot one metric panel.

    entries:
      list of (x_pos, label, color, hatch, vals)
    """
    ylo = ylim[0]
    y_range = ylim[1] - ylim[0]

    for x, label, color, hatch, vals in entries:
        if not vals:
            print(f"[Warning] No values for {label} / {metric}")
            continue

        mean = np.mean(vals)
        std = np.std(vals)

        ax.bar(
            x,
            mean - ylo,
            bottom=ylo,
            width=0.62,
            color=color,
            alpha=0.85,
            hatch=hatch,
            edgecolor="white" if hatch else color,
            linewidth=0.5,
            zorder=2,
        )

        ax.errorbar(
            x,
            mean,
            yerr=std,
            fmt="none",
            color="#222222",
            capsize=3.5,
            capthick=1.3,
            elinewidth=1.3,
            zorder=4,
        )

        ax.text(
            x,
            mean + std + y_range * 0.025,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#222222",
            zorder=6,
            bbox=dict(
                boxstyle="round,pad=0.13",
                fc="white",
                ec="none",
                alpha=0.85,
            ),
        )

    # ax.set_xlim(-0.6, 5.6)
    ax.set_xlim(-0.6, 4.6)
    ax.set_ylim(*ylim)

    ax.set_xticks([e[0] for e in entries])
    ax.set_xticklabels([e[1] for e in entries], rotation=20, ha="right")

    # Highlight circMAC tick label
    for tick, entry in zip(ax.get_xticklabels(), entries):
        _, label, _, _, _ = entry
        if label == "circMAC":
            tick.set_fontweight("bold")
            tick.set_color(CIRCMAC_COLOR)

    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    # Separator between RNA-LMs and circMAC
    # ax.axvline(4.0, color="#999999", lw=0.9, ls=":", zorder=1)


def make_subfig(scores_cm, lm_scores, is_frozen, fname, suptitle):
    """
    Make one figure:
      - frozen RNA-LMs vs circMAC
      - or fine-tuned RNA-LMs vs circMAC
    """
    hatch_label = "Frozen encoder" if is_frozen else "Fine-tuned encoder"
    hatch = FROZEN_HATCH if is_frozen else None

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2))
    fig.suptitle(suptitle, fontsize=12, fontweight="bold", y=1.01)

    for ax, (metric, title, ylim) in zip(axes, METRICS):
        entries = build_entries_single(
            scores_cm=scores_cm,
            lm_scores=lm_scores,
            metric=metric,
            is_frozen=is_frozen,
        )

        plot_panel(
            ax=ax,
            entries=entries,
            metric=metric,
            title=title,
            ylim=ylim,
        )

    legend_handles = [
        mpatches.Patch(
            facecolor=CIRCMAC_COLOR,
            edgecolor=CIRCMAC_COLOR,
            label="circMAC (ours)",
        ),
        mpatches.Patch(
            facecolor="#888888",
            edgecolor="white" if hatch else "#888888",
            hatch=hatch,
            label=hatch_label,
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.16)

    for ext in ["pdf", "png"]:
        path = OUT / f"{fname}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved → {path}")

    plt.close(fig)


def save_score_tables(scores_cm, lm_scores):
    """
    Save raw and summary score tables used for the figure.
    """
    rows = []

    # RNA-LMs
    for label, _, _, is_frozen in RNA_LMS:
        mode = "frozen" if is_frozen else "fine-tuned"
        scores = lm_scores[(label, is_frozen)]

        max_len = max(
            len(scores["f1_macro"]),
            len(scores["roc_auc"]),
            len(scores["auprc"]),
        )

        for i in range(max_len):
            seed = SEEDS[i] if i < len(SEEDS) else i + 1

            row = {
                "model": label,
                "mode": mode,
                "seed": seed,
                "f1_macro": scores["f1_macro"][i] if i < len(scores["f1_macro"]) else np.nan,
                "roc_auc": scores["roc_auc"][i] if i < len(scores["roc_auc"]) else np.nan,
                "auprc": scores["auprc"][i] if i < len(scores["auprc"]) else np.nan,
            }
            rows.append(row)

    # circMAC
    max_len_cm = max(
        len(scores_cm["f1_macro"]),
        len(scores_cm["roc_auc"]),
        len(scores_cm["auprc"]),
    )

    for i in range(max_len_cm):
        seed = SEEDS[i] if i < len(SEEDS) else i + 1

        rows.append({
            "model": "circMAC",
            "mode": "circMAC",
            "seed": seed,
            "f1_macro": scores_cm["f1_macro"][i] if i < len(scores_cm["f1_macro"]) else np.nan,
            "roc_auc": scores_cm["roc_auc"][i] if i < len(scores_cm["roc_auc"]) else np.nan,
            "auprc": scores_cm["auprc"][i] if i < len(scores_cm["auprc"]) else np.nan,
        })

    df = pd.DataFrame(rows)

    summary = (
        df.groupby(["model", "mode"])
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

    raw_path = OUT / "fig2_rna_lm_data.csv"
    summary_path = OUT / "fig2_rna_lm_summary.csv"

    df.to_csv(raw_path, index=False)
    summary.to_csv(summary_path)

    print(f"Saved → {raw_path}")
    print(f"Saved → {summary_path}")
    print(summary.to_string())


def main():
    # ── Load circMAC scores ──────────────────────────────────────────────────
    _, dir_cm, exp_cm, _ = CIRCMAC
    scores_cm = load_scores(dir_cm, exp_cm)

    # ── Load RNA-LM scores ───────────────────────────────────────────────────
    lm_scores = {}

    for label, model_dir, exp_base, is_frozen in RNA_LMS:
        scores = load_scores(model_dir, exp_base)
        lm_scores[(label, is_frozen)] = scores

    # ── Save CSV tables ──────────────────────────────────────────────────────
    save_score_tables(scores_cm, lm_scores)

    # ── Generate figures ─────────────────────────────────────────────────────
    make_subfig(
        scores_cm=scores_cm,
        lm_scores=lm_scores,
        is_frozen=True,
        fname="fig2a_frozen",
        suptitle="RNA Language Model Comparison — Frozen Encoders",
    )

    make_subfig(
        scores_cm=scores_cm,
        lm_scores=lm_scores,
        is_frozen=False,
        fname="fig2b_finetune",
        suptitle="RNA Language Model Comparison — Fine-tuned Encoders",
    )


if __name__ == "__main__":
    main()