"""
실험 진행 상황 확인 + 중간 결과 요약
======================================

logs/{model}/{exp}/{seed}/training.json 을 읽어
실험별 F1 (mean ± std across seeds) 테이블을 출력합니다.

Usage:
  python docs/paper_cmi/check_progress.py          # 전체 요약
  python docs/paper_cmi/check_progress.py --exp1   # EXP1만
  python docs/paper_cmi/check_progress.py --exp2   # EXP2 (사전학습) 만
  python docs/paper_cmi/check_progress.py --exp4   # EXP4만
  python docs/paper_cmi/check_progress.py --exp56  # EXP5+6만
  python docs/paper_cmi/check_progress.py --csv    # CSV로 저장
  python docs/paper_cmi/check_progress.py --plot   # bar chart 생성
"""

import json
import argparse
import numpy as np
from pathlib import Path

ROOT    = Path(__file__).parent.parent.parent
LOG_DIR = ROOT / 'logs'
OUT_DIR = Path(__file__).parent

# ── 실험 정의 ──────────────────────────────────────────────────────────────────
# (group, label, model, final_exp_prefix, old_exp_prefix)
# final_exp_prefix : scripts/final_v2/ 에서 사용하는 최종 naming
# old_exp_prefix   : 이전 실험에서 돌린 naming (--final 모드에선 무시)

EXPERIMENTS = [
    # ── EXP1 Base ──────────────────────────────────────────────────────────
    ("EXP1_base", "CircMAC",     "circmac",     "exp1_circmac",     None),
    ("EXP1_base", "Mamba",       "mamba",       "exp1_mamba",       None),
    ("EXP1_base", "Hymba",       "hymba",       "exp1_hymba",       None),
    ("EXP1_base", "LSTM",        "lstm",        "exp1_lstm",        None),
    ("EXP1_base", "Transformer", "transformer", "exp1_transformer", None),

    # ── EXP1 RNA LM Frozen ─────────────────────────────────────────────────
    ("EXP1_frozen", "RNABERT (frz)",  "rnabert", "exp1_rnabert_frozen",  "exp3_rnabert_frozen"),
    ("EXP1_frozen", "RNAErnie (frz)", "rnaernie","exp1_rnaernie_frozen", "exp3_rnaernie_frozen"),
    ("EXP1_frozen", "RNA-FM (frz)",   "rnafm",   "exp1_rnafm_frozen",    "exp3_rnafm_frozen"),
    ("EXP1_frozen", "RNA-MSM (frz)",  "rnamsm",  "exp1_rnamsm_frozen",   "exp3_rnamsm_frozen"),

    # ── EXP1 RNA LM Trainable ──────────────────────────────────────────────
    ("EXP1_trainable", "RNABERT (tr)",  "rnabert", "exp1_rnabert_trainable",  "exp3_rnabert_trainable"),
    ("EXP1_trainable", "RNAErnie (tr)", "rnaernie","exp1_rnaernie_trainable", "exp3_rnaernie_trainable"),
    ("EXP1_trainable", "RNA-MSM (tr)",  "rnamsm",  "exp1_rnamsm_trainable",   "exp3_rnamsm_trainable"),

    # ── EXP2: _scan_exp2()로 동적 추가 ────────────────────────────────────

    # ── EXP4 Ablation ──────────────────────────────────────────────────────
    ("EXP4_ablation", "Full (CircMAC)",  "circmac", "exp4_full",         None),
    ("EXP4_ablation", "No Attn",         "circmac", "exp4_no_attn",      None),
    ("EXP4_ablation", "No Mamba",        "circmac", "exp4_no_mamba",     None),
    ("EXP4_ablation", "No Conv",         "circmac", "exp4_no_conv",      None),
    ("EXP4_ablation", "No Circ Bias",    "circmac", "exp4_no_circ_bias", "exp4_no_circular_bias"),
    ("EXP4_ablation", "No Circ Pad",     "circmac", "exp4_no_circ_pad",  "exp4_no_circular_pad"),
    ("EXP4_ablation", "Attn Only",       "circmac", "exp4_attn_only",    None),
    ("EXP4_ablation", "Mamba Only",      "circmac", "exp4_mamba_only",   None),
    ("EXP4_ablation", "CNN Only",        "circmac", "exp4_cnn_only",     None),

    # ── EXP5 Interaction ───────────────────────────────────────────────────
    ("EXP5_interaction", "Cross-Attn",  "circmac", "exp5_cross_attn",  "exp5_cross_attention"),
    ("EXP5_interaction", "Concat",      "circmac", "exp5_concat",      None),
    ("EXP5_interaction", "Elementwise", "circmac", "exp5_elementwise", None),

    # ── EXP6 Site Head ─────────────────────────────────────────────────────
    ("EXP6_site_head", "Conv1D",  "circmac", "exp6_conv1d", None),
    ("EXP6_site_head", "Linear",  "circmac", "exp6_linear", None),
]

# ── EXP2 최종 정의 (ptcmp = run_pretrain_comparison.sh 결과) ───────────────────
# ptcmp naming: BS=64 for finetune (same as EXP1)
# old_exp_prefix: fallback for checking progress before ptcmp is done
EXP2_FINAL = [
    ("EXP2_pretrain", "No PT",    "circmac", "ptcmp_nopt",    "exp2v4_nopt_sites"),
    ("EXP2_pretrain", "MLM",      "circmac", "ptcmp_mlm",     "exp2v4_mlm_sites"),
    ("EXP2_pretrain", "NTP",      "circmac", "ptcmp_ntp",     "exp2_ntp_sites"),
    ("EXP2_pretrain", "SSP",      "circmac", "ptcmp_ssp",     "exp2v4_ssp_sites"),
    ("EXP2_pretrain", "Pairing",  "circmac", "ptcmp_pair",    "exp2v4_pair_sites"),
    ("EXP2_pretrain", "CPCL",     "circmac", "ptcmp_cpcl",    "exp2v4_cpcl_sites"),
    ("EXP2_pretrain", "MLM+NTP",  "circmac", "ptcmp_mlm_ntp", "exp2_mlm_ntp_sites"),
    ("EXP2_pretrain", "All",      "circmac", "ptcmp_all",     "exp2_mlm_ntp_cpcl_pair_sites"),
]

SEEDS = [1, 2, 3]

GROUP_ORDER = ["EXP1_base", "EXP1_frozen", "EXP1_trainable",
               "EXP2_pretrain",
               "EXP4_ablation", "EXP5_interaction", "EXP6_site_head"]
GROUP_TITLES = {
    "EXP1_base":        "EXP1 — Base Encoder Comparison",
    "EXP1_frozen":      "EXP1 — RNA LM Frozen",
    "EXP1_trainable":   "EXP1 — RNA LM Trainable",
    "EXP2_pretrain":    "EXP2 — Pretraining Strategy (CircMAC)",
    "EXP4_ablation":    "EXP4 — CircMAC Ablation",
    "EXP5_interaction": "EXP5 — Interaction Mechanism",
    "EXP6_site_head":   "EXP6 — Site Prediction Head",
}

# ── EXP2 자동 스캔 (--all 모드) ────────────────────────────────────────────────
def _scan_exp2_all() -> list:
    """
    logs/circmac/exp2*/ 를 전부 스캔해서 반환 (버전 구분 없이 모두 표시).
    """
    import re
    from collections import defaultdict
    circ_log = LOG_DIR / "circmac"
    if not circ_log.exists():
        return []

    prefix_seeds: dict = defaultdict(set)
    for p in circ_log.iterdir():
        if not p.name.startswith("exp2"):
            continue
        m = re.match(r"^(exp2[^/]+)_s(\d+)$", p.name)
        if m:
            prefix_seeds[m.group(1)].add(int(m.group(2)))

    def version_key(pref):
        if pref.startswith("exp2v4"): return (0, pref)
        if pref.startswith("exp2v3"): return (1, pref)
        if pref.startswith("exp2v2"): return (2, pref)
        return (3, pref)

    rows = []
    for prefix in sorted(prefix_seeds.keys(), key=version_key):
        label = re.sub(r"^exp2v?\d*_", "", prefix).replace("_sites", "").replace("_", "+")
        # 버전 태그 추가
        ver = re.match(r"^(exp2v?\d*)", prefix).group(1)
        label = f"[{ver}] {label}"
        rows.append(("EXP2_pretrain", label, "circmac", prefix, None))
    return rows


# ── 결과 읽기 ──────────────────────────────────────────────────────────────────
def load_result(model: str, exp_prefix: str, seed: int) -> dict | None:
    """logs/{model}/{exp_prefix}_s{seed}/{seed}/training.json 에서 test F1 추출"""
    exp_name = f"{exp_prefix}_s{seed}"
    path = LOG_DIR / model / exp_name / str(seed) / "training.json"
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text())
        fk = list(d.get("final", {}).keys())
        if not fk:
            return None
        scores = d["final"][fk[0]].get("scores", {}).get("sites", {})
        return {
            "epoch":    int(fk[0]),
            "f1":       scores.get("f1_macro", 0.0),
            "f1_pos":   scores.get("f1_pos", 0.0),
            "roc_auc":  scores.get("roc_auc", 0.0),
            "auprc":    scores.get("auprc", 0.0),
            "span_f1":  scores.get("span_f1", 0.0),
        }
    except Exception:
        return None


def collect_results(final: bool = False, all_exp2: bool = False):
    """
    final=True  : final_v2 naming만 (old alt_prefix 무시, EXP2는 EXP2_FINAL만)
    all_exp2=True : EXP2를 전체 스캔 (이전 실험 포함)
    기본값       : final naming 우선, alt_prefix fallback 허용, EXP2_FINAL 사용
    """
    # EXP2 entries 결정
    if all_exp2:
        exp2_entries = _scan_exp2_all()
    else:
        exp2_entries = EXP2_FINAL  # 최종 정의만

    all_experiments = []
    for entry in EXPERIMENTS:
        if entry[0] == "EXP2_pretrain":
            continue
        all_experiments.append(entry)

    insert_idx = next((i for i, e in enumerate(all_experiments) if e[0] == "EXP4_ablation"), len(all_experiments))
    for e in reversed(exp2_entries):
        all_experiments.insert(insert_idx, e)

    rows = []
    for group, label, model, exp_prefix, alt_prefix in all_experiments:
        # --final 모드: alt_prefix 사용 안 함
        if final:
            alt_prefix = None
        seed_results = []
        for seed in SEEDS:
            r = load_result(model, exp_prefix, seed)
            if r is None and alt_prefix:
                r = load_result(model, alt_prefix, seed)
            seed_results.append(r)

        done    = sum(1 for r in seed_results if r is not None)
        f1s     = [r["f1"]    for r in seed_results if r is not None]
        span_f1s= [r["span_f1"] for r in seed_results if r is not None]
        epochs  = [r["epoch"] for r in seed_results if r is not None]

        rows.append({
            "group":    group,
            "label":    label,
            "model":    model,
            "exp":      exp_prefix,
            "done":     done,
            "total":    len(SEEDS),
            "f1_mean":  np.mean(f1s)     if f1s else None,
            "f1_std":   np.std(f1s)      if len(f1s) > 1 else 0.0,
            "span_mean":np.mean(span_f1s) if span_f1s else None,
            "epochs":   epochs,
            "seed_results": seed_results,
        })
    return rows


# ── 출력 ──────────────────────────────────────────────────────────────────────
def print_table(rows, show_span=False):
    for grp in GROUP_ORDER:
        grp_rows = [r for r in rows if r["group"] == grp]
        if not grp_rows:
            continue

        title = GROUP_TITLES[grp]
        done_count = sum(1 for r in grp_rows for _ in range(r["done"]))
        total_count = sum(r["total"] for r in grp_rows)

        print(f"\n{'='*72}")
        print(f"  {title}   [{done_count}/{total_count} seeds done]")
        print(f"{'='*72}")

        header = f"  {'Model':<22} {'Done':>6}  {'F1 (mean±std)':>16}"
        if show_span:
            header += f"  {'Span-F1':>10}"
        header += f"  {'Epochs'}"
        print(header)
        print(f"  {'-'*68}")

        for r in grp_rows:
            status = f"{r['done']}/{r['total']}"
            if r["f1_mean"] is not None:
                f1_str = f"{r['f1_mean']:.4f} ± {r['f1_std']:.4f}"
            else:
                f1_str = "—"

            span_str = f"{r['span_mean']:.4f}" if r["span_mean"] is not None else "—"
            ep_str   = ",".join(str(e) for e in r["epochs"]) if r["epochs"] else "—"

            # 완료 여부 표시
            marker = "✓" if r["done"] == r["total"] else ("…" if r["done"] > 0 else " ")
            line = f"  {marker} {r['label']:<21} {status:>6}  {f1_str:>16}"
            if show_span:
                line += f"  {span_str:>10}"
            line += f"  ep={ep_str}"
            print(line)

    print()


def save_csv(rows, path: Path):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group","label","model","exp","done","total",
                    "f1_mean","f1_std","span_f1_mean",
                    "f1_s1","f1_s2","f1_s3"])
        for r in rows:
            f1s = [sr["f1"] if sr else "" for sr in r["seed_results"]]
            w.writerow([r["group"], r["label"], r["model"], r["exp"],
                        r["done"], r["total"],
                        f"{r['f1_mean']:.4f}" if r["f1_mean"] else "",
                        f"{r['f1_std']:.4f}"  if r["f1_mean"] else "",
                        f"{r['span_mean']:.4f}" if r["span_mean"] else "",
                        *[f"{v:.4f}" if v else "" for v in f1s]])
    print(f"Saved: {path}")


def plot_summary(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    COLORS_GROUP = {
        "EXP1_base":        "#E67E22",
        "EXP1_frozen":      "#8E44AD",
        "EXP1_trainable":   "#9B59B6",
        "EXP4_ablation":    "#2980B9",
        "EXP5_interaction": "#27AE60",
        "EXP6_site_head":   "#E74C3C",
    }

    # completed rows only
    done_rows = [r for r in rows if r["f1_mean"] is not None]
    if not done_rows:
        print("No completed experiments to plot.")
        return

    n = len(done_rows)
    fig, ax = plt.subplots(figsize=(max(10, n * 0.6), 5))
    fig.patch.set_facecolor("white")

    x = np.arange(n)
    colors = [COLORS_GROUP.get(r["group"], "#999") for r in done_rows]
    means  = [r["f1_mean"] for r in done_rows]
    stds   = [r["f1_std"]  for r in done_rows]

    bars = ax.bar(x, means, yerr=stds, color=colors, edgecolor="white",
                  linewidth=0.8, capsize=3,
                  error_kw={"elinewidth": 1.5, "ecolor": "#555"})

    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, m + s + 0.005,
                f"{m:.3f}", ha="center", va="bottom", fontsize=7.5, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels([r["label"] for r in done_rows],
                       rotation=45, ha="right", fontsize=8.5)
    ax.set_ylabel("Test F1 (macro)", fontsize=11)
    ax.set_title("Intermediate Results — All Experiments", fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(m + s for m, s in zip(means, stds)) * 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    legend_patches = [mpatches.Patch(color=c, label=GROUP_TITLES[g].split("—")[1].strip())
                      for g, c in COLORS_GROUP.items()]
    ax.legend(handles=legend_patches, fontsize=8, loc="lower right", framealpha=0.9)

    plt.tight_layout()
    out = OUT_DIR / "interim_results_summary.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.savefig(OUT_DIR / "interim_results_summary.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp1",     action="store_true", help="EXP1 only")
    parser.add_argument("--exp2",     action="store_true", help="EXP2 pretraining only")
    parser.add_argument("--exp4",     action="store_true", help="EXP4 only")
    parser.add_argument("--exp56",    action="store_true", help="EXP5+6 only")
    parser.add_argument("--final",    action="store_true",
                        help="final_v2 naming만 표시 (이전 실험 제외)")
    parser.add_argument("--all_exp2", action="store_true",
                        help="EXP2 모든 버전 표시 (기본: EXP2_FINAL만)")
    parser.add_argument("--csv",      action="store_true", help="Save CSV")
    parser.add_argument("--plot",     action="store_true", help="Save bar chart")
    parser.add_argument("--span",     action="store_true", help="Show span-F1 column")
    args = parser.parse_args()

    rows = collect_results(final=args.final, all_exp2=args.all_exp2)

    # 필터
    if args.exp1:
        rows = [r for r in rows if r["group"].startswith("EXP1")]
    elif args.exp2:
        rows = [r for r in rows if r["group"] == "EXP2_pretrain"]
    elif args.exp4:
        rows = [r for r in rows if r["group"] == "EXP4_ablation"]
    elif args.exp56:
        rows = [r for r in rows if r["group"] in ("EXP5_interaction", "EXP6_site_head")]

    # 전체 완료율
    done_total = sum(r["done"] for r in rows)
    all_total  = sum(r["total"] for r in rows)
    print(f"\n{'='*72}")
    print(f"  전체 진행률: {done_total}/{all_total} seeds  ({done_total/all_total*100:.0f}%)")
    print(f"  logs 경로:   {LOG_DIR}")

    print_table(rows, show_span=args.span)

    if args.csv:
        save_csv(rows, OUT_DIR / "interim_results.csv")

    if args.plot:
        plot_summary(rows)


if __name__ == "__main__":
    main()
