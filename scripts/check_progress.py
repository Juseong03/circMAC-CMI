#!/usr/bin/env python3
"""
check_progress.py — Experiment progress checker

Shows done / running / pending status per experiment.
  DONE    : saved_models/{model}/{exp}_s{seed}/{seed}/train/model.pth exists
  RUNNING : log file modified within the last N minutes (default: 10)
  PENDING : neither condition met

Usage:
    python scripts/check_progress.py
    python scripts/check_progress.py --group encoder
    python scripts/check_progress.py --recent 5       # minutes threshold for RUNNING
    python scripts/check_progress.py --seeds 1 2 3    # check specific seeds only
    python scripts/check_progress.py --verbose        # list incomplete seeds
"""

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAVED = ROOT / "saved_models"
LOGS  = ROOT / "logs" / "v2"

SEEDS_ALL = list(range(1, 11))

# ── Experiment definitions: (group, label, model_name, exp_template, log_subdir) ──
# exp_template: actual path = saved_models/{model}/{template}_s{seed}/{seed}/train/model.pth
# log_subdir  : logs/v2/{log_subdir}/{template}_s{seed}.log  (None = skip log check)

EXPERIMENTS = [
    # ── Encoder comparison ─────────────────────────────────────────────────
    ("encoder", "LSTM",        "lstm",        "v2_enc_lstm",        "enc"),
    ("encoder", "Transformer", "transformer", "v2_enc_transformer", "enc"),
    ("encoder", "Mamba",       "mamba",       "v2_enc_mamba",       "enc"),
    ("encoder", "Hymba",       "hymba",       "v2_enc_hymba",       "enc"),
    ("encoder", "CircMAC",     "circmac",     "v2_abl_full",        "abl"),

    # ── Ablation ──────────────────────────────────────────────────────────
    ("ablation", "CircMAC (full)",  "circmac", "v2_abl_full",         "abl"),
    ("ablation", "Attn only",       "circmac", "v2_abl_attn_only",    "abl"),
    ("ablation", "Mamba only",      "circmac", "v2_abl_mamba_only",   "abl"),
    ("ablation", "CNN only",        "circmac", "v2_abl_cnn_only",     "abl"),
    ("ablation", "No Attn",         "circmac", "v2_abl_no_attn",      "abl"),
    ("ablation", "No Mamba",        "circmac", "v2_abl_no_mamba",     "abl"),
    ("ablation", "No Conv",         "circmac", "v2_abl_no_conv",      "abl"),
    ("ablation", "No CircBias",     "circmac", "v2_abl_no_circ_bias", "abl"),

    # ── Interaction mechanism ─────────────────────────────────────────────
    ("interaction", "Concat",      "circmac", "v2_int_concat",      "int"),
    ("interaction", "Elementwise", "circmac", "v2_int_elementwise", "int"),
    ("interaction", "Cross-Attn",  "circmac", "v2_int_cross_attn",  "int"),

    # ── Site head ─────────────────────────────────────────────────────────
    ("site_head", "Conv1D", "circmac", "v2_head_conv1d", "head"),
    ("site_head", "Linear", "circmac", "v2_head_linear", "head"),

    # ── Pretraining strategy (finetune) ───────────────────────────────────
    ("pretraining", "NoPT",          "circmac", "v2_pt_nopt",          "pt"),
    ("pretraining", "MLM",           "circmac", "v2_pt_mlm",           "pt"),
    ("pretraining", "NTP",           "circmac", "v2_pt_ntp",           "pt"),
    ("pretraining", "SSP",           "circmac", "v2_pt_ssp",           "pt"),
    ("pretraining", "CPCL",          "circmac", "v2_pt_cpcl",          "pt"),
    ("pretraining", "BSJ",           "circmac", "v2_pt_bsj",           "pt"),
    ("pretraining", "MLM+NTP",       "circmac", "v2_pt_mlm_ntp",       "pt"),
    ("pretraining", "MLM+SSP",       "circmac", "v2_pt_mlm_ssp",       "pt"),
    ("pretraining", "MLM+CPCL",      "circmac", "v2_pt_mlm_cpcl",      "pt"),
    ("pretraining", "Pairing",       "circmac", "v2_pt_pairing",       "pt"),
    ("pretraining", "MLM+CPCL+SSP",  "circmac", "v2_pt_mlm_cpcl_ssp",  "pt"),
    ("pretraining", "All",           "circmac", "v2_pt_all",           "pt"),

    # ── RNA-LM Frozen ─────────────────────────────────────────────────────
    ("rna_frozen", "RNABERT (frozen)",  "rnabert",  "exp1_fair_frozen_rnabert",  "rna_frozen"),
    ("rna_frozen", "RNAErnie (frozen)", "rnaernie", "exp1_fair_frozen_rnaernie", "rna_frozen"),
    ("rna_frozen", "RNA-MSM (frozen)",  "rnamsm",   "exp1_fair_frozen_rnamsm",   "rna_frozen"),
    ("rna_frozen", "RNA-FM (frozen)",   "rnafm",    "exp1_fair_frozen_rnafm",    "rna_frozen"),

    # ── RNA-LM Trainable ──────────────────────────────────────────────────
    ("rna_trainable", "RNABERT (ft)",  "rnabert",  "exp1_fair_trainable_rnabert",  "rna_trainable"),
    ("rna_trainable", "RNAErnie (ft)", "rnaernie", "exp1_fair_trainable_rnaernie", "rna_trainable"),
    ("rna_trainable", "RNA-MSM (ft)",  "rnamsm",   "exp1_fair_trainable_rnamsm",   "rna_trainable"),
    ("rna_trainable", "RNA-FM (ft)",   "rnafm",    "exp1_fair_trainable_rnafm",    "rna_trainable"),

    # ── Isoform-disjoint split ────────────────────────────────────────────
    ("iso_disjoint", "CircMAC (NoPT)", "circmac",     "iso_circmac",        None),
    ("iso_disjoint", "CircMAC (MLM)",  "circmac",     "iso_pt_mlm",         None),
    ("iso_disjoint", "CircMAC (NTP)",  "circmac",     "iso_pt_ntp",         None),
    ("iso_disjoint", "CircMAC (SSP)",  "circmac",     "iso_pt_ssp",         None),
    ("iso_disjoint", "CircMAC (CPCL)", "circmac",     "iso_pt_cpcl",        None),
    ("iso_disjoint", "CircMAC (BSJ)",  "circmac",     "iso_pt_bsj",         None),
    ("iso_disjoint", "CircMAC (MLM+NTP)",     "circmac", "iso_pt_mlm_ntp",     None),
    ("iso_disjoint", "CircMAC (MLM+SSP)",     "circmac", "iso_pt_mlm_ssp",     None),
    ("iso_disjoint", "CircMAC (MLM+CPCL)",    "circmac", "iso_pt_mlm_cpcl",    None),
    ("iso_disjoint", "CircMAC (Pairing)",     "circmac", "iso_pt_pairing",     None),
    ("iso_disjoint", "CircMAC (MLM+CPCL+SSP)","circmac","iso_pt_mlm_cpcl_ssp",None),
    ("iso_disjoint", "CircMAC (All)",  "circmac",     "iso_pt_all",         None),
    ("iso_disjoint", "Hymba",          "hymba",       "iso_hymba",       None),
    ("iso_disjoint", "Mamba",          "mamba",       "iso_mamba",       None),
    ("iso_disjoint", "LSTM",           "lstm",        "iso_lstm",        None),
    ("iso_disjoint", "Transformer",    "transformer", "iso_transformer", None),
    ("iso_disjoint", "RNABERT (ft)",   "rnabert",     "iso_rnabert_ft",  None),
    ("iso_disjoint", "RNAErnie (ft)",  "rnaernie",    "iso_rnaernie_ft", None),
    ("iso_disjoint", "RNAMSM (ft)",    "rnamsm",      "iso_rnamsm_ft",   None),
    ("iso_disjoint", "RNA-FM (ft)",    "rnafm",       "iso_rnafm_ft",    None),

    # ── BSJ-disjoint split ────────────────────────────────────────────────
    ("bsj_disjoint", "CircMAC (NoPT)", "circmac",     "bsj_circmac",        None),
    ("bsj_disjoint", "CircMAC (MLM)",  "circmac",     "bsj_pt_mlm",         None),
    ("bsj_disjoint", "CircMAC (NTP)",  "circmac",     "bsj_pt_ntp",         None),
    ("bsj_disjoint", "CircMAC (SSP)",  "circmac",     "bsj_pt_ssp",         None),
    ("bsj_disjoint", "CircMAC (CPCL)", "circmac",     "bsj_pt_cpcl",        None),
    ("bsj_disjoint", "CircMAC (BSJ)",  "circmac",     "bsj_pt_bsj",         None),
    ("bsj_disjoint", "CircMAC (MLM+NTP)",     "circmac", "bsj_pt_mlm_ntp",     None),
    ("bsj_disjoint", "CircMAC (MLM+SSP)",     "circmac", "bsj_pt_mlm_ssp",     None),
    ("bsj_disjoint", "CircMAC (MLM+CPCL)",    "circmac", "bsj_pt_mlm_cpcl",    None),
    ("bsj_disjoint", "CircMAC (Pairing)",     "circmac", "bsj_pt_pairing",     None),
    ("bsj_disjoint", "CircMAC (MLM+CPCL+SSP)","circmac","bsj_pt_mlm_cpcl_ssp",None),
    ("bsj_disjoint", "CircMAC (All)",  "circmac",     "bsj_pt_all",         None),
    ("bsj_disjoint", "Hymba",          "hymba",       "bsj_hymba",       None),
    ("bsj_disjoint", "Mamba",          "mamba",       "bsj_mamba",       None),
    ("bsj_disjoint", "LSTM",           "lstm",        "bsj_lstm",        None),
    ("bsj_disjoint", "Transformer",    "transformer", "bsj_transformer", None),
    ("bsj_disjoint", "RNABERT (ft)",   "rnabert",     "bsj_rnabert_ft",  None),
    ("bsj_disjoint", "RNAErnie (ft)",  "rnaernie",    "bsj_rnaernie_ft", None),
    ("bsj_disjoint", "RNAMSM (ft)",    "rnamsm",      "bsj_rnamsm_ft",   None),
    ("bsj_disjoint", "RNA-FM (ft)",    "rnafm",       "bsj_rnafm_ft",    None),
]

# ── Pretraining checkpoints (model.pth, seed=42) ─────────────────────────────
PRETRAIN_STRATEGIES = [
    ("mlm",          "--mlm"),
    ("ntp",          "--ntp"),
    ("mlm_ssp",      "--mlm --ssp"),
    ("mlm_cpcl",     "--mlm --cpcl"),
    ("mlm_ntp",      "--mlm --ntp"),
    ("all",          "--mlm --ntp --ssp --pairing --cpcl --bsj_mlm"),
    ("ssp",          "--ssp"),
    ("cpcl",         "--cpcl"),
    ("bsj",          "--bsj_mlm"),
    ("pairing",      "--pairing"),
    ("mlm_cpcl_ssp", "--mlm --cpcl --ssp"),
]


def model_done(model_name: str, exp_template: str, seed: int) -> bool:
    """Returns True if the trained model checkpoint exists."""
    p = SAVED / model_name / f"{exp_template}_s{seed}" / str(seed) / "train" / "model.pth"
    return p.exists()


def log_running(log_subdir: str, exp_template: str, seed: int, recent_min: int) -> bool:
    """Returns True if the log file was modified within recent_min minutes."""
    if not log_subdir:
        return False
    log_path = LOGS / log_subdir / f"{exp_template}_s{seed}.log"
    if not log_path.exists():
        return False
    age_sec = time.time() - log_path.stat().st_mtime
    return age_sec <= recent_min * 60


def pretrain_done(strategy: str) -> bool:
    p = SAVED / "circmac" / f"v2_ptm_{strategy}" / "42" / "pretrain" / "model.pth"
    return p.exists()


def bar(done: int, total: int, width: int = 20) -> str:
    filled = int(width * done / total) if total else 0
    pct = int(100 * done / total) if total else 0
    return f"[{'#' * filled}{'-' * (width - filled)}] {done:>3}/{total} ({pct}%)"


GROUPS_ORDER = [
    "encoder", "ablation", "interaction", "site_head",
    "pretraining", "rna_frozen", "rna_trainable",
    "iso_disjoint", "bsj_disjoint",
]

GROUP_LABELS = {
    "encoder":      "EXP1  Encoder Comparison",
    "ablation":     "EXP4  CircMAC Ablation",
    "interaction":  "EXP5  Interaction Mechanism",
    "site_head":    "EXP6  Site Head",
    "pretraining":  "EXP2  Pretraining Strategy",
    "rna_frozen":   "EXP1  RNA-LM Frozen",
    "rna_trainable":"EXP1  RNA-LM Trainable",
    "iso_disjoint": "REV   Isoform-Disjoint Split",
    "bsj_disjoint": "REV   BSJ-Disjoint Split",
}


def main():
    parser = argparse.ArgumentParser(description="Experiment progress checker")
    parser.add_argument("--group", nargs="*", choices=list(GROUP_LABELS.keys()),
                        help="Show specific groups only (default: all)")
    parser.add_argument("--seeds", nargs="*", type=int, default=None,
                        help="Seeds to check (default: 1-10)")
    parser.add_argument("--recent", type=int, default=10,
                        help="Minutes threshold for RUNNING detection (default: 10)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="List incomplete seeds per experiment")
    args = parser.parse_args()

    seeds = args.seeds or SEEDS_ALL
    target_groups = set(args.group) if args.group else set(GROUP_LABELS.keys())

    # ── Per-group aggregation ─────────────────────────────────────────────────
    # Deduplicate by (group, template) — e.g. v2_abl_full appears in both encoder and ablation
    seen_templates = set()
    group_data: dict[str, list] = {g: [] for g in GROUPS_ORDER}

    for group, label, model, template, log_sub in EXPERIMENTS:
        if group not in target_groups:
            continue
        key = (group, template)
        if key in seen_templates:
            continue
        seen_templates.add(key)

        seed_status = {}  # seed -> "done" | "running" | "pending"
        for s in seeds:
            if model_done(model, template, s):
                seed_status[s] = "done"
            elif log_running(log_sub, template, s, args.recent):
                seed_status[s] = "running"
            else:
                seed_status[s] = "pending"

        group_data[group].append({
            "label":    label,
            "model":    model,
            "template": template,
            "status":   seed_status,
        })

    # ── Print output ──────────────────────────────────────────────────────────
    now_str = time.strftime("%Y-%m-%d %H:%M:%S")
    total_done = total_run = total_pend = 0

    print()
    print("=" * 70)
    print(f"  Experiment Progress   (seeds: {seeds[0]}–{seeds[-1]},  recent={args.recent}min)")
    print(f"  {now_str}")
    print("=" * 70)

    # ── Pretraining checkpoint status ─────────────────────────────────────
    if "pretraining" in target_groups:
        pt_done = sum(1 for s, _ in PRETRAIN_STRATEGIES if pretrain_done(s))
        pt_total = len(PRETRAIN_STRATEGIES)
        print()
        print(f"  [Pretrain checkpoints]  {bar(pt_done, pt_total)}  ({pt_done}/{pt_total} done)")
        if args.verbose or pt_done < pt_total:
            for strat, flags in PRETRAIN_STRATEGIES:
                mark = "[Y]" if pretrain_done(strat) else "[ ]"
                print(f"    {mark}  v2_ptm_{strat:<20}  {flags}")

    # ── Per-group status ───────────────────────────────────────────────────
    for group in GROUPS_ORDER:
        if group not in target_groups:
            continue
        rows = group_data[group]
        if not rows:
            continue

        n_seeds = len(seeds)
        g_done = g_run = g_pend = 0
        for r in rows:
            for s in seeds:
                st = r["status"].get(s, "pending")
                if st == "done":    g_done += 1
                elif st == "running": g_run += 1
                else:               g_pend += 1

        g_total = len(rows) * n_seeds
        total_done += g_done; total_run += g_run; total_pend += g_pend

        print()
        print(f"  ── {GROUP_LABELS[group]} ──")
        print(f"     Progress: {bar(g_done, g_total)}  "
              f"(done={g_done}, running={g_run}, pending={g_pend})")

        seed_header = "  ".join(f"s{s:02d}" for s in seeds)
        print(f"     {'Model':<28}  {seed_header}")
        print(f"     {'-'*28}  {'-'*5*n_seeds}")
        for r in rows:
            marks = []
            for s in seeds:
                st = r["status"].get(s, "pending")
                if st == "done":      marks.append(" O ")
                elif st == "running": marks.append("[>]")
                else:                 marks.append(" . ")
            print(f"     {r['label']:<28}  {'  '.join(marks)}")

            if args.verbose:
                pending_seeds = [s for s in seeds if r["status"].get(s) != "done"]
                if pending_seeds:
                    print(f"       └─ pending seeds: {pending_seeds}")

    # ── Overall summary ────────────────────────────────────────────────────
    grand_total = total_done + total_run + total_pend
    print()
    print("=" * 70)
    print(f"  Overall: {bar(total_done, grand_total, 30)}")
    print(f"    done={total_done}  running={total_run}  pending={total_pend}  "
          f"total={grand_total}")
    print("=" * 70)
    print()

    # ── Currently running processes ────────────────────────────────────────
    print("  [Running processes]")
    result = os.popen("ps aux | grep 'python training\\|python pretraining' | grep -v grep").read().strip()
    if result:
        for line in result.splitlines():
            parts = line.split()
            pid  = parts[1]
            cpu  = parts[2]
            mem  = parts[3]
            cmd  = " ".join(parts[10:])
            exp_name = ""
            if "--exp" in cmd:
                idx = cmd.split().index("--exp")
                exp_name = cmd.split()[idx + 1] if idx + 1 < len(cmd.split()) else ""
            gpu_val = ""
            if "--device" in cmd:
                idx = cmd.split().index("--device")
                gpu_val = "GPU" + cmd.split()[idx + 1] if idx + 1 < len(cmd.split()) else ""
            print(f"    PID={pid:<7} {gpu_val:<5} CPU={cpu}%  MEM={mem}%  exp={exp_name}")
    else:
        print("    (no running experiments)")
    print()


if __name__ == "__main__":
    main()
