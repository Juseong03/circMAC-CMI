#!/usr/bin/env python3
"""
revision_data_analysis.py
=========================
Computes all dataset statistics needed for the circMAC revision.

Covers:
  Section 2  — Length filtering statistics (150–1000 nt)
  Section 3  — Train/val/test split leakage analysis
  Section 6  — Multi-site binding pair statistics
  Section 7  — Label sparsity statistics
  Section 9  — Full dataset summary table

Output:
  results/revision/  (created automatically)
    data_summary.txt        — all computed statistics (console + file)
    split_overlap.csv       — shared isoform / BSJ / gene / miRNA counts
    length_dist_raw.csv     — circRNA lengths before filtering
    multisite_stats.csv     — per-pair binding segment counts
    label_sparsity.csv      — per-pair positive ratio

Usage:
    python scripts/revision_data_analysis.py
    python scripts/revision_data_analysis.py --outdir results/revision
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_DEFAULT = ROOT / "results" / "revision"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_isoform_id(isoform_id: str):
    """
    Parse isoform_ID into components.
    Expected formats:
      'chr4|84678168|84679116|+'            → single-BSJ
      'chr6|130955105,130955994|130955317,130956499|-'  → multi-exon
    Returns dict: chrom, bsj_start, bsj_end, strand, host_gene_key, bsj_key
    """
    parts = isoform_id.split("|")
    chrom  = parts[0] if len(parts) > 0 else ""
    starts = parts[1] if len(parts) > 1 else ""
    ends   = parts[2] if len(parts) > 2 else ""
    strand = parts[3] if len(parts) > 3 else ""

    # BSJ = outermost donor–acceptor coordinates
    bsj_start = starts.split(",")[0]   # leftmost start
    bsj_end   = ends.split(",")[-1]    # rightmost end
    bsj_key   = f"{chrom}|{bsj_start}|{bsj_end}|{strand}"
    host_gene_key = f"{chrom}|{strand}"  # coarse host-gene proxy

    return dict(
        chrom=chrom,
        bsj_start=bsj_start,
        bsj_end=bsj_end,
        strand=strand,
        bsj_key=bsj_key,
        host_gene_key=host_gene_key,
    )


def count_binding_segments(sites) -> int:
    """Count number of contiguous positive (=1) segments in a sites array."""
    arr = np.asarray(sites, dtype=int)
    if arr.sum() == 0:
        return 0
    # count transitions 0→1
    padded = np.concatenate([[0], arr, [0]])
    starts = np.where(np.diff(padded) == 1)[0]
    return len(starts)


def section(title: str, lines: list[str], out_f):
    header = f"\n{'='*70}\n  {title}\n{'='*70}"
    print(header)
    out_f.write(header + "\n")
    for line in lines:
        print(line)
        out_f.write(line + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=str(OUT_DEFAULT))
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out_txt = open(outdir / "data_summary.txt", "w")

    # ── Load data ────────────────────────────────────────────────────────────
    print("Loading data files...")
    df_train = pd.read_pickle(ROOT / "data/df_train_final.pkl")
    df_test  = pd.read_pickle(ROOT / "data/df_test_final.pkl")

    # Validation split (recreate from train using seed=42, same as trainer)
    from sklearn.model_selection import train_test_split
    df_tr, df_val = train_test_split(df_train, test_size=0.1, random_state=42)
    df_tr  = df_tr.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)

    # Raw (before length filtering) — use df_final.pkl if available,
    # else reconstruct from train_raw_filtered + test_raw_filtered
    raw_paths = [
        ROOT / "data/df_final.pkl",
        ROOT / "data/df_train_raw_filtered.pkl",
    ]
    df_raw = None
    for p in raw_paths:
        if p.exists():
            print(f"  Using raw data: {p.name}")
            df_raw = pd.read_pickle(p)
            if "length" not in df_raw.columns and "circRNA" in df_raw.columns:
                df_raw["length"] = df_raw["circRNA"].str.len()
            break
    if df_raw is None:
        print("  [WARN] No raw (pre-filter) data file found. Skipping length-filter stats.")

    print(f"  Train: {len(df_train):,} rows")
    print(f"  Test : {len(df_test):,} rows")

    # ── Parse isoform IDs ────────────────────────────────────────────────────
    for df, tag in [(df_train, "train"), (df_test, "test"), (df_val, "val"),
                    (df_tr, "tr")]:
        parsed = df["isoform_ID"].apply(parse_isoform_id)
        df["bsj_key"]        = parsed.apply(lambda x: x["bsj_key"])
        df["host_gene_key"]  = parsed.apply(lambda x: x["host_gene_key"])

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 2 — Length filtering
    # ─────────────────────────────────────────────────────────────────────────
    lines = []

    if df_raw is not None:
        n_raw      = len(df_raw)
        n_iso_raw  = df_raw["isoform_ID"].nunique() if "isoform_ID" in df_raw.columns else "N/A"
        raw_len    = df_raw["length"]
        n_in_range = ((raw_len >= 150) & (raw_len <= 1000)).sum()
        pct_in     = 100 * n_in_range / n_raw
        lines += [
            f"  Raw pairs (before 150-1000 nt filter) : {n_raw:>10,}",
            f"  Unique isoforms (raw)                  : {n_iso_raw!s:>10}",
            f"  Pairs within 150-1000 nt               : {n_in_range:>10,}  ({pct_in:.1f}%)",
            f"  Median length (raw)                    : {raw_len.median():>10.0f} nt",
            f"  Mean   length (raw)                    : {raw_len.mean():>10.1f} nt",
            f"  Min    length (raw)                    : {raw_len.min():>10} nt",
            f"  Max    length (raw)                    : {raw_len.max():>10} nt",
        ]
        # Save raw length distribution
        raw_len.to_frame("length").to_csv(outdir / "length_dist_raw.csv", index=False)
        lines.append(f"\n  → Saved raw length distribution: {outdir/'length_dist_raw.csv'}")
    else:
        lines.append("  [SKIP] No pre-filter data available.")

    n_final = len(df_train) + len(df_test)
    tr_len  = df_train["length"]
    te_len  = df_test["length"]
    all_len = pd.concat([tr_len, te_len])
    lines += [
        f"\n  After filtering (train+test)           : {n_final:>10,} pairs",
        f"  Median length (filtered)               : {all_len.median():>10.0f} nt",
        f"  Mean   length (filtered)               : {all_len.mean():>10.1f} nt",
        f"  Min    length (filtered)               : {all_len.min():>10} nt",
        f"  Max    length (filtered)               : {all_len.max():>10} nt",
    ]
    if df_raw is not None:
        lines.append(f"  Pair retention rate                    : {100*n_final/n_raw:>10.1f}%")

    section("2. LENGTH FILTERING STATISTICS", lines, out_txt)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 3 — Split leakage
    # ─────────────────────────────────────────────────────────────────────────
    def overlap_stats(dfA, dfB, nameA, nameB):
        pair_A   = set(zip(dfA["isoform_ID"], dfA["miRNA_ID"]))
        pair_B   = set(zip(dfB["isoform_ID"], dfB["miRNA_ID"]))
        iso_A    = set(dfA["isoform_ID"]);   iso_B  = set(dfB["isoform_ID"])
        bsj_A    = set(dfA["bsj_key"]);      bsj_B  = set(dfB["bsj_key"])
        gene_A   = set(dfA["host_gene_key"]); gene_B = set(dfB["host_gene_key"])
        mirna_A  = set(dfA["miRNA_ID"]);     mirna_B= set(dfB["miRNA_ID"])

        return {
            "comparison":             f"{nameA} vs {nameB}",
            "shared_pairs":           len(pair_A  & pair_B),
            "shared_isoforms":        len(iso_A   & iso_B),
            "pct_isoform_in_A":       100 * len(iso_A & iso_B) / len(iso_A),
            "pct_isoform_in_B":       100 * len(iso_A & iso_B) / len(iso_B),
            "shared_BSJs":            len(bsj_A   & bsj_B),
            "shared_host_genes":      len(gene_A  & gene_B),
            "shared_miRNAs":          len(mirna_A & mirna_B),
            "unique_isoforms_A":      len(iso_A),
            "unique_isoforms_B":      len(iso_B),
            "unique_BSJs_A":          len(bsj_A),
            "unique_BSJs_B":          len(bsj_B),
        }

    rows = [
        overlap_stats(df_tr,  df_val, "Train",      "Validation"),
        overlap_stats(df_tr,  df_test,"Train",      "Test"),
        overlap_stats(df_val, df_test,"Validation", "Test"),
    ]
    df_overlap = pd.DataFrame(rows)
    df_overlap.to_csv(outdir / "split_overlap.csv", index=False)

    lines = []
    for r in rows:
        lines += [
            f"\n  {r['comparison']}",
            f"    Shared exact pairs      : {r['shared_pairs']:>6}",
            f"    Shared circRNA isoforms : {r['shared_isoforms']:>6}  "
            f"({r['pct_isoform_in_A']:.1f}% of A,  {r['pct_isoform_in_B']:.1f}% of B)",
            f"    Shared BSJs             : {r['shared_BSJs']:>6}",
            f"    Shared host genes       : {r['shared_host_genes']:>6}",
            f"    Shared miRNAs           : {r['shared_miRNAs']:>6}",
        ]

    # Split type determination
    train_iso = set(df_tr["isoform_ID"]); test_iso = set(df_test["isoform_ID"])
    shared_iso = train_iso & test_iso
    if len(shared_iso) == 0:
        split_type = "ISOFORM-DISJOINT (no leakage)"
    elif len(set(zip(df_tr["isoform_ID"], df_tr["miRNA_ID"])) &
             set(zip(df_test["isoform_ID"], df_test["miRNA_ID"]))) == 0:
        split_type = "PAIR-DISJOINT (isoform leakage present)"
    else:
        split_type = "PAIR-LEVEL OVERLAP (full leakage)"

    lines = [f"  Current split type: {split_type}"] + lines
    lines.append(f"\n  → Saved split overlap: {outdir/'split_overlap.csv'}")
    section("3. TRAIN/VAL/TEST SPLIT LEAKAGE", lines, out_txt)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 6 — Multi-site binding
    # ─────────────────────────────────────────────────────────────────────────
    # Use only positive pairs (binding=1) for meaningful multi-site analysis
    df_pos = df_test[df_test["binding"] == 1].copy()

    if "sites" in df_pos.columns:
        df_pos["n_segments"] = df_pos["sites"].apply(count_binding_segments)
        n_total  = len(df_pos)
        n_single = (df_pos["n_segments"] == 1).sum()
        n_multi  = (df_pos["n_segments"] >= 2).sum()
        n_zero   = (df_pos["n_segments"] == 0).sum()
        max_seg  = df_pos["n_segments"].max()

        df_pos[["isoform_ID","miRNA_ID","length","n_segments"]].to_csv(
            outdir / "multisite_stats.csv", index=False)

        lines = [
            f"  Positive test pairs total  : {n_total:>7,}",
            f"  Zero-site pairs (anomaly)  : {n_zero:>7,}",
            f"  Single-site pairs          : {n_single:>7,}  ({100*n_single/n_total:.1f}%)",
            f"  Multi-site pairs (>=2)     : {n_multi:>7,}  ({100*n_multi/n_total:.1f}%)",
            f"  Max annotated sites/pair   : {max_seg:>7}",
            "",
        ]
        # top multi-site examples
        top = df_pos.nlargest(5, "n_segments")[
            ["isoform_ID","miRNA_ID","length","n_segments"]]
        lines.append("  Top multi-site examples:")
        for _, row in top.iterrows():
            lines.append(f"    {row['isoform_ID']}  x  {row['miRNA_ID']}"
                         f"  —  {row['n_segments']} sites  (len={row['length']})")
        lines.append(f"\n  → Saved multisite stats: {outdir/'multisite_stats.csv'}")
    else:
        lines = ["  [SKIP] 'sites' column not found in test data."]

    section("6. MULTI-SITE BINDING PAIRS", lines, out_txt)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 7 — Label sparsity
    # ─────────────────────────────────────────────────────────────────────────
    all_df = pd.concat([df_train, df_test], ignore_index=True)
    pos_df = all_df[all_df["binding"] == 1].copy()

    if "sites" in pos_df.columns:
        pos_df["n_pos_nt"] = pos_df["sites"].apply(lambda s: int(np.sum(s)))
        pos_df["n_total_nt"] = pos_df["length"]
        pos_df["pos_ratio"]  = pos_df["n_pos_nt"] / pos_df["n_total_nt"]

        total_nt  = all_df["length"].sum()
        pos_nt    = pos_df["n_pos_nt"].sum()
        pos_ratio_global = 100 * pos_nt / total_nt

        # Binding site lengths from segments
        seg_lengths = []
        for sites in pos_df["sites"]:
            arr = np.asarray(sites, dtype=int)
            padded = np.concatenate([[0], arr, [0]])
            starts = np.where(np.diff(padded) ==  1)[0]
            ends   = np.where(np.diff(padded) == -1)[0]
            seg_lengths.extend((ends - starts).tolist())
        seg_lengths = np.array(seg_lengths)

        pos_df[["isoform_ID","miRNA_ID","n_pos_nt","n_total_nt","pos_ratio"]].to_csv(
            outdir / "label_sparsity.csv", index=False)

        lines = [
            f"  Total nucleotides (all pairs)          : {total_nt:>12,}",
            f"  Positive nucleotides (binding=1 pairs) : {pos_nt:>12,}",
            f"  Global positive ratio                  : {pos_ratio_global:>11.2f}%",
            f"",
            f"  Per-pair positive ratio (binding=1):",
            f"    Median : {pos_df['pos_ratio'].median()*100:>8.2f}%",
            f"    Mean   : {pos_df['pos_ratio'].mean()*100:>8.2f}%",
            f"    Min    : {pos_df['pos_ratio'].min()*100:>8.2f}%",
            f"    Max    : {pos_df['pos_ratio'].max()*100:>8.2f}%",
            f"",
            f"  Binding segment lengths (nt):",
            f"    Count  : {len(seg_lengths):>8,}",
            f"    Median : {np.median(seg_lengths):>8.1f}",
            f"    Mean   : {np.mean(seg_lengths):>8.1f}",
            f"    Min    : {np.min(seg_lengths):>8}",
            f"    Max    : {np.max(seg_lengths):>8}",
            f"\n  → Saved label sparsity: {outdir/'label_sparsity.csv'}",
        ]
    else:
        lines = ["  [SKIP] 'sites' column not found."]

    section("7. LABEL SPARSITY", lines, out_txt)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 9 — Full dataset summary table
    # ─────────────────────────────────────────────────────────────────────────
    n_iso_train = df_train["isoform_ID"].nunique()
    n_iso_test  = df_test["isoform_ID"].nunique()
    n_iso_all   = pd.concat([df_train["isoform_ID"], df_test["isoform_ID"]]).nunique()
    n_bsj_all   = pd.concat([df_train["bsj_key"], df_test["bsj_key"]]).nunique()
    n_gene_all  = pd.concat([df_train["host_gene_key"], df_test["host_gene_key"]]).nunique()
    n_mirna_all = pd.concat([df_train["miRNA_ID"], df_test["miRNA_ID"]]).nunique()

    n_tr_pos    = (df_tr["binding"] == 1).sum()
    n_tr_neg    = (df_tr["binding"] == 0).sum()
    n_val_pos   = (df_val["binding"] == 1).sum()
    n_val_neg   = (df_val["binding"] == 0).sum()
    n_te_pos    = (df_test["binding"] == 1).sum()
    n_te_neg    = (df_test["binding"] == 0).sum()

    miRNA_seq_col = None
    for c in ["miRNA","mirna","miRNA_seq"]:
        if c in df_train.columns:
            miRNA_seq_col = c
            break
    if miRNA_seq_col:
        med_mirna_len = df_train[miRNA_seq_col].str.len().median()
    else:
        med_mirna_len = "N/A"

    lines = [
        f"  {'Statistic':<45}  {'Value':>12}",
        f"  {'-'*58}",
    ]

    if df_raw is not None and "isoform_ID" in df_raw.columns:
        lines.append(f"  {'Full-length isoforms before filtering':<45}  {df_raw['isoform_ID'].nunique():>12,}")
        lines.append(f"  {'Pairs before filtering':<45}  {len(df_raw):>12,}")
    lines += [
        f"  {'Full-length isoforms after filtering':<45}  {n_iso_all:>12,}",
        f"  {'Pairs after filtering (total)':<45}  {len(df_train)+len(df_test):>12,}",
        f"  {'  Training pairs (pos / neg)':<45}  {len(df_tr):>12,}  ({n_tr_pos:,} / {n_tr_neg:,})",
        f"  {'  Validation pairs (pos / neg)':<45}  {len(df_val):>12,}  ({n_val_pos:,} / {n_val_neg:,})",
        f"  {'  Test pairs (pos / neg)':<45}  {len(df_test):>12,}  ({n_te_pos:,} / {n_te_neg:,})",
        f"  {'Unique circRNA isoforms':<45}  {n_iso_all:>12,}",
        f"  {'Unique BSJs':<45}  {n_bsj_all:>12,}",
        f"  {'Unique host genes (chrom+strand)':<45}  {n_gene_all:>12,}",
        f"  {'Unique mature miRNAs':<45}  {n_mirna_all:>12,}",
        f"  {'Median circRNA length (filtered)':<45}  {all_len.median():>11.0f} nt",
        f"  {'Mean circRNA length (filtered)':<45}  {all_len.mean():>11.1f} nt",
        f"  {'Median miRNA length':<45}  {med_mirna_len!s:>12}",
    ]
    if "sites" in all_df.columns:
        lines += [
            f"  {'Total nucleotides':<45}  {total_nt:>12,}",
            f"  {'Positive nucleotides':<45}  {pos_nt:>12,}",
            f"  {'Positive nucleotide ratio':<45}  {pos_ratio_global:>11.2f}%",
            f"  {'Median binding-site length':<45}  {np.median(seg_lengths):>11.1f} nt",
            f"  {'Multi-site pair proportion (test, pos)':<45}  {100*n_multi/n_total:>11.1f}%",
        ]

    section("9. DATASET SUMMARY TABLE", lines, out_txt)

    print(f"\n  All results saved to: {outdir}/")
    out_txt.write(f"\n  All results saved to: {outdir}/\n")
    out_txt.close()


if __name__ == "__main__":
    main()
