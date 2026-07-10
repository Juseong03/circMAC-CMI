#!/usr/bin/env python3
"""
prepare_fasta.py
Converts df_test_final.pkl to per-pair FASTA files for miRanda/RNAhybrid/IntaRNA.

Output:
  {outdir}/pairs_index.pkl    — list of (pair_id, isoform_ID, miRNA_ID, sites, length)
  {outdir}/fasta/
    {pair_id}_circ.fa         — circRNA sequence (T→U, label in header)
    {pair_id}_mirna.fa        — miRNA sequence
  {outdir}/all_circrna.fa     — all unique circRNAs (for batch miRanda)
  {outdir}/all_mirna.fa       — all unique miRNAs
"""
import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def to_dna(seq: str) -> str:
    """RNA → DNA (U→T) for tools that prefer DNA input."""
    return seq.upper().replace("U", "T")


def to_rna(seq: str) -> str:
    return seq.upper().replace("T", "U")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_pairs", default="all",
                        help="Number of test pairs to process (default: all)")
    parser.add_argument("--outdir", default="results/baseline_tools")
    parser.add_argument("--data_file", default="data/df_test_final.pkl")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    fasta_dir = outdir / "fasta"
    fasta_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.data_file} ...")
    df = pd.read_pickle(ROOT / args.data_file)

    # Positive pairs only for fair comparison (tools predict binding sites)
    df_pos = df[df["binding"] == 1].reset_index(drop=True)
    df_neg = df[df["binding"] == 0].reset_index(drop=True)

    # Subsample if requested
    if args.n_pairs != "all":
        n = int(args.n_pairs)
        # keep balanced pos/neg
        n_pos = min(n // 2, len(df_pos))
        n_neg = min(n // 2, len(df_neg))
        df_pos = df_pos.sample(n_pos, random_state=42)
        df_neg = df_neg.sample(n_neg, random_state=42)
        print(f"  Subsampled: {n_pos} pos + {n_neg} neg pairs")

    df_eval = pd.concat([df_pos, df_neg], ignore_index=True)
    print(f"  Total evaluation pairs: {len(df_eval):,}")

    # ── Determine miRNA sequence column ──────────────────────────────────────
    mirna_col = None
    for c in ["miRNA", "mirna", "miRNA_seq"]:
        if c in df_eval.columns:
            mirna_col = c
            break
    if mirna_col is None:
        # Try to load from binding_miRNA_seq.csv
        mirna_map_path = ROOT / "data/binding_miRNA_seq.csv"
        if mirna_map_path.exists():
            mirna_map = pd.read_csv(mirna_map_path).groupby("miRNA_ID")["miRNA"].first().to_dict()
            df_eval["_mirna_seq"] = df_eval["miRNA_ID"].map(mirna_map)
            mirna_col = "_mirna_seq"
            print(f"  Loaded miRNA sequences from {mirna_map_path.name}")
        else:
            raise ValueError("Cannot find miRNA sequences. "
                             "Need column 'miRNA' or data/binding_miRNA_seq.csv")

    # ── Write per-pair FASTA files ────────────────────────────────────────────
    index = []
    skipped = 0

    for i, row in df_eval.iterrows():
        pair_id   = f"pair_{i:06d}"
        circ_seq  = str(row["circRNA"]).upper().replace("T", "U")
        mirna_seq = str(row[mirna_col]).upper().replace("T", "U")
        sites     = list(row["sites"]) if "sites" in df_eval.columns else None
        binding   = int(row["binding"])

        if not circ_seq or not mirna_seq or mirna_seq == "nan":
            skipped += 1
            continue

        # circRNA FASTA (DNA format — most tools prefer T over U)
        circ_fa = fasta_dir / f"{pair_id}_circ.fa"
        circ_fa.write_text(
            f">{pair_id}|{row['isoform_ID']}|binding={binding}\n"
            f"{to_dna(circ_seq)}\n"
        )

        # miRNA FASTA
        mirna_fa = fasta_dir / f"{pair_id}_mirna.fa"
        mirna_fa.write_text(
            f">{pair_id}|{row['miRNA_ID']}\n"
            f"{to_dna(mirna_seq)}\n"
        )

        index.append({
            "pair_id":    pair_id,
            "isoform_ID": row["isoform_ID"],
            "miRNA_ID":   row["miRNA_ID"],
            "mirna_seq":  mirna_seq,
            "circ_seq":   circ_seq,
            "sites":      sites,
            "binding":    binding,
            "length":     len(circ_seq),
        })

    print(f"  Written {len(index):,} pairs  (skipped {skipped})")

    # Save index
    idx_path = outdir / "pairs_index.pkl"
    with open(idx_path, "wb") as f:
        pickle.dump(index, f)
    print(f"  Saved index: {idx_path}")

    # ── Write batch FASTA (all unique sequences) ──────────────────────────────
    # all_circrna.fa: unique isoforms
    seen_circ = {}
    for item in index:
        if item["isoform_ID"] not in seen_circ:
            seen_circ[item["isoform_ID"]] = item["circ_seq"]

    with open(outdir / "all_circrna.fa", "w") as f:
        for iso_id, seq in seen_circ.items():
            safe_id = iso_id.replace(" ", "_")
            f.write(f">{safe_id}\n{to_dna(seq)}\n")
    print(f"  all_circrna.fa: {len(seen_circ):,} unique sequences")

    # all_mirna.fa: unique miRNAs
    seen_mirna = {}
    for item in index:
        if item["miRNA_ID"] not in seen_mirna:
            seen_mirna[item["miRNA_ID"]] = item["mirna_seq"]

    with open(outdir / "all_mirna.fa", "w") as f:
        for mir_id, seq in seen_mirna.items():
            f.write(f">{mir_id}\n{to_dna(seq)}\n")
    print(f"  all_mirna.fa: {len(seen_mirna):,} unique miRNAs")


if __name__ == "__main__":
    main()
