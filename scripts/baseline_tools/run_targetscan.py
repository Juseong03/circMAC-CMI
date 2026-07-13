#!/usr/bin/env python3
"""
run_targetscan.py
Runs TargetScan 7.0 on every pair in pairs_index.pkl.

TargetScan uses seed-region matching (miRNA positions 2-8, 7mer).
Designed for linear mRNA 3'UTR — applied here to circRNA for comparison.

Input format:
  miRNA file : family_name  seed(7mer)  species_id
  UTR file   : gene_id      species_id  sequence

Strategy:
  - For each unique (miRNA, circRNA), check for seed matches in circRNA.
  - Predicted sites: positions where 7mer seed matches (with A1 or m8 extension).
  - Binding prediction: 1 if ≥1 site found, else 0.
"""
import argparse
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TARGETSCAN = next(
    (p for p in [
        "/root/.local/share/mamba/envs/targetscan_env/bin/targetscan_70.pl",
        "targetscan_70.pl",
    ] if Path(p).exists() or p == "targetscan_70.pl"),
    "targetscan_70.pl"
)

SPECIES_ID = "9606"  # human


def get_seed(mirna_seq: str) -> str:
    """Extract 7mer seed region: positions 2-8 of miRNA (1-indexed), RNA format."""
    seq = mirna_seq.upper().replace("T", "U")
    if len(seq) < 8:
        return seq[1:] if len(seq) > 1 else seq
    return seq[1:8]  # positions 2-8 (0-indexed: 1:8)


def run_targetscan_pair(mirna_id: str, mirna_seq: str,
                         circ_id: str, circ_seq: str) -> list[dict]:
    """Run TargetScan on one (miRNA, circRNA) pair. Returns list of hit dicts."""
    seed = get_seed(mirna_seq)
    # family name = miRNA_ID (safe version)
    family = mirna_id.replace(" ", "_").replace(";", "_")

    with tempfile.TemporaryDirectory() as tmpdir:
        mirna_file = os.path.join(tmpdir, "mirna.txt")
        utr_file   = os.path.join(tmpdir, "utr.txt")
        out_file   = os.path.join(tmpdir, "out.txt")

        # miRNA file
        with open(mirna_file, "w") as f:
            f.write(f"{family}\t{seed}\t{SPECIES_ID}\n")

        # UTR file (circRNA as "UTR")
        circ_rna = circ_seq.upper().replace("T", "U")
        safe_id = circ_id.replace(" ", "_").replace(";", "_")
        with open(utr_file, "w") as f:
            f.write(f"{safe_id}\t{SPECIES_ID}\t{circ_rna}\n")

        try:
            result = subprocess.run(
                [TARGETSCAN, mirna_file, utr_file, out_file],
                capture_output=True, text=True, timeout=30
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return []

        if not os.path.exists(out_file):
            return []

        hits = []
        with open(out_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("Gene"):
                    continue
                parts = line.split("\t")
                # Output: Gene_ID | Species_ID | miRNA_family | site_type |
                #         UTR_start | UTR_end | ...
                if len(parts) < 6:
                    continue
                try:
                    t_start = int(parts[4]) - 1  # 1-based → 0-based
                    t_end   = int(parts[5])
                    site_type = parts[3]
                    hits.append({
                        "t_start":   max(0, t_start),
                        "t_end":     t_end,
                        "site_type": site_type,
                    })
                except (ValueError, IndexError):
                    continue

    return hits


def hits_to_site_mask(hits: list[dict], seq_len: int) -> np.ndarray:
    mask = np.zeros(seq_len, dtype=np.float32)
    for h in hits:
        s = max(0, h["t_start"])
        e = min(seq_len, h["t_end"])
        if s < e:
            mask[s:e] = 1.0
    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="results/baseline_tools")
    parser.add_argument("--split", default="pair", choices=["pair", "iso", "bsj"])
    args = parser.parse_args()

    split_outdir = Path(args.outdir) / args.split
    idx_path = split_outdir / "pairs_index.pkl"
    if not idx_path.exists():
        raise FileNotFoundError(f"Run prepare_fasta.py first: {idx_path}")

    with open(idx_path, "rb") as f:
        pairs = pickle.load(f)

    print(f"Running TargetScan on {len(pairs):,} pairs ({args.split} split) ...")

    results = []
    n_hits = 0

    for i, p in enumerate(pairs):
        if i % 200 == 0:
            print(f"  {i}/{len(pairs)} ...", end="\r", flush=True)

        hits = run_targetscan_pair(
            p["miRNA_ID"], p["mirna_seq"],
            p["isoform_ID"], p["circ_seq"],
        )

        seq_len   = p["length"]
        site_mask = hits_to_site_mask(hits, seq_len)
        pred_bind = 1 if len(hits) > 0 else 0
        n_sites   = len(hits)

        if hits:
            n_hits += 1

        results.append({
            "pair_id":    p["pair_id"],
            "isoform_ID": p["isoform_ID"],
            "miRNA_ID":   p["miRNA_ID"],
            "binding":    p["binding"],
            "sites":      p["sites"],
            "pred_sites": site_mask.tolist(),
            "pred_bind":  pred_bind,
            "bind_score": float(n_sites),  # number of sites as score proxy
            "n_hits":     n_sites,
            "length":     seq_len,
        })

    print(f"\n  Pairs with ≥1 hit: {n_hits:,}/{len(pairs):,}")

    out_path = split_outdir / "targetscan_preds.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
