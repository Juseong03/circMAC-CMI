#!/usr/bin/env python3
"""
run_intarna.py
Runs IntaRNA on every pair and converts output to per-nucleotide predictions.

IntaRNA is a modern RNA-RNA interaction predictor.
Output CSV columns: id1, id2, start1, end1, start2, end2, E (energy), ...

Strategy:
  - Run with --outMode C (CSV output)
  - Use all reported interactions
  - Map target interaction interval to site mask
"""
import argparse
import csv
import io
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ENERGY_THR = -5.0   # IntaRNA default, less strict than miRanda/RNAhybrid


def run_intarna_pair(mirna_seq: str, circ_seq: str,
                     energy_thr: float = ENERGY_THR) -> list[dict]:
    """Run IntaRNA on one pair. Returns list of hit dicts."""
    mirna_rna = mirna_seq.upper().replace("T", "U")
    circ_rna  = circ_seq.upper().replace("T", "U")

    cmd = [
        "IntaRNA",
        "-q", mirna_rna,      # query (miRNA)
        "-t", circ_rna,       # target (circRNA)
        "--outMode", "C",     # CSV output
        "--outCsvCols", "start1,end1,start2,end2,E",
        "--pred", "S",        # single best interaction per query-target pair
        "--tAccW", "0",       # no accessibility window restriction
        "--tAccL", "0",
        "--qAccW", "0",
        "--qAccL", "0",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        output = result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        output = ""

    hits = []
    if not output.strip():
        return hits

    try:
        reader = csv.DictReader(io.StringIO(output), delimiter=";")
        for row in reader:
            try:
                energy  = float(row.get("E", "0") or "0")
                t_start = int(row.get("start2", "1") or "1") - 1  # 1-based → 0-based
                t_end   = int(row.get("end2", "1") or "1")
                if energy <= energy_thr:
                    hits.append({
                        "energy":  energy,
                        "t_start": max(0, t_start),
                        "t_end":   t_end,
                    })
            except (ValueError, KeyError):
                continue
    except Exception:
        pass

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
    parser.add_argument("--outdir",     default="results/baseline_tools")
    parser.add_argument("--energy_thr", type=float, default=ENERGY_THR)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    with open(outdir / "pairs_index.pkl", "rb") as f:
        pairs = pickle.load(f)

    print(f"Running IntaRNA on {len(pairs):,} pairs  "
          f"(energy <= {args.energy_thr}) ...")

    results = []
    n_hits = 0

    for i, p in enumerate(pairs):
        if i % 500 == 0:
            print(f"  {i}/{len(pairs)} ...", end="\r", flush=True)

        hits = run_intarna_pair(
            p["mirna_seq"], p["circ_seq"],
            energy_thr=args.energy_thr,
        )

        seq_len   = p["length"]
        site_mask = hits_to_site_mask(hits, seq_len)
        pred_bind = 1 if len(hits) > 0 else 0
        best_energy = min((h["energy"] for h in hits), default=0.0)
        bind_score  = -best_energy if hits else 0.0

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
            "bind_score": bind_score,
            "n_hits":     len(hits),
            "length":     seq_len,
        })

    print(f"\n  Pairs with ≥1 hit: {n_hits:,}/{len(pairs):,}")

    out_path = outdir / "intarna_preds.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
