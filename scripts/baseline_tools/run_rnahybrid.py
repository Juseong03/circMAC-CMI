#!/usr/bin/env python3
"""
run_rnahybrid.py
Runs RNAhybrid on every pair and converts hit output to per-nucleotide
binary predictions.

RNAhybrid output (key fields):
  target: <target_id>
  miRNA : <mirna_id>
  mfe   : -XX.X kcal/mol
  position: NNN   (1-based start on target)
  length  : NNN

Strategy:
  - Run per-pair (RNAhybrid -t target.fa -q mirna.fa -s 3utr_human)
  - Use all hits above energy threshold
  - Mark hit intervals as 1 in site mask
"""
import argparse
import pickle
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ENERGY_THR = -20.0   # max free energy (kcal/mol)


def run_rnahybrid_pair(mirna_seq: str, circ_seq: str,
                       energy_thr: float = ENERGY_THR) -> list[dict]:
    """Run RNAhybrid on one pair. Returns list of hit dicts."""
    mirna_dna = mirna_seq.upper().replace("U", "T")
    circ_dna  = circ_seq.upper().replace("U", "T")

    with tempfile.NamedTemporaryFile("w", suffix=".fa", delete=False) as mf:
        mf.write(f">mirna\n{mirna_dna}\n")
        mirna_file = mf.name
    with tempfile.NamedTemporaryFile("w", suffix=".fa", delete=False) as tf:
        tf.write(f">target\n{circ_dna}\n")
        target_file = tf.name

    # -s 3utr_human: species-specific size correction
    # -f 2,7: seed required at positions 2-7 of miRNA
    # -e: energy threshold
    cmd = [
        "RNAhybrid",
        "-t", target_file,
        "-q", mirna_file,
        "-s", "3utr_human",
        "-e", str(energy_thr),
        "-m", "100000",  # allow long targets
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        output = result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        output = ""
    finally:
        Path(mirna_file).unlink(missing_ok=True)
        Path(target_file).unlink(missing_ok=True)

    hits = []
    # Parse output blocks
    mfe_re      = re.compile(r"mfe:\s*([-\d.]+)\s*kcal/mol")
    pos_re      = re.compile(r"position\s+(\d+)")
    mirna_len   = len(mirna_seq)

    current_mfe = None
    current_pos = None
    for line in output.splitlines():
        m = mfe_re.search(line)
        if m:
            current_mfe = float(m.group(1))
        p = pos_re.search(line)
        if p:
            current_pos = int(p.group(1)) - 1  # 1-based → 0-based
        if current_mfe is not None and current_pos is not None:
            hits.append({
                "energy":  current_mfe,
                "t_start": current_pos,
                "t_end":   current_pos + mirna_len,
            })
            current_mfe = None
            current_pos = None

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

    print(f"Running RNAhybrid on {len(pairs):,} pairs  "
          f"(energy <= {args.energy_thr}) ...")

    results = []
    n_hits = 0

    for i, p in enumerate(pairs):
        if i % 500 == 0:
            print(f"  {i}/{len(pairs)} ...", end="\r", flush=True)

        hits = run_rnahybrid_pair(
            p["mirna_seq"], p["circ_seq"],
            energy_thr=args.energy_thr,
        )

        seq_len   = p["length"]
        site_mask = hits_to_site_mask(hits, seq_len)
        pred_bind = 1 if len(hits) > 0 else 0
        best_energy = min((h["energy"] for h in hits), default=0.0)
        bind_score  = -best_energy if hits else 0.0  # lower energy = stronger

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

    out_path = outdir / "rnahybrid_preds.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
