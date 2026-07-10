#!/usr/bin/env python3
"""
run_miranda.py
Runs miRanda on every pair in pairs_index.pkl and converts hit output
to per-nucleotide binary predictions.

miRanda output format (per hit line):
  >miRNA  >target  score  energy  query_start  query_end  target_start  target_end  aln_len  identity  similarity  ...

Strategy:
  - For each (miRNA, circRNA) pair, mark the BEST hit's target interval as 1.
  - If no hit meets threshold, prediction is all zeros.
  - Pairs with at least one hit are predicted as binding=1.
"""
import argparse
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# miRanda thresholds (defaults from Miranda documentation)
SCORE_THR   = 155     # minimum alignment score
ENERGY_THR  = -20.0  # maximum free energy (kcal/mol)
STRICT_FLAG = ""     # "-strict" for seed-only matching


def run_miranda_pair(mirna_seq: str, circ_seq: str, pair_id: str,
                     score_thr: float = SCORE_THR,
                     energy_thr: float = ENERGY_THR) -> list[dict]:
    """Run miRanda on one pair. Returns list of hit dicts."""
    with tempfile.NamedTemporaryFile("w", suffix=".fa", delete=False) as mf:
        mf.write(f">{pair_id}_mirna\n{mirna_seq.upper().replace('U','T')}\n")
        mirna_file = mf.name
    with tempfile.NamedTemporaryFile("w", suffix=".fa", delete=False) as tf:
        tf.write(f">{pair_id}_circ\n{circ_seq.upper().replace('U','T')}\n")
        target_file = tf.name

    MIRANDA = next(
        (p for p in [
            "/opt/miranda_env/bin/miranda",
            "/usr/bin/miranda",
            "miranda",
        ] if Path(p).exists() or p == "miranda"),
        "miranda"
    )
    cmd = [
        MIRANDA, mirna_file, target_file,
        "-sc", str(score_thr),
        "-en", str(energy_thr),
        "-quiet",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        output = result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        output = ""
    finally:
        Path(mirna_file).unlink(missing_ok=True)
        Path(target_file).unlink(missing_ok=True)

    hits = []
    for line in output.splitlines():
        if not line.startswith(">>"):
            continue
        parts = line.split("\t")
        if len(parts) < 9:
            continue
        try:
            score      = float(parts[4])
            energy     = float(parts[5])
            t_start    = int(parts[7]) - 1   # 1-based → 0-based
            t_end      = int(parts[8])
            hits.append({
                "score":   score,
                "energy":  energy,
                "t_start": max(0, t_start),
                "t_end":   t_end,
            })
        except (ValueError, IndexError):
            continue
    return hits


def hits_to_site_mask(hits: list[dict], seq_len: int) -> np.ndarray:
    """Convert hit intervals to per-nucleotide binary mask."""
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
    parser.add_argument("--score_thr",  type=float, default=SCORE_THR)
    parser.add_argument("--energy_thr", type=float, default=ENERGY_THR)
    parser.add_argument("--n_workers",  type=int,   default=8)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    idx_path = outdir / "pairs_index.pkl"
    if not idx_path.exists():
        raise FileNotFoundError(f"Run prepare_fasta.py first: {idx_path}")

    with open(idx_path, "rb") as f:
        pairs = pickle.load(f)

    print(f"Running miRanda on {len(pairs):,} pairs  "
          f"(score>={args.score_thr}, energy<={args.energy_thr}) ...")

    results = []
    n_hits = 0

    for i, p in enumerate(pairs):
        if i % 500 == 0:
            print(f"  {i}/{len(pairs)} ...", end="\r", flush=True)

        hits = run_miranda_pair(
            p["mirna_seq"], p["circ_seq"], p["pair_id"],
            score_thr=args.score_thr, energy_thr=args.energy_thr,
        )

        seq_len   = p["length"]
        site_mask = hits_to_site_mask(hits, seq_len)
        pred_bind = 1 if len(hits) > 0 else 0
        # Use best-hit score as binding probability proxy (normalise 0-1)
        best_score = max((h["score"] for h in hits), default=0.0)

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
            "bind_score": best_score,
            "n_hits":     len(hits),
            "length":     seq_len,
        })

    print(f"\n  Pairs with ≥1 hit: {n_hits:,}/{len(pairs):,}")

    out_path = outdir / "miranda_preds.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
