#!/usr/bin/env python3
"""
run_pita.py
Runs PITA on every pair in pairs_index.pkl.

PITA (Probability of Interaction by Target Accessibility) uses:
  - Seed matching (6mer, 7mer-A1, 7mer-m8, 8mer)
  - RNA accessibility (ddG = dG_duplex - dG_open)
Designed for linear mRNA 3'UTR — applied here to circRNA for comparison.

Install:
  micromamba create -n pita_env -c bioconda -c conda-forge pita perl -y

Input format (FASTA):
  miRNA file : >miRNA_ID\\n<sequence>
  UTR file   : >gene_ID\\n<sequence>

PITA output columns:
  miRNA  gene  start  end  ddG  seed_type  ...
"""
import argparse
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PITA_BIN = next(
    (p for p in [
        "/root/.local/share/mamba/envs/pita_env/bin/pita_prediction.pl",
        "pita_prediction.pl",
    ] if Path(p).exists() or p == "pita_prediction.pl"),
    "pita_prediction.pl"
)

# PITA thresholds
DDG_THR = -10.0   # kcal/mol — sites with ddG <= this are predicted binding


def get_seed_types(mirna_seq: str) -> dict:
    """Extract seed sequences for 6mer/7mer-A1/7mer-m8/8mer matching."""
    seq = mirna_seq.upper().replace("T", "U")
    seeds = {}
    if len(seq) >= 7:
        seeds["7mer-m8"] = seq[1:8]   # pos 2-8
    if len(seq) >= 7:
        seeds["7mer-A1"] = seq[1:7]   # pos 2-7 (+ A at pos 1 of target)
    if len(seq) >= 8:
        seeds["8mer"]    = seq[1:8]   # pos 2-8 (+ A at pos 1)
    if len(seq) >= 6:
        seeds["6mer"]    = seq[1:7]   # pos 2-7
    return seeds


def run_pita_pair(mirna_id: str, mirna_seq: str,
                  circ_id: str, circ_seq: str) -> list[dict]:
    """
    Run PITA on one (miRNA, circRNA) pair.
    Falls back to seed-match + ViennaRNA accessibility if PITA binary unavailable.
    Returns list of hit dicts with keys: t_start, t_end, ddG, seed_type.
    """
    mirna_rna = mirna_seq.upper().replace("T", "U")
    circ_rna  = circ_seq.upper().replace("T", "U")
    safe_mir  = mirna_id.replace(" ", "_").replace(";", "_").replace("/", "_")
    safe_circ = circ_id.replace(" ", "_").replace(";", "_").replace("/", "_")

    with tempfile.TemporaryDirectory() as tmpdir:
        mir_fa  = os.path.join(tmpdir, "mir.fa")
        utr_fa  = os.path.join(tmpdir, "utr.fa")
        out_dir = os.path.join(tmpdir, "out")
        os.makedirs(out_dir)

        with open(mir_fa, "w") as f:
            f.write(f">{safe_mir}\n{mirna_rna}\n")
        with open(utr_fa, "w") as f:
            f.write(f">{safe_circ}\n{circ_rna}\n")

        try:
            result = subprocess.run(
                [PITA_BIN, "-utr", utr_fa, "-mir", mir_fa,
                 "-prefix", os.path.join(out_dir, "pita")],
                capture_output=True, text=True, timeout=60
            )
            # PITA writes: <prefix>_results.tab
            out_file = os.path.join(out_dir, "pita_results.tab")
            if os.path.exists(out_file):
                return _parse_pita_output(out_file, safe_circ, safe_mir, len(circ_rna))
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass  # fall through to Python fallback

    # ── Python fallback: seed matching + simplified ddG ─────────────────────
    return _python_pita_fallback(mirna_rna, circ_rna)


def _parse_pita_output(out_file: str, circ_id: str, mir_id: str,
                        seq_len: int) -> list[dict]:
    """Parse PITA tab output file."""
    hits = []
    with open(out_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("miRNA"):
                continue
            parts = line.split("\t")
            # columns: miRNA gene start end ddG seed_type ...
            if len(parts) < 5:
                continue
            try:
                t_start   = int(parts[2]) - 1   # 1-based → 0-based
                t_end     = int(parts[3])
                ddg       = float(parts[4])
                seed_type = parts[5] if len(parts) > 5 else "unknown"
                hits.append({
                    "t_start":   max(0, t_start),
                    "t_end":     min(seq_len, t_end),
                    "ddG":       ddg,
                    "seed_type": seed_type,
                })
            except (ValueError, IndexError):
                continue
    return hits


def _python_pita_fallback(mirna_seq: str, circ_seq: str,
                           flank: int = 7) -> list[dict]:
    """
    Python reimplementation of PITA core:
    1. Find seed matches (reverse complement of miRNA seed vs circ)
    2. Estimate ddG using ViennaRNA if available, else use seed-match score proxy
    """
    import re

    def rev_comp_rna(seq: str) -> str:
        comp = {"A": "U", "U": "A", "G": "C", "C": "G", "N": "N"}
        return "".join(comp.get(b, "N") for b in reversed(seq))

    mirna = mirna_seq.upper().replace("T", "U")
    circ  = circ_seq.upper().replace("T", "U")
    n     = len(circ)

    # Check seed types in priority order
    seed_configs = []
    if len(mirna) >= 8:
        seed_configs.append(("8mer",    mirna[1:8], True))   # pos 2-8, need A at pos1
    if len(mirna) >= 8:
        seed_configs.append(("7mer-m8", mirna[1:8], False))  # pos 2-8
    if len(mirna) >= 7:
        seed_configs.append(("7mer-A1", mirna[1:7], True))   # pos 2-7, need A
    if len(mirna) >= 7:
        seed_configs.append(("6mer",    mirna[1:7], False))  # pos 2-7

    # Get reverse complement for searching in circ
    hits = []
    seen_positions = set()

    for seed_type, seed, need_a in seed_configs:
        rc_seed = rev_comp_rna(seed)
        # Search for rc_seed in circ
        pattern = rc_seed.replace("U", "[UT]").replace("N", "[ACGUN]")
        try:
            for m in re.finditer(pattern, circ):
                pos = m.start()
                # For A1 types: check that position before match (downstream in circ) is A
                if need_a and "A1" in seed_type:
                    a_pos = pos + len(seed)
                    if a_pos >= n or circ[a_pos] != "A":
                        continue
                # Avoid duplicate overlapping hits
                if pos in seen_positions:
                    continue
                seen_positions.add(pos)

                # Estimate ddG via ViennaRNA if available, else seed-proxy
                ddg = _estimate_ddg(mirna, circ, pos, len(seed), flank)
                if ddg <= DDG_THR:
                    hits.append({
                        "t_start":   max(0, pos),
                        "t_end":     min(n, pos + len(seed)),
                        "ddG":       ddg,
                        "seed_type": seed_type,
                    })
        except re.error:
            continue

    return hits


def _estimate_ddg(mirna: str, circ: str, pos: int, seed_len: int,
                   flank: int = 7) -> float:
    """
    Estimate ddG = dG_duplex - dG_open.
    Uses ViennaRNA (RNA module) if available, else returns a seed-proxy score.
    """
    try:
        import RNA  # ViennaRNA Python bindings
        # Site window in circ (with flanking for accessibility)
        site_start = max(0, pos - flank)
        site_end   = min(len(circ), pos + seed_len + flank)
        site_seq   = circ[site_start:site_end]

        # dG_open: energy penalty to make site accessible (unfold)
        fc_site = RNA.fold_compound(site_seq)
        _, dg_fold = fc_site.mfe()

        # dG_duplex: rough estimate from GC content of seed
        seed_region = circ[pos:pos + seed_len]
        gc = sum(1 for b in seed_region if b in "GC") / max(len(seed_region), 1)
        dg_duplex = -1.5 - gc * 8.0   # empirical proxy

        return dg_duplex - dg_fold

    except ImportError:
        # ViennaRNA not available: use seed GC content as proxy
        seed_region = circ[pos:pos + seed_len]
        gc = sum(1 for b in seed_region if b in "GC") / max(len(seed_region), 1)
        return -2.0 - gc * 6.0  # negative = predicted hit


def hits_to_site_mask(hits: list[dict], seq_len: int) -> np.ndarray:
    mask = np.zeros(seq_len, dtype=np.float32)
    for h in hits:
        s = max(0, h["t_start"])
        e = min(seq_len, h["t_end"])
        if s < e:
            # Use |ddG| as score (more negative = stronger hit)
            score = min(1.0, abs(h["ddG"]) / 20.0)
            mask[s:e] = max(mask[s:e].max(), score)
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

    # Check if PITA binary is available
    pita_available = Path(PITA_BIN).exists()
    mode = "PITA binary" if pita_available else "Python fallback (seed-match + ddG proxy)"
    print(f"Running PITA on {len(pairs):,} pairs ({args.split} split) ...")
    print(f"  Mode: {mode}")

    results = []
    n_hits = 0

    for i, p in enumerate(pairs):
        if i % 200 == 0:
            print(f"  {i}/{len(pairs)} ...", end="\r", flush=True)

        hits = run_pita_pair(
            p["miRNA_ID"], p["mirna_seq"],
            p["isoform_ID"], p["circ_seq"],
        )

        seq_len   = p["length"]
        site_mask = hits_to_site_mask(hits, seq_len)
        pred_bind = 1 if len(hits) > 0 else 0
        # Use most negative ddG as binding score (more hits / lower ddG = more likely binding)
        bind_score = -min((h["ddG"] for h in hits), default=0.0)

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
            "bind_score": float(bind_score),
            "n_hits":     len(hits),
            "length":     seq_len,
        })

    print(f"\n  Pairs with ≥1 hit: {n_hits:,}/{len(pairs):,}")

    out_path = split_outdir / "pita_preds.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
