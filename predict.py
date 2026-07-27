#!/usr/bin/env python3
"""
predict.py — CircMAC Inference Tool
====================================
Predict circRNA–miRNA binding sites for one or more pairs.

Input
-----
Sequences can be provided directly on the command line or via a CSV/FASTA file.

Output columns
--------------
  circRNA_id            user-supplied ID (or auto-generated)
  miRNA_id              user-supplied ID (or auto-generated)
  rank                  site rank by score (1 = strongest)
  site_start            0-based start position on circRNA (BSJ = position 0)
  site_end              0-based exclusive end position
  site_length           site_end - site_start
  site_score            mean predicted probability within the site
  peak_position         position of highest probability within the site
  peak_probability      probability at peak_position
  BSJ_relation          'BSJ-adjacent' | 'distal'
  distance_to_BSJ       nt from nearest site edge to BSJ (position 0)
  circRNA_site_sequence subsequence of circRNA at [site_start:site_end]
  miRNA_sequence        full miRNA guide-strand sequence

Examples
--------
  # Single pair
  python predict.py \
      --circRNA GUGUGCACAUU...  --circRNA_id circFANCA \
      --miRNA   GUGAGGAGG...    --miRNA_id   hsa-miR-6858-5p \
      --model_path saved_models/circmac/max_circmac_pairing_s1/train/model.pth

  # Batch CSV (columns: circRNA_id, miRNA_id, circRNA, miRNA)
  python predict.py \
      --input pairs.csv \
      --model_path saved_models/circmac/max_circmac_pairing_s1/train/model.pth \
      --out results.csv

  # FASTA inputs
  python predict.py \
      --circRNA_fasta circrna.fa \
      --miRNA_fasta   mirna.fa \
      --model_path ...
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data import KmerTokenizer
from utils import get_device
from utils_config import get_model_config
from trainer import Trainer

# ── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_MODEL_NAME  = "circmac"
DEFAULT_CKPT        = "saved_models/circmac/max_circmac_pairing_s1/train/model.pth"
SITE_PROB_THRESHOLD = 0.5     # nucleotide classified as binding if prob >= threshold
BSJ_WINDOW          = 40      # nt; site is 'BSJ-adjacent' if any edge within this window
MAX_LEN             = 1022
D_MODEL             = 128
N_LAYER             = 6


# ── Sequence helpers ──────────────────────────────────────────────────────────
def read_fasta(path: str) -> List[Tuple[str, str]]:
    """Return [(id, sequence), ...] from a FASTA file."""
    pairs = []
    seq_id, seq_lines = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq_id is not None:
                    pairs.append((seq_id, "".join(seq_lines).upper().replace("T", "U")))
                seq_id = line[1:].split()[0]
                seq_lines = []
            else:
                seq_lines.append(line)
    if seq_id is not None:
        pairs.append((seq_id, "".join(seq_lines).upper().replace("T", "U")))
    return pairs


def clean_seq(seq: str) -> str:
    return seq.upper().strip().replace("T", "U")


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model(ckpt_path: str, model_name: str, device: torch.device) -> Trainer:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    # Infer vocab size from checkpoint
    vocab_size = 11  # default k=1 tokenizer
    for k, v in state.items():
        if "embedding" in k and "weight" in k and hasattr(v, "ndim") and v.ndim == 2 and v.shape[-1] == D_MODEL:
            vocab_size = int(v.shape[0])
            break

    cfg = get_model_config(model_name=model_name, d_model=D_MODEL, n_layer=N_LAYER,
                           vocab_size=vocab_size)
    tr = Trainer(seed=1, device=device, experiment_name="predict", verbose=False)
    tr.task = "sites"
    tr.rc   = False
    tr.use_unified_head = False
    tr.interaction = "cross_attention"
    tr.define_model(cfg, model_name=model_name, pretrain=False,
                    is_cross_attention=True, interaction="cross_attention",
                    site_head_type="conv1d")
    tr.set_pretrained_target(target="mirna", rna_model="rnabert")
    tr.load_model_from_path(ckpt_path, verbose=True)
    tr.model.eval()
    return tr


# ── Single-pair inference ─────────────────────────────────────────────────────
def infer_pair(trainer: Trainer, circ_seq: str, mirna_seq: str,
               max_len: int = MAX_LEN) -> Optional[np.ndarray]:
    """Return per-nucleotide predicted probabilities (length = min(len(circ), max_len))."""
    from data import CircRNABindingSitesDataset
    import pandas as pd

    circ_seq  = clean_seq(circ_seq)[:max_len]
    mirna_seq = clean_seq(mirna_seq)

    # Build a one-row dataframe matching dataset expectations
    dummy_sites = [0] * len(circ_seq)   # placeholder labels
    df = pd.DataFrame([{
        "isoform_ID": "query",
        "miRNA_ID":   "query_mirna",
        "circRNA":    circ_seq,
        "miRNA":      mirna_seq,
        "sites":      dummy_sites,
        "binding":    1,
        "length":     len(circ_seq),
        "n_binding_site": 0,
        "ratio_binding_site": 0.0,
    }])

    ds = CircRNABindingSitesDataset(df, max_len=max_len + 2, target_type="mirna", k=1, k_target=1)

    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    with torch.no_grad():
        data = next(iter(loader))
        target, target_mask = trainer.forward_target(data)
        emb, _ = trainer.forward(data)
        emb, _ = trainer.forward_cross_attention(emb, target, target_mask)
        logits = trainer.forward_task(emb, target, task="sites")

        # logits: (1, L, 2) or (1, L)
        if logits.ndim == 3 and logits.shape[-1] == 2:
            probs = torch.softmax(logits.float(), dim=-1)[0, :, 1]
        else:
            probs = torch.sigmoid(logits.float())[0].squeeze(-1)

    probs_np = probs.cpu().numpy()
    # Strip CLS/SEP tokens if present (length mismatch)
    target_len = len(circ_seq)
    if len(probs_np) > target_len:
        probs_np = probs_np[:target_len]
    return probs_np


# ── Site extraction ───────────────────────────────────────────────────────────
def extract_sites(
    probs: np.ndarray,
    circ_seq: str,
    mirna_seq: str,
    circ_id: str,
    mirna_id: str,
    threshold: float = SITE_PROB_THRESHOLD,
    bsj_window: int  = BSJ_WINDOW,
    min_site_len: int = 1,
) -> pd.DataFrame:
    """
    Convert per-nucleotide probabilities to a ranked site table.
    Returns a DataFrame with all output columns.
    """
    L = len(probs)
    binary = (probs >= threshold).astype(int)

    # Find contiguous runs of 1s
    sites = []
    i = 0
    while i < L:
        if binary[i] == 1:
            j = i
            while j < L and binary[j] == 1:
                j += 1
            sites.append((i, j))   # [start, end)
            i = j
        else:
            i += 1

    if not sites:
        return pd.DataFrame()

    rows = []
    for start, end in sites:
        site_len  = end - start
        if site_len < min_site_len:
            continue
        site_probs = probs[start:end]
        score      = float(site_probs.mean())
        peak_rel   = int(np.argmax(site_probs))
        peak_pos   = start + peak_rel
        peak_prob  = float(site_probs[peak_rel])

        # BSJ relation: circRNA starts at BSJ (position 0)
        # Also check wrap-around distance (circular)
        dist_start_fwd = start
        dist_end_fwd   = end - 1
        dist_start_rev = L - start        # wrap around
        dist_to_bsj    = min(dist_start_fwd, dist_end_fwd, dist_start_rev)
        bsj_relation   = "BSJ-adjacent" if dist_to_bsj <= bsj_window else "distal"

        site_seq = circ_seq[start:end].upper().replace("T", "U")

        rows.append({
            "circRNA_id":            circ_id,
            "miRNA_id":              mirna_id,
            "site_start":            start,
            "site_end":              end,
            "site_length":           site_len,
            "site_score":            round(score, 5),
            "peak_position":         peak_pos,
            "peak_probability":      round(peak_prob, 5),
            "BSJ_relation":          bsj_relation,
            "distance_to_BSJ":       dist_to_bsj,
            "circRNA_site_sequence": site_seq,
            "miRNA_sequence":        mirna_seq.upper().replace("T", "U"),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values("site_score", ascending=False).reset_index(drop=True)
    df.insert(2, "rank", range(1, len(df) + 1))
    return df


# ── Batch processing ──────────────────────────────────────────────────────────
def run_pairs(
    pairs: List[dict],          # [{"circRNA_id", "miRNA_id", "circRNA", "miRNA"}, ...]
    trainer: Trainer,
    threshold: float,
    bsj_window: int,
    max_len: int,
    min_site_len: int,
) -> pd.DataFrame:
    all_rows = []
    for i, p in enumerate(pairs):
        circ_id  = p.get("circRNA_id",  f"circRNA_{i+1}")
        mirna_id = p.get("miRNA_id",    f"miRNA_{i+1}")
        circ_seq = clean_seq(p["circRNA"])
        mirna_seq= clean_seq(p["miRNA"])

        print(f"  [{i+1}/{len(pairs)}] {circ_id} × {mirna_id}  "
              f"(circRNA {len(circ_seq)} nt, miRNA {len(mirna_seq)} nt)", flush=True)

        if len(circ_seq) == 0 or len(mirna_seq) == 0:
            print("    [SKIP] empty sequence")
            continue

        probs = infer_pair(trainer, circ_seq, mirna_seq, max_len=max_len)
        if probs is None:
            print("    [SKIP] inference failed")
            continue

        df_sites = extract_sites(
            probs, circ_seq, mirna_seq, circ_id, mirna_id,
            threshold=threshold, bsj_window=bsj_window, min_site_len=min_site_len,
        )
        if df_sites.empty:
            print(f"    → No sites found above threshold {threshold}")
        else:
            print(f"    → {len(df_sites)} site(s) found")
            all_rows.append(df_sites)

    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="CircMAC: predict circRNA–miRNA binding sites",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model
    parser.add_argument("--model_path",  default=DEFAULT_CKPT,
                        help=f"Path to model.pth checkpoint (default: {DEFAULT_CKPT})")
    parser.add_argument("--model_name",  default=DEFAULT_MODEL_NAME,
                        choices=["circmac","lstm","transformer","mamba","hymba",
                                 "rnabert","rnaernie","rnamsm","rnafm"],
                        help="Model architecture (default: circmac)")
    parser.add_argument("--device",      type=int, default=0,
                        help="GPU index, or -1 for CPU (default: 0)")

    # Single-pair input
    parser.add_argument("--circRNA",     type=str, default=None,
                        help="circRNA sequence (RNA, 5'→3' from BSJ)")
    parser.add_argument("--circRNA_id",  type=str, default=None)
    parser.add_argument("--miRNA",       type=str, default=None,
                        help="miRNA guide-strand sequence")
    parser.add_argument("--miRNA_id",    type=str, default=None)

    # FASTA inputs
    parser.add_argument("--circRNA_fasta", type=str, default=None,
                        help="FASTA file of circRNA sequences")
    parser.add_argument("--miRNA_fasta",   type=str, default=None,
                        help="FASTA file of miRNA sequences (paired 1:1 with circRNA_fasta)")

    # Batch CSV input
    parser.add_argument("--input", type=str, default=None,
                        help="CSV with columns: circRNA_id, miRNA_id, circRNA, miRNA")

    # Output
    parser.add_argument("--out",     type=str, default=None,
                        help="Output CSV path (default: print to stdout)")
    parser.add_argument("--out_format", choices=["csv","tsv","json"], default="csv")

    # Prediction settings
    parser.add_argument("--threshold",   type=float, default=SITE_PROB_THRESHOLD,
                        help=f"Per-nucleotide probability threshold (default: {SITE_PROB_THRESHOLD})")
    parser.add_argument("--bsj_window",  type=int,   default=BSJ_WINDOW,
                        help=f"Nt from BSJ to call a site 'BSJ-adjacent' (default: {BSJ_WINDOW})")
    parser.add_argument("--max_len",     type=int,   default=MAX_LEN,
                        help=f"Max circRNA length; longer sequences are truncated (default: {MAX_LEN})")
    parser.add_argument("--min_site_len",type=int,   default=1,
                        help="Minimum site length in nt to report (default: 1)")
    parser.add_argument("--top_n",       type=int,   default=None,
                        help="Report only top-N sites per pair (default: all)")

    args = parser.parse_args()

    # ── Collect input pairs ──────────────────────────────────────────────────
    pairs = []

    if args.input:
        df_in = pd.read_csv(args.input)
        for _, row in df_in.iterrows():
            pairs.append({
                "circRNA_id": str(row.get("circRNA_id", f"circ_{len(pairs)+1}")),
                "miRNA_id":   str(row.get("miRNA_id",   f"mirna_{len(pairs)+1}")),
                "circRNA":    str(row["circRNA"]),
                "miRNA":      str(row["miRNA"]),
            })

    elif args.circRNA_fasta and args.miRNA_fasta:
        circs  = read_fasta(args.circRNA_fasta)
        mirnas = read_fasta(args.miRNA_fasta)
        if len(circs) != len(mirnas):
            print(f"[ERROR] FASTA files have different numbers of sequences "
                  f"({len(circs)} vs {len(mirnas)})", file=sys.stderr)
            sys.exit(1)
        for (cid, cseq), (mid, mseq) in zip(circs, mirnas):
            pairs.append({"circRNA_id": cid, "miRNA_id": mid,
                          "circRNA": cseq, "miRNA": mseq})

    elif args.circRNA and args.miRNA:
        pairs.append({
            "circRNA_id": args.circRNA_id  or "circRNA_1",
            "miRNA_id":   args.miRNA_id    or "miRNA_1",
            "circRNA":    args.circRNA,
            "miRNA":      args.miRNA,
        })

    else:
        parser.error("Provide one of: --circRNA+--miRNA, --input CSV, "
                     "or --circRNA_fasta+--miRNA_fasta")

    # ── Load model ───────────────────────────────────────────────────────────
    device = get_device(args.device)
    print(f"Loading model: {args.model_path}")
    trainer = load_model(args.model_path, args.model_name, device)

    # ── Run inference ─────────────────────────────────────────────────────────
    print(f"\nPredicting binding sites for {len(pairs)} pair(s) "
          f"(threshold={args.threshold}, BSJ window={args.bsj_window} nt) ...\n")

    df_out = run_pairs(
        pairs, trainer,
        threshold=args.threshold,
        bsj_window=args.bsj_window,
        max_len=args.max_len,
        min_site_len=args.min_site_len,
    )

    if df_out.empty:
        print("\nNo binding sites predicted.")
        sys.exit(0)

    if args.top_n:
        df_out = (df_out.groupby(["circRNA_id","miRNA_id"], group_keys=False)
                  .apply(lambda g: g.head(args.top_n))
                  .reset_index(drop=True))

    # ── Output ────────────────────────────────────────────────────────────────
    col_order = [
        "circRNA_id", "miRNA_id", "rank",
        "site_start", "site_end", "site_length",
        "site_score", "peak_position", "peak_probability",
        "BSJ_relation", "distance_to_BSJ",
        "circRNA_site_sequence", "miRNA_sequence",
    ]
    df_out = df_out[[c for c in col_order if c in df_out.columns]]

    if args.out:
        out_path = Path(args.out)
        if args.out_format == "tsv":
            df_out.to_csv(out_path, sep="\t", index=False)
        elif args.out_format == "json":
            df_out.to_json(out_path, orient="records", indent=2)
        else:
            df_out.to_csv(out_path, index=False)
        print(f"\nSaved {len(df_out)} site(s) → {out_path}")
    else:
        print()
        if args.out_format == "tsv":
            print(df_out.to_csv(sep="\t", index=False))
        elif args.out_format == "json":
            print(df_out.to_json(orient="records", indent=2))
        else:
            print(df_out.to_csv(index=False))


if __name__ == "__main__":
    main()
