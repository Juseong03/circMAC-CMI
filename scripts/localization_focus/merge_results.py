#!/usr/bin/env python3
"""
merge_results.py
================
Merge per-model localization focus CSV files into a single combined CSV.

Usage:
    python scripts/localization_focus/merge_results.py
    python scripts/localization_focus/merge_results.py --splits pair iso bsj
"""
import argparse
from pathlib import Path
import pandas as pd

ROOT    = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "figures_paper" / "fig_localization_focus"


def merge_split(split: str, out_dir: Path) -> pd.DataFrame | None:
    pattern = f"localization_focus_{split}_summary_*.csv"
    files   = sorted(out_dir.glob(pattern))

    if not files:
        print(f"  [{split}] no per-model files found (pattern: {pattern})")
        return None

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        print(f"  [{split}] loaded {f.name}  ({len(df)} rows, models: {df['model'].unique().tolist()})")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values(["split", "model", "radius"]).reset_index(drop=True)

    out_path = out_dir / f"localization_focus_{split}_summary.csv"
    combined.to_csv(out_path, index=False)
    print(f"  [{split}] merged {len(files)} files -> {out_path}  ({len(combined)} rows total)")
    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["pair", "iso", "bsj"])
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_dfs = []
    for split in args.splits:
        df = merge_split(split, OUT_DIR)
        if df is not None:
            all_dfs.append(df)

    if len(all_dfs) > 1:
        combined_all = pd.concat(all_dfs, ignore_index=True)
        all_path = OUT_DIR / "localization_focus_all_summary.csv"
        combined_all.to_csv(all_path, index=False)
        print(f"\nMerged all splits -> {all_path}  ({len(combined_all)} rows)")


if __name__ == "__main__":
    main()
