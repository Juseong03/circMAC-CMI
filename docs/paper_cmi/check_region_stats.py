#!/usr/bin/env python3
"""
GT Binding Site Region 분포 분석

사용법:
    python docs/paper_cmi/check_region_stats.py
    python docs/paper_cmi/check_region_stats.py --data data/df_test_final.pkl
"""

import argparse
import pickle
import numpy as np
from pathlib import Path
from collections import Counter


def get_regions(arr):
    """Binary array → list of (start, end, length) tuples."""
    arr = np.array(arr, dtype=int)
    regions = []
    in_r, start = False, 0
    for i, v in enumerate(arr):
        if v and not in_r:
            start, in_r = i, True
        elif not v and in_r:
            regions.append((start, i, i - start))
            in_r = False
    if in_r:
        regions.append((start, len(arr), len(arr) - start))
    return regions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/df_test_final.pkl',
                        help='Path to df_test_final.pkl')
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        # fallback: try relative to script location
        data_path = Path(__file__).parent.parent.parent / 'data' / 'df_test_final.pkl'

    print(f"Loading: {data_path}")
    df = pickle.load(open(data_path, 'rb'))

    print(f"Shape   : {df.shape}")
    print(f"Columns : {df.columns.tolist()}")

    # sites 컬럼 찾기
    for col in ['sites', 'binding_sites', 'site_labels', 'label']:
        if col in df.columns:
            sites_col = col
            break
    else:
        print("\n[ERROR] sites 컬럼을 찾을 수 없습니다.")
        print("사용 가능한 컬럼:", df.columns.tolist())
        return

    print(f"Sites col: '{sites_col}'")

    # binding==1인 row만
    if 'binding' in df.columns:
        df_bind = df[df['binding'] == 1].copy()
    else:
        df_bind = df.copy()

    print(f"Binding samples: {len(df_bind)}")

    # Region 분석
    all_lengths = []
    n_sites_per_row = []
    gap_lengths = []    # 인접 region 간 gap

    for _, row in df_bind.iterrows():
        regs = get_regions(row[sites_col])
        lengths = [r[2] for r in regs]
        all_lengths.extend(lengths)
        n_sites_per_row.append(len(regs))

        # gap between consecutive regions
        for i in range(len(regs) - 1):
            gap = regs[i+1][0] - regs[i][1]
            gap_lengths.append(gap)

    all_lengths = np.array(all_lengths)
    n_sites = np.array(n_sites_per_row)
    gap_lengths = np.array(gap_lengths) if gap_lengths else np.array([0])

    print("\n" + "="*55)
    print(f"  GT Binding Site Region 길이 분포  (n={len(all_lengths)})")
    print("="*55)
    print(f"  min    : {all_lengths.min()}")
    print(f"  max    : {all_lengths.max()}")
    print(f"  mean   : {all_lengths.mean():.2f}")
    print(f"  median : {np.median(all_lengths):.1f}")
    print(f"  std    : {all_lengths.std():.2f}")
    print()
    print(f"  == 1 nt  : {(all_lengths==1).sum():5d}  ({(all_lengths==1).mean()*100:.1f}%)")
    print(f"  <= 3 nt  : {(all_lengths<=3).sum():5d}  ({(all_lengths<=3).mean()*100:.1f}%)")
    print(f"  <= 5 nt  : {(all_lengths<=5).sum():5d}  ({(all_lengths<=5).mean()*100:.1f}%)")
    print(f"  <= 8 nt  : {(all_lengths<=8).sum():5d}  ({(all_lengths<=8).mean()*100:.1f}%)")
    print(f"  <= 10 nt : {(all_lengths<=10).sum():5d}  ({(all_lengths<=10).mean()*100:.1f}%)")
    print(f"  <= 22 nt : {(all_lengths<=22).sum():5d}  ({(all_lengths<=22).mean()*100:.1f}%)")
    print(f"  > 22 nt  : {(all_lengths>22).sum():5d}  ({(all_lengths>22).mean()*100:.1f}%)")

    # 길이 분포 히스토그램 (text)
    print()
    print("  길이 분포 (1~30+):")
    cnt = Counter(all_lengths.tolist())
    for l in range(1, 31):
        bar = '#' * min(cnt.get(l, 0), 50)
        print(f"  {l:3d}nt: {cnt.get(l,0):5d} {bar}")
    over30 = sum(v for k, v in cnt.items() if k > 30)
    if over30:
        print(f"  >30nt: {over30:5d}")

    print("\n" + "="*55)
    print(f"  샘플당 binding site region 개수")
    print("="*55)
    print(f"  min    : {n_sites.min()}")
    print(f"  max    : {n_sites.max()}")
    print(f"  mean   : {n_sites.mean():.2f}")
    print(f"  median : {np.median(n_sites):.1f}")
    print()
    for k in sorted(Counter(n_sites.tolist()).keys())[:15]:
        print(f"  {k:2d} sites: {Counter(n_sites.tolist())[k]:5d} samples")

    print("\n" + "="*55)
    print(f"  인접 region 간 gap 분포  (n={len(gap_lengths)})")
    print("="*55)
    print(f"  min    : {gap_lengths.min()}")
    print(f"  max    : {gap_lengths.max()}")
    print(f"  mean   : {gap_lengths.mean():.2f}")
    print(f"  median : {np.median(gap_lengths):.1f}")
    print()
    print(f"  <= 2 nt  : {(gap_lengths<=2).sum():5d}  ({(gap_lengths<=2).mean()*100:.1f}%)")
    print(f"  <= 5 nt  : {(gap_lengths<=5).sum():5d}  ({(gap_lengths<=5).mean()*100:.1f}%)")
    print(f"  <= 10 nt : {(gap_lengths<=10).sum():5d}  ({(gap_lengths<=10).mean()*100:.1f}%)")
    print(f"  > 10 nt  : {(gap_lengths>10).sum():5d}  ({(gap_lengths>10).mean()*100:.1f}%)")
    print()
    print("  → gap이 작으면 merge 고려 가능")
    print("="*55)


if __name__ == '__main__':
    main()
