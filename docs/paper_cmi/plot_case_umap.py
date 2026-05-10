#!/usr/bin/env python3
"""
Step 2: Load saved embeddings/coords and plot.

Reads from: figures_claude/emb_cache/
Writes to:
  figures_claude/encoders/position/   — CircMAC + encoder models, position color
  figures_claude/encoders/binding/    — CircMAC + encoder models, binding color
  figures_claude/pretrained/position/ — CircMAC + RNA LM models,  position color
  figures_claude/pretrained/binding/  — CircMAC + RNA LM models,  binding color

Each figure: rows=models, cols=[w/o miRNA | w/ miRNA]

Usage:
    python docs/paper_cmi/plot_case_umap.py
    python docs/paper_cmi/plot_case_umap.py --method umap
    python docs/paper_cmi/plot_case_umap.py --case cdyl2
"""

import argparse, sys, pickle, matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

ROOT  = Path(__file__).resolve().parents[2]
CACHE = ROOT / 'figures_claude' / 'emb_cache'
OUT   = ROOT / 'figures_claude'

BIND_COLOR = '#e6550d'
NEG_COLOR  = '#6baed6'
POS_CMAP   = 'viridis'

CASES = {
    'cdyl2': ('circCDYL2', 'hsa-miR-449a'),
    'mapk1': ('circMAPK1', 'hsa-miR-12119'),
    'app':   ('circAPP',   'hsa-miR-5001-3p'),
}

# CircMAC must appear in both groups
MODEL_GROUPS = {
    'encoders': [
        'CircMAC',
        'Transformer',
        'Mamba',
        'LSTM',
        'Hymba',
    ],
    'pretrained': [                  # main — trainable
        'CircMAC',
        'RNABert (train)',
        'RNAErnie (train)',
        'RNA-FM (train)',
        'RNA-MSM (train)',
    ],
    'pretrained_frozen': [           # supplementary — frozen
        'CircMAC',
        'RNABert (frozen)',
        'RNAErnie (frozen)',
        'RNA-FM (frozen)',
        'RNA-MSM (frozen)',
    ],
}


# ── Separation score ──────────────────────────────────────────────────────────
def sep_score(coords, lbls):
    p, n = coords[lbls==1], coords[lbls==0]
    if len(p) == 0 or len(n) == 0: return 0.0
    cp, cn = p.mean(0), n.mean(0)
    d = np.linalg.norm(cp - cn)
    return d / (np.sqrt(((p-cp)**2).sum(1)).mean() +
                np.sqrt(((n-cn)**2).sum(1)).mean() + 1e-8)


# ── Single plot: one group × one color_type ───────────────────────────────────
def plot_group(emb_data, coords_data, labels,
               case_name, mirna_id, method_label,
               color_type, out_path):
    """
    Horizontal layout:
      rows = [w/o miRNA, w/ miRNA]
      cols = models

    color_type: 'position' | 'binding'
    """
    has_wi = any(coords_data[l]['coords_wi'] is not None for l in labels
                 if l in coords_data)
    n_rows = 2 if has_wi else 1
    n_cols = len(labels)

    row_titles = (['w/o miRNA', 'w/ miRNA'] if has_wi else ['w/o miRNA'])

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 3.0, n_rows * 3.0 + 0.9),
                              squeeze=False)

    # Column headers = model names
    for ci, label in enumerate(labels):
        axes[0][ci].set_title(label, fontsize=9, fontweight='bold', pad=5)

    # Row labels = w/o miRNA / w/ miRNA
    for ri, rt in enumerate(row_titles):
        axes[ri][0].set_ylabel(rt, fontsize=9, fontweight='bold')

    for ci, label in enumerate(labels):
        if label not in coords_data:
            for ri in range(n_rows):
                axes[ri][ci].set_visible(False)
            continue

        cd      = coords_data[label]
        ed      = emb_data[label]
        lbls    = ed['lbls']
        pos_idx = ed['pos_idx']
        seq_len = ed['seq_len']
        pos_norm = pos_idx / max(seq_len - 1, 1)
        pos_c    = matplotlib.colormaps[POS_CMAP](pos_norm)

        for ri, use_wi in enumerate([False, True][:n_rows]):
            ax     = axes[ri][ci]
            coords = cd['coords_wi'] if use_wi else cd['coords_wo']

            if coords is None:
                ax.set_visible(False); continue

            if color_type == 'position':
                ax.scatter(coords[:, 0], coords[:, 1],
                           c=pos_c, s=10, alpha=0.80,
                           linewidths=0, rasterized=True)
            else:  # binding
                s = sep_score(coords, lbls)
                ax.scatter(coords[lbls==0, 0], coords[lbls==0, 1],
                           c=NEG_COLOR, s=10, alpha=0.35,
                           linewidths=0, rasterized=True)
                ax.scatter(coords[lbls==1, 0], coords[lbls==1, 1],
                           c=BIND_COLOR, s=22, alpha=0.85,
                           linewidths=0, rasterized=True)
                ax.set_xlabel(f'Sep={s:.3f}', fontsize=7)

            ax.set_xticks([]); ax.set_yticks([])

    short_mirna = mirna_id.replace('hsa-', '')
    fig.suptitle(f'{case_name}  ×  {short_mirna}   |   {method_label}',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()

    # ── Legend / colorbar (after tight_layout) ────────────────────────────────
    if color_type == 'position':
        fig.subplots_adjust(bottom=0.10)
        cax = fig.add_axes([0.15, 0.02, 0.70, 0.022])
        sm  = plt.cm.ScalarMappable(cmap=POS_CMAP, norm=plt.Normalize(0, 1))
        sm.set_array([])
        cb  = fig.colorbar(sm, cax=cax, orientation='horizontal')
        cb.set_label("Sequence position (5′ → 3′)", fontsize=8)
        cb.set_ticks([0, 0.5, 1])
        cb.set_ticklabels(["5′", "mid", "3′"])
    else:
        fig.subplots_adjust(bottom=0.08)
        handles = [
            Line2D([0],[0], marker='o', color='w', markerfacecolor=BIND_COLOR,
                   markersize=7, label='Binding'),
            Line2D([0],[0], marker='o', color='w', markerfacecolor=NEG_COLOR,
                   markersize=7, label='Non-binding'),
        ]
        fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=9,
                   framealpha=0.9, bbox_to_anchor=(0.5, 0.01))

    for ext in ['pdf', 'png']:
        p = f'{out_path}.{ext}'
        fig.savefig(p, dpi=150, bbox_inches='tight')
        print(f"    Saved → {p}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case',   default='all', help='cdyl2|mapk1|app|all')
    parser.add_argument('--method', default='both', choices=['umap','tsne','both'])
    args = parser.parse_args()

    # Create output folders
    for group in MODEL_GROUPS:
        for ctype in ['position', 'binding']:
            (OUT / group / ctype).mkdir(parents=True, exist_ok=True)

    cases_to_run = CASES if args.case == 'all' else {args.case: CASES[args.case]}
    methods = ['umap','tsne'] if args.method == 'both' else [args.method]

    for case_key, (case_name, mirna_id) in cases_to_run.items():
        emb_path = CACHE / f'case_{case_key}_embeddings.pkl'
        if not emb_path.exists():
            print(f"[SKIP] {case_key}: run extract_case_embeddings.py first")
            continue
        with open(emb_path, 'rb') as f:
            emb_data = pickle.load(f)

        for method in methods:
            coord_path = CACHE / f'case_{case_key}_{method}_coords.pkl'
            if not coord_path.exists():
                print(f"[SKIP] {case_key}/{method}: coords not found")
                continue
            with open(coord_path, 'rb') as f:
                coords_data = pickle.load(f)

            for group, model_list in MODEL_GROUPS.items():
                # Only keep models that are in the cache
                labels = [l for l in model_list if l in coords_data]
                if not labels:
                    continue

                for ctype in ['position', 'binding']:
                    out_path = str(
                        OUT / group / ctype / f'case_{case_key}_{method}'
                    )
                    print(f"  [{group}/{ctype}] {case_name} {method.upper()}...")
                    plot_group(emb_data, coords_data, labels,
                               case_name, mirna_id, method.upper(),
                               ctype, out_path)

    print("\nAll done!")


if __name__ == '__main__':
    main()
