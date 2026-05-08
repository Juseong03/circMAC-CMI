#!/usr/bin/env python3
"""
Position-level embedding UMAP visualization.

각 nucleotide position의 backbone embedding을 UMAP으로 시각화하여
모델별 binding/non-binding site 분리도를 비교합니다.

Requirements (서버에서 최초 실행 시):
    pip install umap-learn

Usage:
    # 기본: NoPT vs MLM 비교 (test set)
    python docs/paper_cmi/plot_umap_embeddings.py

    # EXP2 전체 비교
    python docs/paper_cmi/plot_umap_embeddings.py --models nopt mlm ntp ssp cpcl bsj --layout grid

    # 특정 모델 + t-SNE
    python docs/paper_cmi/plot_umap_embeddings.py --models nopt mlm --method tsne

    # 다른 GPU / sample 수
    python docs/paper_cmi/plot_umap_embeddings.py --device 0 --n_samples 30000

    # 저장 경로 지정
    python docs/paper_cmi/plot_umap_embeddings.py --out docs/paper_cmi/umap_embeddings.pdf
"""

import argparse
import sys
import os
import pickle
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from torch.utils.data import DataLoader

# ── project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data import CircRNABindingSitesDataset
from models.model import ModelWrapper
from utils_config import get_model_config
from utils import load_model

# ── model registry ────────────────────────────────────────────────────────────
# key → (exp_name, seed, model_name, display_label)
MODEL_REGISTRY = {
    'nopt':  ('v2_pt_nopt_s1',  1, 'circmac', 'NoPT'),
    'mlm':   ('v2_pt_mlm_s1',   1, 'circmac', 'MLM'),
    'ntp':   ('v2_pt_ntp_s1',   1, 'circmac', 'NTP'),
    'ssp':   ('v2_pt_ssp_s1',   1, 'circmac', 'SSP'),
    'cpcl':  ('v2_pt_cpcl_s1',  1, 'circmac', 'CPCL'),
    'bsj':   ('v2_pt_bsj_s1',   1, 'circmac', 'BSJ'),
    'all':   ('v2_pt_all_s1',   1, 'circmac', 'ALL'),
    'mlm_ssp':  ('v2_pt_mlm_ssp_s1',  1, 'circmac', 'MLM+SSP'),
    'mlm_cpcl': ('v2_pt_mlm_cpcl_s1', 1, 'circmac', 'MLM+CPCL'),
    'mlm_ntp':  ('v2_pt_mlm_ntp_s1',  1, 'circmac', 'MLM+NTP'),
    'mlm_bsj':  ('v2_pt_mlm_bsj_s1',  1, 'circmac', 'MLM+BSJ'),
}

COLORS = {
    'non-binding': '#6baed6',   # blue
    'binding':     '#e6550d',   # orange-red
}


# ── data loading ──────────────────────────────────────────────────────────────
def build_test_loader(batch_size=32, num_workers=4, max_len=1022):
    test_df = pickle.load(open(ROOT / 'data/df_test_final.pkl', 'rb'))
    ds = CircRNABindingSitesDataset(
        df=test_df,
        max_len=max_len,
        k=1,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=False)


# ── model setup ───────────────────────────────────────────────────────────────
def build_model(exp_name, seed, model_name, device, d_model=128, n_layer=6):
    config = get_model_config(model_name, d_model=d_model, n_layer=n_layer, verbose=False)
    model = ModelWrapper(config=config, name=model_name, device=device).to(device)
    model = load_model(
        model=model,
        dir_save=str(ROOT / 'saved_models'),
        model_name=model_name,
        experiment_name=exp_name,
        seed=seed,
        device=device,
        pretrain=False,
        verbose=True,
    )
    model.eval()
    return model


# ── embedding extraction ───────────────────────────────────────────────────────
@torch.no_grad()
def extract_embeddings(model, loader, device, n_samples=None, skip_cls=True):
    """
    Returns:
        embeddings: np.ndarray [N, d_model]
        labels:     np.ndarray [N]  (1=binding, 0=non-binding)
    """
    all_emb = []
    all_lbl = []
    n_collected = 0

    for batch in loader:
        x      = batch['circRNA'].to(device)
        mask   = batch['circRNA_mask'].to(device)
        sites  = batch['sites']             # [B, L]  (-100=pad, 0/1=label)
        lengths = batch['length']           # [B]

        # Forward through backbone only
        x_emb = model.embedding(x)
        emb, _ = model.backbone(x_emb, mask, None, None)  # [B, L, D]

        emb_np   = emb.cpu().float().numpy()
        sites_np = sites.numpy()  # [B, L]

        for b in range(emb.shape[0]):
            seq_len = int(lengths[b, 0].item())
            # positions 1..seq_len (skip CLS=0 if skip_cls=True)
            start = 1 if skip_cls else 0
            end   = min(start + seq_len, emb_np.shape[1])

            pos_emb = emb_np[b, start:end]          # [L, D]
            pos_lbl = sites_np[b, start:end]         # [L]

            # Keep only non-pad positions
            valid_mask = (pos_lbl != -100)
            pos_emb = pos_emb[valid_mask]
            pos_lbl = pos_lbl[valid_mask]

            all_emb.append(pos_emb)
            all_lbl.append(pos_lbl)
            n_collected += pos_emb.shape[0]

        if n_samples is not None and n_collected >= n_samples:
            break

    embeddings = np.concatenate(all_emb, axis=0)
    labels     = np.concatenate(all_lbl, axis=0)

    # subsample if needed
    if n_samples is not None and len(embeddings) > n_samples:
        # Stratified subsample: preserve binding/non-binding ratio
        idx_pos = np.where(labels == 1)[0]
        idx_neg = np.where(labels == 0)[0]
        ratio   = len(idx_pos) / (len(idx_pos) + len(idx_neg))
        n_pos   = int(n_samples * ratio)
        n_neg   = n_samples - n_pos
        n_pos   = min(n_pos, len(idx_pos))
        n_neg   = min(n_neg, len(idx_neg))
        rng     = np.random.default_rng(42)
        sel_pos = rng.choice(idx_pos, n_pos, replace=False)
        sel_neg = rng.choice(idx_neg, n_neg, replace=False)
        sel     = np.concatenate([sel_pos, sel_neg])
        embeddings = embeddings[sel]
        labels     = labels[sel]

    return embeddings, labels


# ── UMAP / t-SNE ─────────────────────────────────────────────────────────────
def run_reduction(embeddings, method='umap', seed=42):
    if method == 'umap':
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=seed,
                                n_neighbors=30, min_dist=0.1, metric='cosine')
        except ImportError:
            raise ImportError("pip install umap-learn")
    elif method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=seed, perplexity=50,
                       learning_rate='auto', init='pca', n_jobs=-1)
    else:
        raise ValueError(f"Unknown method: {method}")

    print(f"  Running {method.upper()} on {len(embeddings)} points (d={embeddings.shape[1]})...")
    return reducer.fit_transform(embeddings)


# ── plotting ──────────────────────────────────────────────────────────────────
def compute_separation_score(coords, labels):
    """
    Simple separation metric:
    ratio of inter-class centroid distance to mean within-class spread.
    Higher = better separation.
    """
    pos = coords[labels == 1]
    neg = coords[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    c_pos = pos.mean(axis=0)
    c_neg = neg.mean(axis=0)
    inter_dist = np.linalg.norm(c_pos - c_neg)
    spread_pos = np.sqrt(((pos - c_pos)**2).sum(axis=1)).mean()
    spread_neg = np.sqrt(((neg - c_neg)**2).sum(axis=1)).mean()
    return inter_dist / (spread_pos + spread_neg + 1e-8)


def plot_umap_grid(results, out_path, method='UMAP', n_samples=None):
    """
    results: list of (label, coords, labels_01, separation_score)
    """
    n = len(results)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig_w = ncols * 4.0
    fig_h = nrows * 4.2

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
    axes = np.array(axes).flatten()

    for i, (label, coords, lbl, sep) in enumerate(results):
        ax = axes[i]
        idx_neg = lbl == 0
        idx_pos = lbl == 1

        # Non-binding first (background)
        ax.scatter(coords[idx_neg, 0], coords[idx_neg, 1],
                   c=COLORS['non-binding'], s=1.5, alpha=0.25, linewidths=0,
                   label='Non-binding', rasterized=True)
        ax.scatter(coords[idx_pos, 0], coords[idx_pos, 1],
                   c=COLORS['binding'], s=2.5, alpha=0.60, linewidths=0,
                   label='Binding', rasterized=True)

        ax.set_title(f'{label}\nSep={sep:.3f}', fontsize=11, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(f'{method}1', fontsize=8)
        ax.set_ylabel(f'{method}2', fontsize=8)

        if i == 0:
            ax.legend(markerscale=4, fontsize=8, loc='upper right',
                      framealpha=0.8, edgecolor='gray')

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    n_str = f'{n_samples//1000}k' if n_samples else 'all'
    fig.suptitle(
        f'Position-level Backbone Embeddings ({method})\n'
        f'binding=orange · non-binding=blue · n≈{n_str}/model',
        fontsize=12, y=1.01
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved → {out_path}")
    plt.close()


def plot_side_by_side_shared(results, out_path, method='UMAP'):
    """
    All models overlaid in same UMAP space is misleading since each model has
    its own projection. This version shows them side-by-side with a shared
    colorbar / legend and separation score bar chart on the right.
    """
    n = len(results)
    fig = plt.figure(figsize=(n * 3.8 + 2.5, 4.5))
    gs  = fig.add_gridspec(1, n + 1, width_ratios=[3.5]*n + [2.0], wspace=0.05)

    for i, (label, coords, lbl, sep) in enumerate(results):
        ax = fig.add_subplot(gs[i])
        idx_neg, idx_pos = lbl == 0, lbl == 1

        ax.scatter(coords[idx_neg, 0], coords[idx_neg, 1],
                   c=COLORS['non-binding'], s=1.5, alpha=0.25, linewidths=0, rasterized=True)
        ax.scatter(coords[idx_pos, 0], coords[idx_pos, 1],
                   c=COLORS['binding'], s=2.5, alpha=0.55, linewidths=0, rasterized=True)

        pct_pos = 100 * lbl.sum() / len(lbl)
        ax.set_title(f'{label}', fontsize=11, fontweight='bold')
        ax.set_xlabel(f'Sep={sep:.3f}  |  {pct_pos:.1f}% binding', fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            ax.set_ylabel(f'{method}', fontsize=9)

    # Separation score bar chart
    ax_bar = fig.add_subplot(gs[n])
    labels_  = [r[0] for r in results]
    seps_    = [r[3] for r in results]
    colors_  = plt.cm.RdYlGn(np.linspace(0.2, 0.85, len(seps_)))
    bars = ax_bar.barh(range(len(seps_)), seps_, color=colors_, edgecolor='gray', linewidth=0.5)
    ax_bar.set_yticks(range(len(seps_)))
    ax_bar.set_yticklabels(labels_, fontsize=9)
    ax_bar.set_xlabel('Separation Score', fontsize=9)
    ax_bar.set_title('Separation', fontsize=10, fontweight='bold')
    ax_bar.invert_yaxis()
    for bar, v in zip(bars, seps_):
        ax_bar.text(v + 0.005, bar.get_y() + bar.get_height()/2,
                    f'{v:.3f}', va='center', fontsize=8)

    # Legend
    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor=COLORS['binding'],    markersize=6, label='Binding'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor=COLORS['non-binding'],markersize=6, label='Non-binding'),
    ]
    fig.legend(handles=legend_els, loc='lower center', ncol=2, fontsize=9,
               bbox_to_anchor=(0.5*(n)/(n+0.55), -0.06), framealpha=0.9)

    fig.suptitle(
        f'Position-level Backbone Embeddings ({method})\nEach panel: independent projection',
        fontsize=12, y=1.03
    )
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved → {out_path}")
    plt.close()


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['nopt', 'mlm'],
                        choices=list(MODEL_REGISTRY.keys()),
                        help='Models to compare (default: nopt mlm)')
    parser.add_argument('--method', default='umap', choices=['umap', 'tsne'],
                        help='Dimensionality reduction method')
    parser.add_argument('--n_samples', type=int, default=20000,
                        help='Max positions per model (stratified subsample)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='cuda / cpu / 0 / 1 ...')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--out', default=None,
                        help='Output path (default: docs/paper_cmi/umap_<models>.pdf)')
    parser.add_argument('--layout', default='side',
                        choices=['side', 'grid'],
                        help='side=side-by-side+sep-bar  grid=4-per-row')
    args = parser.parse_args()

    device_str = args.device
    if device_str.isdigit():
        device_str = f'cuda:{device_str}'
    device = torch.device(device_str)

    # default output path
    if args.out is None:
        model_tag = '_'.join(args.models)
        args.out = str(ROOT / f'docs/paper_cmi/umap_{model_tag}_{args.method}.pdf')

    print(f"Device: {device}")
    print(f"Models: {args.models}")
    print(f"Method: {args.method.upper()}, n_samples={args.n_samples}")
    print()

    # Build data loader
    print("Loading test dataset...")
    loader = build_test_loader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    results = []
    for key in args.models:
        exp_name, seed, model_name, display = MODEL_REGISTRY[key]
        print(f"\n[{display}] {exp_name} seed={seed}")

        # Check model exists
        model_dir = ROOT / 'saved_models' / model_name / exp_name / str(seed)
        if not model_dir.exists():
            print(f"  WARNING: {model_dir} not found, skipping.")
            continue

        # Load model
        model = build_model(exp_name, seed, model_name, device)

        # Extract embeddings
        print(f"  Extracting embeddings...")
        embs, lbls = extract_embeddings(
            model, loader, device, n_samples=args.n_samples
        )
        print(f"  → {len(embs)} positions  (binding={lbls.sum()}, non-binding={(lbls==0).sum()})")

        # Run UMAP/t-SNE
        coords = run_reduction(embs, method=args.method)

        # Separation score
        sep = compute_separation_score(coords, lbls)
        print(f"  → Separation score: {sep:.4f}")

        results.append((display, coords, lbls, sep))

        # Free model memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if not results:
        print("No models available. Check saved_models/ paths.")
        return

    # Plot
    print(f"\nPlotting {len(results)} models...")
    if args.layout == 'side' and len(results) <= 6:
        plot_side_by_side_shared(results, args.out, method=args.method.upper())
    else:
        plot_umap_grid(results, args.out, method=args.method.upper(),
                       n_samples=args.n_samples)

    # Also save PNG
    png_path = args.out.replace('.pdf', '.png')
    plot_func = plot_side_by_side_shared if (args.layout == 'side' and len(results) <= 6) else plot_umap_grid
    plot_func(results, png_path, method=args.method.upper())

    print(f"\nDone!")
    for label, _, lbls, sep in results:
        print(f"  {label:<12} sep={sep:.4f}  n={len(lbls)}")


if __name__ == '__main__':
    main()
