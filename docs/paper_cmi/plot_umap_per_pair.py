#!/usr/bin/env python3
"""
Per-pair position-level UMAP/t-SNE visualization for case studies.

하나의 circRNA에 대해 miRNA pair별로 binding site 레이블을 붙여서
backbone embedding의 분리도를 시각화합니다.

두 가지 레벨:
  backbone : circRNA encoder output (miRNA context 없음, pair-independent)
  cross_attn: cross-attention 이후 embedding (miRNA context 있음, 진짜 per-pair)
              → 이 옵션은 rnabert model_target 필요

Usage:
    # chr4 case, backbone level, NoPT vs MLM 비교
    python docs/paper_cmi/plot_umap_per_pair.py

    # 특정 isoform + miRNA 지정
    python docs/paper_cmi/plot_umap_per_pair.py \
        --isoform "chr4|84678168,84679116|84678259,84679242|-" \
        --mirna hsa-miR-449b-5p

    # cross-attention 레벨 (rnabert 필요)
    python docs/paper_cmi/plot_umap_per_pair.py --level cross_attn
"""

import argparse, sys, pickle, torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data import CircRNABindingSitesDataset
from models.model import ModelWrapper
from utils_config import get_model_config

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ── 비교할 모델 목록 ──────────────────────────────────────────────────────────
MODELS_TO_COMPARE = [
    ('circmac', 'v2_abl_full_s1',   1, 'NoPT'),
    ('circmac', 'v2_pt_mlm_s1',     1, 'MLM'),
    ('circmac', 'v2_pt_mlm_ssp_s1', 1, 'MLM+SSP'),
]

# Case study 기본값 (chr4 main case)
DEFAULT_ISOFORM = 'chr4|84678168,84679116|84678259,84679242|-'

# miRNA별 색상
PAIR_COLORS = ['#e41a1c','#377eb8','#4daf4a','#984ea3',
               '#ff7f00','#a65628','#f781bf','#999999']

BIND_COLOR  = '#e6550d'
NEG_COLOR   = '#6baed6'


# ── model loader ─────────────────────────────────────────────────────────────
def load_m(model_name, exp_name, seed, with_cross_attn=False):
    config = get_model_config(model_name, d_model=128, n_layer=6,
                              verbose=False, vocab_size=11)
    model = ModelWrapper(config=config, name=model_name, device=DEVICE).to(DEVICE)
    model_path = ROOT / 'saved_models' / model_name / exp_name / str(seed) / 'train' / 'model.pth'
    sd = torch.load(str(model_path), map_location=DEVICE, weights_only=False)

    # Detect d_target from checkpoint (rnabert=120, default=128)
    if 'proj_target.0.weight' in sd:
        d_target = sd['proj_target.0.weight'].shape[1]
        model._set_proj_target(d_target=d_target)

    # Initialize cross_attention head if needed
    if with_cross_attn and 'cross_attention.q_proj.weight' in sd:
        model._set_cross_attention()

    model = model.to(DEVICE)
    ms = model.state_dict()
    filtered = {k: v for k, v in sd.items() if k in ms and v.shape == ms[k].shape}
    ms.update(filtered)
    model.load_state_dict(ms)
    print(f"  Loaded {len(filtered)}/{len(sd)} params")
    model.eval()
    return model


# ── dataset for a single circRNA + all its miRNA pairs ───────────────────────
def get_pairs(df, isoform_id):
    """Returns subset df filtered by isoform_id."""
    sub = df[df['isoform_ID'] == isoform_id].reset_index(drop=True)
    return sub


class SingleCircDataset(Dataset):
    """Wraps a subset df for a single circRNA."""
    def __init__(self, df_sub, max_len=1022):
        self.ds = CircRNABindingSitesDataset(df=df_sub, max_len=max_len, k=1)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        return item


# ── embedding extraction for one circRNA ────────────────────────────────────
@torch.no_grad()
def get_backbone_embs(model, circRNA_tokens, mask):
    """
    circRNA_tokens: [1, L] LongTensor
    Returns: [L-1, 128]  (skip CLS)
    """
    x_emb = model.embedding(circRNA_tokens)
    emb, _ = model.backbone(x_emb, mask, None, None)  # [1, L, D]
    return emb[0, 1:].cpu().float().numpy()           # [L-1, D]


@torch.no_grad()
def get_cross_attn_embs(model, model_target, circRNA_tokens, mask,
                         target_tokens, target_mask):
    """
    Returns per-position embedding AFTER cross-attention: [L-1, 128]
    Requires model.cross_attention to be loaded.
    """
    x_emb = model.embedding(circRNA_tokens)
    emb, _ = model.backbone(x_emb, mask, None, None)  # [1, L, D]

    # miRNA through target model
    target_out = model_target(target_tokens, target_mask)
    target_hs  = target_out['last_hidden_state']       # [1, L_t, D_t]

    # project target (sequence-level, no pooling)
    target_proj = model.get_target_projected(target_hs, mode='None')  # [1, L_t, D]

    emb_out, _ = model.cross_attention(emb, target_proj, target_proj, target_mask)
    return emb_out[0, 1:].cpu().float().numpy()


# ── dimensionality reduction ─────────────────────────────────────────────────
def run_umap(embs, seed=42):
    import umap
    return umap.UMAP(n_components=2, random_state=seed,
                     n_neighbors=min(30, len(embs)//5),
                     min_dist=0.1, metric='cosine').fit_transform(embs)


def run_tsne(embs, seed=42):
    from sklearn.manifold import TSNE
    return TSNE(n_components=2, random_state=seed, perplexity=min(50, len(embs)//5),
                learning_rate='auto', init='pca', n_jobs=-1).fit_transform(embs)


def sep_score(coords, lbls):
    p, n = coords[lbls==1], coords[lbls==0]
    if len(p) == 0 or len(n) == 0: return 0.0
    cp, cn = p.mean(0), n.mean(0)
    d = np.linalg.norm(cp - cn)
    sp = np.sqrt(((p-cp)**2).sum(1)).mean()
    sn = np.sqrt(((n-cn)**2).sum(1)).mean()
    return d / (sp + sn + 1e-8)


# ── plotting ─────────────────────────────────────────────────────────────────
def plot_per_pair(results_by_model, isoform_id, pairs_info, method, out_prefix):
    """
    results_by_model: dict model_label → list of (mirna, coords, lbls)
    pairs_info: list of (mirna_id, n_binding)
    method: 'UMAP' | 't-SNE'
    """
    n_models = len(results_by_model)
    n_pairs  = len(pairs_info)

    # Layout: rows = models, cols = pairs
    fig, axes = plt.subplots(n_models, n_pairs,
                             figsize=(n_pairs * 3.5, n_models * 3.5 + 0.8),
                             squeeze=False)

    for mi, (model_label, pair_results) in enumerate(results_by_model.items()):
        for pi, (mirna_id, coords, lbls) in enumerate(pair_results):
            ax = axes[mi][pi]
            idx_n = lbls == 0
            idx_p = lbls == 1
            ax.scatter(coords[idx_n,0], coords[idx_n,1],
                       c=NEG_COLOR,  s=8, alpha=0.35, linewidths=0,
                       rasterized=True, label='Non-binding')
            ax.scatter(coords[idx_p,0], coords[idx_p,1],
                       c=BIND_COLOR, s=18, alpha=0.75, linewidths=0,
                       rasterized=True, label='Binding')

            s = sep_score(coords, lbls)
            n_bind = lbls.sum()
            ax.set_xticks([]); ax.set_yticks([])

            # Column header: miRNA name (top row only)
            if mi == 0:
                short_mirna = mirna_id.replace('hsa-', '')
                ax.set_title(f'{short_mirna}\n(n_bind={n_bind})', fontsize=9)

            # Row header: model name (leftmost col only)
            if pi == 0:
                ax.set_ylabel(model_label, fontsize=10, fontweight='bold')

            ax.set_xlabel(f'Sep={s:.3f}', fontsize=8)

            # Legend only once
            if mi == 0 and pi == 0:
                ax.legend(markerscale=2, fontsize=7, loc='upper right',
                          framealpha=0.8, edgecolor='gray')

    circ_short = isoform_id.split('|')[0] + '|' + isoform_id.split('|')[1][:20]
    fig.suptitle(
        f'Per-pair Position Embeddings ({method})\n{circ_short}',
        fontsize=11, y=1.01
    )
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        p = f'{out_prefix}.{ext}'
        fig.savefig(p, dpi=150, bbox_inches='tight')
        print(f"  Saved → {p}")
    plt.close(fig)


def plot_multi_pair_overlay(results_by_model, isoform_id, pairs_info, method, out_prefix):
    """
    Overlay all pairs in one UMAP space (shared embedding per model).
    Each pair has its own color for binding sites; non-binding = gray background.
    Row = models, single UMAP per model (all pairs combined).
    """
    n_models = len(results_by_model)
    fig, axes = plt.subplots(1, n_models, figsize=(n_models * 4, 4), squeeze=False)
    axes = axes[0]

    for mi, (model_label, pair_results) in enumerate(results_by_model.items()):
        ax = axes[mi]

        # All pairs share the same circRNA embedding → same UMAP coords
        # Use first pair's coords (same circRNA, same model = same embedding)
        coords = pair_results[0][1]
        # Plot non-binding background (gray)
        all_bind = np.zeros(len(coords), dtype=bool)
        for _, _, lbls in pair_results:
            all_bind |= (lbls == 1)

        ax.scatter(coords[~all_bind,0], coords[~all_bind,1],
                   c='#cccccc', s=6, alpha=0.3, linewidths=0,
                   rasterized=True, zorder=1, label='Non-binding (any)')

        # Each pair's binding sites in a different color
        for pi, (mirna_id, coords_p, lbls) in enumerate(pair_results):
            color = PAIR_COLORS[pi % len(PAIR_COLORS)]
            short = mirna_id.replace('hsa-miR-', 'miR-')
            ax.scatter(coords_p[lbls==1,0], coords_p[lbls==1,1],
                       c=color, s=25, alpha=0.8, linewidths=0,
                       rasterized=True, zorder=pi+2, label=short)

        ax.set_title(model_label, fontsize=11, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(method, fontsize=8)
        ax.legend(fontsize=7, loc='upper right', framealpha=0.8,
                  markerscale=2, edgecolor='gray')

    circ_short = isoform_id.split('|')[0] + '|' + isoform_id.split('|')[1][:15]
    fig.suptitle(
        f'Per-pair Binding Sites Overlay ({method})\n{circ_short}\n'
        f'(shared backbone embedding per model, colored by miRNA pair)',
        fontsize=10, y=1.04
    )
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        p = f'{out_prefix}_overlay.{ext}'
        fig.savefig(p, dpi=150, bbox_inches='tight')
        print(f"  Saved → {p}")
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--isoform', default=DEFAULT_ISOFORM)
    parser.add_argument('--mirna', default=None,
                        help='If set, use only this miRNA pair. Else use all.')
    parser.add_argument('--method', default='umap', choices=['umap', 'tsne', 'both'])
    parser.add_argument('--level', default='backbone',
                        choices=['backbone', 'cross_attn'],
                        help='backbone: no miRNA context; cross_attn: rnabert needed')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    # ── load data ─────────────────────────────────────────────────────────────
    test_df = pickle.load(open(ROOT / 'data/df_test_final.pkl', 'rb'))
    pairs_df = get_pairs(test_df, args.isoform)

    if len(pairs_df) == 0:
        print(f"No pairs found for isoform: {args.isoform}")
        return

    if args.mirna:
        pairs_df = pairs_df[pairs_df['miRNA_ID'] == args.mirna]

    print(f"Isoform: {args.isoform}")
    print(f"Found {len(pairs_df)} miRNA pairs:")
    for _, row in pairs_df.iterrows():
        print(f"  {row['miRNA_ID']}  n_binding={row['n_binding_site']}")

    # circRNA is the same for all pairs → tokenize once
    ds_full = CircRNABindingSitesDataset(df=pairs_df, max_len=1022, k=1)
    # All rows have the same circRNA sequence

    circ_item   = ds_full[0]
    circ_tokens = circ_item['circRNA'].unsqueeze(0).to(DEVICE)   # [1, L]
    circ_mask   = circ_item['circRNA_mask'].unsqueeze(0).to(DEVICE)
    circ_len    = int(circ_item['length'].item())

    # Ground truth labels per pair
    pairs_info = []
    for i, row in pairs_df.iterrows():
        item   = ds_full[pairs_df.index.get_loc(i)]
        sites  = item['sites'].numpy()   # [L]
        # valid positions (not -100)
        valid  = sites != -100
        lbls   = sites[valid]
        pairs_info.append((row['miRNA_ID'], lbls, item))

    print(f"circRNA length: {circ_len}")

    # ── target model for cross_attn ───────────────────────────────────────────
    model_target = None
    if args.level == 'cross_attn':
        from multimolecule import RnaTokenizer
        from multimolecule.models import RnaBertModel
        print("Loading rnabert target model...")
        model_target = RnaBertModel.from_pretrained('multimolecule/rnabert').to(DEVICE)
        model_target.eval()
        for p in model_target.parameters():
            p.requires_grad_(False)

    # ── process each model ────────────────────────────────────────────────────
    methods = ['umap', 'tsne'] if args.method == 'both' else [args.method]
    out_dir = Path(ROOT / 'docs/paper_cmi')
    circ_tag = args.isoform.replace('|', '_').replace(',', '-')[:40]

    results_by_model = {}
    for model_name, exp_name, seed, label in MODELS_TO_COMPARE:
        mdir = ROOT / 'saved_models' / model_name / exp_name / str(seed)
        if not mdir.exists():
            print(f"  SKIP {label}: not found")
            continue

        print(f"\n[{label}] loading...")
        model = load_m(model_name, exp_name, seed,
                       with_cross_attn=(args.level == 'cross_attn'))

        # Get backbone embedding for this circRNA (same for all pairs)
        with torch.no_grad():
            backbone_emb = get_backbone_embs(model, circ_tokens, circ_mask)
            # backbone_emb: [circ_len, 128]  (positions 1..circ_len)

        results_by_model[label] = []
        for mirna_id, lbls, item in pairs_info:
            print(f"  pair: {mirna_id}  n_binding={lbls.sum()}")

            if args.level == 'cross_attn' and model_target is not None:
                # Get target tokens for this miRNA
                tgt_tokens = item['target'].unsqueeze(0).to(DEVICE)
                tgt_mask   = item['target_mask'].unsqueeze(0).to(DEVICE)
                emb_use = get_cross_attn_embs(
                    model, model_target,
                    circ_tokens, circ_mask,
                    tgt_tokens, tgt_mask
                )
            else:
                emb_use = backbone_emb  # same for all pairs (backbone is miRNA-agnostic)

            # emb_use: [circ_len, 128]  lbls: [circ_len]
            min_len = min(len(emb_use), len(lbls))
            emb_use = emb_use[:min_len]
            lbls_use = lbls[:min_len]

            results_by_model[label].append((mirna_id, emb_use, lbls_use))

        del model
        torch.cuda.empty_cache()

    if not results_by_model:
        print("No models loaded.")
        return

    # ── run reduction + plot ──────────────────────────────────────────────────
    for method in methods:
        print(f"\nRunning {method.upper()}...")
        reduced_by_model = {}
        for label, pair_list in results_by_model.items():
            reduced = []
            for mirna_id, emb, lbls in pair_list:
                print(f"  [{label}] {mirna_id} ({method})...")
                fn = run_umap if method == 'umap' else run_tsne
                coords = fn(emb)
                s = sep_score(coords, lbls)
                print(f"    → sep={s:.4f}")
                reduced.append((mirna_id, coords, lbls))
            reduced_by_model[label] = reduced

        # Plot 1: grid (rows=models, cols=pairs)
        out_prefix = str(out_dir / f'umap_pair_{circ_tag}_{method}_{args.level}')
        pairs_meta = [(m, l.sum()) for m, l, _ in results_by_model[list(results_by_model.keys())[0]]]
        plot_per_pair(reduced_by_model, args.isoform, pairs_meta, method.upper(), out_prefix)

        # Plot 2: overlay (all pairs on one plot per model)
        plot_multi_pair_overlay(reduced_by_model, args.isoform, pairs_meta,
                                method.upper(), out_prefix)

    print("\nDone!")


if __name__ == '__main__':
    main()
