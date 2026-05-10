#!/usr/bin/env python3
"""
Position-level UMAP/t-SNE 시각화 (EXP1 & EXP3)

EXP3: circmac vs transformer vs mamba vs lstm vs hymba
EXP1: circmac vs rnabert/rnaernie/rnafm/rnamsm (frozen & trainable)

Usage:
    python docs/paper_cmi/run_umap_viz.py              # EXP3 + EXP1 전체
    python docs/paper_cmi/run_umap_viz.py --exp exp3   # EXP3만
    python docs/paper_cmi/run_umap_viz.py --exp exp1   # EXP1만
    python docs/paper_cmi/run_umap_viz.py --exp exp3 --mode pair           # backbone per-pair
    python docs/paper_cmi/run_umap_viz.py --exp exp3 --mode pair --level cross_attn  # after cross-attention
"""

import argparse, sys, os, pickle
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data import CircRNABindingSitesDataset
from models.model import ModelWrapper
from utils_config import get_model_config

DEVICE     = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
N_SAMPLES  = 20000
BATCH_SIZE = 64

BIND_COLOR = '#e6550d'
NEG_COLOR  = '#6baed6'

# ─────────────────────────────────────────────────────────────────────────────
# EXP3: Encoder Architecture (all circmac family, no RNA LM)
EXP3 = [
    # (model_name, exp_name, seed, display_label, model_type)
    # model_type: 'seq' = circmac/transformer/mamba/lstm/hymba (KmerTokenizer)
    ('circmac',     'v2_abl_full_s1',         1, 'CircMAC',     'seq'),
    ('transformer', 'v2_enc_transformer_s1',   1, 'Transformer', 'seq'),
    ('mamba',       'v2_enc_mamba_s1',         1, 'Mamba',       'seq'),
    ('lstm',        'v2_enc_lstm_s1',          1, 'LSTM',        'seq'),
    ('hymba',       'v2_enc_hymba_s1',         1, 'Hymba',       'seq'),
]

# EXP1: RNA LM Comparison (circmac baseline + RNA LMs)
# cfg tuple: (model_name, exp_name, seed, label, mtype [, rna_model_name [, max_pos]])
EXP1 = [
    ('circmac', 'v2_abl_full_s1',                 1, 'CircMAC',          'seq'),
    ('rnabert', 'exp1_fair_frozen_rnabert_s1',     1, 'RNABert (frozen)', 'frozen_rna', 'rnabert',  440),
    ('rnabert', 'exp1_fair_trainable_rnabert_s1',  1, 'RNABert (train)',  'train_rna',  'rnabert',  440),
    ('rnaernie','exp1_fair_frozen_rnaernie_s1',    1, 'RNAErnie (frozen)','frozen_rna', 'rnaernie', 512),
    ('rnaernie','exp1_fair_trainable_rnaernie_s1', 1, 'RNAErnie (train)', 'train_rna',  'rnaernie', 512),
    ('rnafm',   'exp1_fair_frozen_rnafm_s1',       1, 'RNA-FM (frozen)',  'frozen_rna', 'rnafm',    1024),
    ('rnafm',   'exp1_fair_trainable_rnafm_s1',    1, 'RNA-FM (train)',   'train_rna',  'rnafm',    1024),
    ('rnamsm',  'exp1_fair_frozen_rnamsm_s1',      1, 'RNA-MSM (frozen)', 'frozen_rna', 'rnamsm',   512),
    ('rnamsm',  'exp1_fair_trainable_rnamsm_s1',   1, 'RNA-MSM (train)',  'train_rna',  'rnamsm',   512),
]

# chr4 case study for per-pair mode
CHR4_ISOFORM = 'chr4|84678168,84679116|84678259,84679242|-'


# ── model loaders ────────────────────────────────────────────────────────────
def load_seq_model(model_name, exp_name, seed, with_cross_attn=False):
    """Load circmac/transformer/mamba/lstm/hymba."""
    config = get_model_config(model_name, d_model=128, n_layer=6,
                              verbose=False, vocab_size=11)
    model = ModelWrapper(config=config, name=model_name, device=DEVICE).to(DEVICE)
    path  = ROOT / 'saved_models' / model_name / exp_name / str(seed) / 'train' / 'model.pth'
    sd    = torch.load(str(path), map_location=DEVICE, weights_only=False)

    # Detect d_target from checkpoint
    if 'proj_target.0.weight' in sd:
        d_target = sd['proj_target.0.weight'].shape[1]
        model._set_proj_target(d_target=d_target)

    # Initialize cross_attention if requested and available
    if with_cross_attn and 'cross_attention.q_proj.weight' in sd:
        model._set_cross_attention()
        model = model.to(DEVICE)

    ms    = model.state_dict()
    filt  = {k: v for k, v in sd.items() if k in ms and v.shape == ms[k].shape}
    ms.update(filt); model.load_state_dict(ms)
    has_ca = 'cross_attention.q_proj.weight' in sd
    print(f"  Loaded {len(filt)}/{len(sd)} params  cross_attn={'yes' if has_ca else 'no'}")
    model.eval()
    return model, has_ca


def load_train_rna_model(model_name, exp_name, seed, with_cross_attn=False):
    """Load trainable RNA LM (encoder inside backbone)."""
    config = get_model_config(model_name, d_model=128, n_layer=0, verbose=False)
    config.rc = False
    config.trainable = True  # triggers PretrainedModel with encoder loaded
    model = ModelWrapper(config=config, name=model_name, device=DEVICE).to(DEVICE)
    path = ROOT / 'saved_models' / model_name / exp_name / str(seed) / 'train' / 'model.pth'
    sd   = torch.load(str(path), map_location=DEVICE, weights_only=False)
    if 'proj_target.0.weight' in sd:
        model._set_proj_target(d_target=sd['proj_target.0.weight'].shape[1])
    if with_cross_attn and 'cross_attention.q_proj.weight' in sd:
        model._set_cross_attention()
        model = model.to(DEVICE)
    ms   = model.state_dict()
    filt = {k: v for k, v in sd.items() if k in ms and v.shape == ms[k].shape}
    ms.update(filt); model.load_state_dict(ms)
    has_ca = 'cross_attention.q_proj.weight' in sd
    print(f"  Loaded {len(filt)}/{len(sd)} params  cross_attn={'yes' if has_ca else 'no'}")
    model.eval()
    return model, has_ca


def load_frozen_rna_model(model_name, exp_name, seed, rna_model_name,
                          with_cross_attn=False):
    """Load frozen RNA LM.
    Returns (model_wrapper, model_pt, has_ca).
    """
    from multimolecule.models import (RnaBertModel, RnaErnieModel,
                                       RnaFmModel, RnaMsmModel)
    _lm = {'rnabert': RnaBertModel, 'rnaernie': RnaErnieModel,
           'rnafm': RnaFmModel, 'rnamsm': RnaMsmModel}
    _hub = {'rnabert': 'multimolecule/rnabert', 'rnaernie': 'multimolecule/rnaernie',
            'rnafm': 'multimolecule/rnafm', 'rnamsm': 'multimolecule/rnamsm'}
    print(f"  Loading external {rna_model_name}...")
    model_pt = _lm[rna_model_name].from_pretrained(_hub[rna_model_name]).to(DEVICE)
    model_pt.eval()
    for p in model_pt.parameters(): p.requires_grad_(False)

    config = get_model_config(model_name, d_model=128, n_layer=0, verbose=False)
    config.rc = False
    model = ModelWrapper(config=config, name=model_name, device=DEVICE).to(DEVICE)
    path  = ROOT / 'saved_models' / model_name / exp_name / str(seed) / 'train' / 'model.pth'
    sd    = torch.load(str(path), map_location=DEVICE, weights_only=False)
    if 'proj_target.0.weight' in sd:
        model._set_proj_target(d_target=sd['proj_target.0.weight'].shape[1])
    if with_cross_attn and 'cross_attention.q_proj.weight' in sd:
        model._set_cross_attention()
        model = model.to(DEVICE)
    ms    = model.state_dict()
    filt  = {k: v for k, v in sd.items() if k in ms and v.shape == ms[k].shape}
    ms.update(filt); model.load_state_dict(ms)
    has_ca = 'cross_attention.q_proj.weight' in sd
    print(f"  Loaded {len(filt)}/{len(sd)} params  cross_attn={'yes' if has_ca else 'no'}")
    model.eval()
    return model, model_pt, has_ca


# ── embedding extraction ─────────────────────────────────────────────────────
@torch.no_grad()
def get_emb_seq(model, x, mask):
    """circmac/transformer/mamba/lstm/hymba → [L, D]"""
    x_emb = model.embedding(x)
    emb, _ = model.backbone(x_emb, mask, None, None)
    return emb  # [B, L, D]


@torch.no_grad()
def get_emb_train_rna(model, x, mask, max_pos=440):
    """trainable RNA LM → [B, L, D]"""
    emb, _ = model.backbone(x[:, :max_pos], mask[:, :max_pos], None, None)
    return emb


@torch.no_grad()
def get_emb_frozen_rna(model, model_pt, x, mask, max_pos=440):
    """frozen RNA LM: model_pt → hidden_state → backbone.to_in → [B, L, D]
    max_pos: model_pt's max position length (rnabert=440, others=512 or higher)
    """
    x_t    = x[:, :max_pos]
    mask_t = mask[:, :max_pos]
    out    = model_pt(input_ids=x_t, attention_mask=mask_t)
    hs     = out['last_hidden_state']       # [B, max_pos, D_rna]
    emb, _ = model.backbone(hs, mask_t, None, None)
    return emb   # [B, max_pos, D]


# ── global extraction (test set) ─────────────────────────────────────────────
def extract_global(model_cfg, loader, n_samples=N_SAMPLES):
    """Run all batches through model, collect position embeddings + labels."""
    model_name, exp_name, seed, label, mtype = model_cfg[:5]
    rna_model_name = model_cfg[5] if len(model_cfg) > 5 else None
    max_pos        = model_cfg[6] if len(model_cfg) > 6 else 1022

    mdir = ROOT / 'saved_models' / model_name / exp_name / str(seed)
    if not mdir.exists():
        print(f"  SKIP {label}: {mdir} not found")
        return None

    print(f"\n[{label}]")
    model_pt = None
    if mtype == 'seq':
        model, _ = load_seq_model(model_name, exp_name, seed)
        get_emb = lambda x, mask: get_emb_seq(model, x, mask)
    elif mtype == 'train_rna':
        model, _ = load_train_rna_model(model_name, exp_name, seed)
        get_emb = lambda x, mask: get_emb_train_rna(model, x, mask, max_pos=max_pos)
    elif mtype == 'frozen_rna':
        model, model_pt, _ = load_frozen_rna_model(model_name, exp_name, seed, rna_model_name)
        get_emb = lambda x, mask: get_emb_frozen_rna(model, model_pt, x, mask, max_pos=max_pos)

    all_e, all_l, n_total = [], [], 0
    for batch in loader:
        x     = batch['circRNA'].to(DEVICE)
        mask  = batch['circRNA_mask'].to(DEVICE)
        sites = batch['sites'].numpy()
        lens  = batch['length']

        emb_b  = get_emb(x, mask)  # [B, L', D]  L' may be < max_len
        emb_np = emb_b.cpu().float().numpy()

        for b in range(emb_np.shape[0]):
            sl  = int(lens[b, 0].item())
            end = min(1 + sl, emb_np.shape[1])  # clipped to actual emb length
            e   = emb_np[b, 1:end]
            l   = sites[b, 1:end]
            valid = l != -100
            all_e.append(e[valid]); all_l.append(l[valid])
            n_total += valid.sum()

        if n_total >= n_samples * 2:
            break

    embs = np.concatenate(all_e)
    lbls = np.concatenate(all_l)

    # stratified subsample
    idx_p = np.where(lbls == 1)[0]
    idx_n = np.where(lbls == 0)[0]
    ratio = len(idx_p) / (len(idx_p) + len(idx_n) + 1e-9)
    n_pos = min(int(n_samples * ratio), len(idx_p))
    n_neg = min(n_samples - n_pos, len(idx_n))
    rng   = np.random.default_rng(42)
    sel   = np.concatenate([rng.choice(idx_p, n_pos, replace=False),
                             rng.choice(idx_n, n_neg, replace=False)])
    embs, lbls = embs[sel], lbls[sel]

    print(f"  → {len(embs)} pts  binding={lbls.sum()}  non-binding={(lbls==0).sum()}")

    del model
    if model_pt is not None: del model_pt
    torch.cuda.empty_cache()
    return label, embs, lbls


# ── cross-attention embedding ────────────────────────────────────────────────
@torch.no_grad()
def get_emb_cross_attn(model, model_target, circ_t, circ_m, tgt_t, tgt_m,
                       mtype='seq', model_pt=None, max_pos=1022):
    """Apply cross-attention: circRNA backbone + miRNA rnabert context → [B, L, D]"""
    # 1. circRNA backbone embedding
    if mtype == 'seq':
        x_emb = model.embedding(circ_t)
        emb, _ = model.backbone(x_emb, circ_m, None, None)
    elif mtype == 'train_rna':
        emb, _ = model.backbone(circ_t[:, :max_pos], circ_m[:, :max_pos], None, None)
    elif mtype == 'frozen_rna':
        out = model_pt(input_ids=circ_t[:, :max_pos], attention_mask=circ_m[:, :max_pos])
        hs  = out['last_hidden_state']
        emb, _ = model.backbone(hs, circ_m[:, :max_pos], None, None)
    # 2. miRNA via rnabert target model
    target_out  = model_target(tgt_t, tgt_m)
    target_hs   = target_out['last_hidden_state']             # [1, L_t, D_t]
    target_proj = model.get_target_projected(target_hs, mode='None')  # [1, L_t, D]
    # 3. cross-attention
    emb_out, _  = model.cross_attention(emb, target_proj, target_proj, tgt_m)
    return emb_out  # [1, L, D]


# ── per-pair extraction (case study) ────────────────────────────────────────
def extract_pair(model_cfg, isoform_id, test_df, level='backbone',
                 target_model_rnabert=None):
    """For a specific circRNA, extract per-position embeddings × all miRNA pairs.

    level: 'backbone'   → circRNA encoder only (same for all pairs)
           'cross_attn' → after cross-attention with rnabert miRNA (per-pair)
    """
    model_name, exp_name, seed, label, mtype = model_cfg[:5]
    rna_model_name = model_cfg[5] if len(model_cfg) > 5 else None
    max_pos        = model_cfg[6] if len(model_cfg) > 6 else 1022

    mdir = ROOT / 'saved_models' / model_name / exp_name / str(seed)
    if not mdir.exists():
        print(f"  SKIP {label}")
        return None

    print(f"\n[{label}]")
    model_pt = None
    has_ca = False
    want_ca = (level == 'cross_attn')
    if mtype == 'seq':
        model, has_ca = load_seq_model(model_name, exp_name, seed,
                                       with_cross_attn=want_ca)
        get_emb = lambda x, mask: get_emb_seq(model, x, mask)
    elif mtype == 'train_rna':
        model, has_ca = load_train_rna_model(model_name, exp_name, seed,
                                             with_cross_attn=want_ca)
        get_emb = lambda x, mask: get_emb_train_rna(model, x, mask, max_pos=max_pos)
    elif mtype == 'frozen_rna':
        model, model_pt, has_ca = load_frozen_rna_model(
            model_name, exp_name, seed, rna_model_name, with_cross_attn=want_ca)
        get_emb = lambda x, mask: get_emb_frozen_rna(model, model_pt, x, mask, max_pos=max_pos)

    use_cross_attn = (want_ca and has_ca and target_model_rnabert is not None)
    if want_ca and not use_cross_attn:
        print(f"  [WARN] {label}: cross_attn weights not in checkpoint → backbone fallback")

    sub = test_df[test_df['isoform_ID'] == isoform_id].reset_index(drop=True)
    ds  = CircRNABindingSitesDataset(df=sub, max_len=1022, k=1)

    item0 = ds[0]
    circ_t   = item0['circRNA'].unsqueeze(0).to(DEVICE)
    circ_m   = item0['circRNA_mask'].unsqueeze(0).to(DEVICE)
    circ_len = int(item0['length'].item())

    if not use_cross_attn:
        # Backbone: extract once, shared for all pairs
        with torch.no_grad():
            emb_b = get_emb(circ_t, circ_m)  # [1, L, D]
        seq_end = min(1 + circ_len, emb_b.shape[1])
        backbone_np = emb_b[0, 1:seq_end].cpu().float().numpy()

    pairs = []
    for i in range(len(sub)):
        item     = ds[i]
        sites    = item['sites'].numpy()
        lbls     = sites[1:1+circ_len]
        valid    = lbls != -100
        mirna_id = sub.iloc[i]['miRNA_ID']
        n_bind   = sub.iloc[i]['n_binding_site']

        if use_cross_attn:
            tgt_t = item['target'].unsqueeze(0).to(DEVICE)
            tgt_m = item['target_mask'].unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                emb_ca = get_emb_cross_attn(model, target_model_rnabert,
                                            circ_t, circ_m, tgt_t, tgt_m,
                                            mtype=mtype, model_pt=model_pt,
                                            max_pos=max_pos)
            # for frozen/train_rna, emb is truncated to max_pos → use min
            seq_end = min(1 + circ_len, emb_ca.shape[1])
            emb_np = emb_ca[0, 1:seq_end].cpu().float().numpy()
        else:
            emb_np = backbone_np

        # labels must match embedding length
        min_len = min(len(emb_np), len(lbls))
        pairs.append((mirna_id, emb_np[:min_len][valid[:min_len]],
                      lbls[:min_len][valid[:min_len]], n_bind))
        print(f"  pair: {mirna_id}  n_binding={n_bind}")

    del model
    if model_pt is not None: del model_pt
    torch.cuda.empty_cache()
    return label, pairs


# ── dimensionality reduction ─────────────────────────────────────────────────
def run_umap(embs, seed=42):
    import umap
    print(f"    UMAP on {len(embs)} pts...")
    return umap.UMAP(n_components=2, random_state=seed,
                     n_neighbors=min(30, max(5, len(embs)//50)),
                     min_dist=0.1, metric='cosine').fit_transform(embs)


def run_tsne(embs, seed=42):
    from sklearn.manifold import TSNE
    print(f"    t-SNE on {len(embs)} pts...")
    return TSNE(n_components=2, random_state=seed,
                perplexity=min(50, max(5, len(embs)//20)),
                learning_rate='auto', init='pca', n_jobs=-1).fit_transform(embs)


def sep_score(coords, lbls):
    p, n = coords[lbls==1], coords[lbls==0]
    if len(p) == 0 or len(n) == 0: return 0.0
    cp, cn = p.mean(0), n.mean(0)
    d  = np.linalg.norm(cp - cn)
    sp = np.sqrt(((p-cp)**2).sum(1)).mean()
    sn = np.sqrt(((n-cn)**2).sum(1)).mean()
    return d / (sp + sn + 1e-8)


# ── plotting ─────────────────────────────────────────────────────────────────
def plot_comparison(results, out_prefix, method_label, n_samples):
    """results: list of (label, coords, lbls)"""
    n = len(results)
    ncols = min(n, 5)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols*3.8 + 1.5, nrows*4.0),
                             squeeze=False)

    for i, (label, coords, lbls) in enumerate(results):
        ax   = axes[i // ncols][i % ncols]
        s    = sep_score(coords, lbls)
        pct  = 100*lbls.sum()/len(lbls)
        ax.scatter(coords[lbls==0,0], coords[lbls==0,1],
                   c=NEG_COLOR,  s=1.5, alpha=0.25, linewidths=0, rasterized=True)
        ax.scatter(coords[lbls==1,0], coords[lbls==1,1],
                   c=BIND_COLOR, s=2.5, alpha=0.60, linewidths=0, rasterized=True)
        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.set_xlabel(f'Sep={s:.3f} | {pct:.1f}% binding', fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            from matplotlib.lines import Line2D
            ax.legend(handles=[
                Line2D([0],[0],marker='o',color='w',markerfacecolor=BIND_COLOR,markersize=6,label='Binding'),
                Line2D([0],[0],marker='o',color='w',markerfacecolor=NEG_COLOR, markersize=6,label='Non-binding'),
            ], fontsize=7, loc='upper right', framealpha=0.85)

    # Hide unused axes
    for j in range(i+1, nrows*ncols):
        axes[j//ncols][j%ncols].set_visible(False)

    # Sep bar
    seps   = [sep_score(c, l) for _, c, l in results]
    labels = [r[0] for r in results]
    fig2, ax2 = plt.subplots(figsize=(2.8, max(2.5, n*0.55 + 1.2)))
    colors_ = plt.cm.RdYlGn(np.linspace(0.2, 0.85, n))
    bars = ax2.barh(range(n), seps, color=colors_, edgecolor='gray', lw=0.5)
    ax2.set_yticks(range(n)); ax2.set_yticklabels(labels, fontsize=8)
    ax2.set_xlabel('Separation Score', fontsize=9)
    ax2.set_title('Separation', fontsize=10, fontweight='bold')
    ax2.invert_yaxis()
    for bar, v in zip(bars, seps):
        ax2.text(v+0.003, bar.get_y()+bar.get_height()/2, f'{v:.3f}', va='center', fontsize=7)
    fig2.tight_layout()
    for ext in ['pdf','png']:
        fig2.savefig(f'{out_prefix}_sep.{ext}', bbox_inches='tight', dpi=150)
    plt.close(fig2)

    n_str = f'{n_samples//1000}k' if n_samples else 'all'
    fig.suptitle(f'Position Embeddings ({method_label}) · n≈{n_str}/model\n'
                 f'binding=orange · non-binding=blue', fontsize=11, y=1.01)
    fig.tight_layout()
    for ext in ['pdf','png']:
        fig.savefig(f'{out_prefix}.{ext}', dpi=150, bbox_inches='tight')
        print(f"  Saved → {out_prefix}.{ext}")
    plt.close(fig)


def plot_per_pair_grid(results_by_model, isoform_id, method_label, out_prefix):
    """rows=models, cols=miRNA pairs"""
    n_models = len(results_by_model)
    n_pairs  = max(len(v) for v in results_by_model.values())
    fig, axes = plt.subplots(n_models, n_pairs,
                             figsize=(n_pairs*3.5, n_models*3.5 + 0.8),
                             squeeze=False)

    for mi, (model_label, pairs) in enumerate(results_by_model.items()):
        for pi, (mirna_id, coords, lbls, _) in enumerate(pairs):
            ax  = axes[mi][pi]
            s   = sep_score(coords, lbls)
            ax.scatter(coords[lbls==0,0], coords[lbls==0,1],
                       c=NEG_COLOR,  s=8,  alpha=0.30, linewidths=0, rasterized=True)
            ax.scatter(coords[lbls==1,0], coords[lbls==1,1],
                       c=BIND_COLOR, s=18, alpha=0.75, linewidths=0, rasterized=True)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlabel(f'Sep={s:.3f}', fontsize=8)
            if mi == 0:
                short = mirna_id.replace('hsa-','')
                ax.set_title(f'{short}\n(n={int(lbls.sum())})', fontsize=9)
            if pi == 0:
                ax.set_ylabel(model_label, fontsize=10, fontweight='bold')
            if mi == 0 and pi == 0:
                from matplotlib.lines import Line2D
                ax.legend(handles=[
                    Line2D([0],[0],marker='o',color='w',markerfacecolor=BIND_COLOR,markersize=5,label='Binding'),
                    Line2D([0],[0],marker='o',color='w',markerfacecolor=NEG_COLOR, markersize=5,label='Non-binding'),
                ], fontsize=7, loc='upper right', framealpha=0.8)

    circ_tag = isoform_id.split('|')[0]+'|'+isoform_id.split('|')[1][:15]
    fig.suptitle(f'Per-pair Embeddings ({method_label})\n{circ_tag}', fontsize=11, y=1.01)
    fig.tight_layout()
    for ext in ['pdf','png']:
        fig.savefig(f'{out_prefix}.{ext}', dpi=150, bbox_inches='tight')
        print(f"  Saved → {out_prefix}.{ext}")
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default='both', choices=['exp1','exp3','both'])
    parser.add_argument('--mode', default='global', choices=['global','pair'])
    parser.add_argument('--method', default='both', choices=['umap','tsne','both'])
    parser.add_argument('--level', default='backbone', choices=['backbone','cross_attn'],
                        help='backbone: w/o miRNA (encoder only); cross_attn: w/ miRNA (after interaction)')
    args = parser.parse_args()

    print(f"Device: {DEVICE}  level={args.level}")
    OUT = ROOT / 'figures_claude'
    OUT.mkdir(exist_ok=True)

    test_df = pickle.load(open(ROOT / 'data/df_test_final.pkl', 'rb'))

    methods = ['umap','tsne'] if args.method == 'both' else [args.method]
    exps    = {'exp3': EXP3, 'exp1': EXP1}
    if args.exp == 'both': run_exps = [('exp3', EXP3), ('exp1', EXP1)]
    else: run_exps = [(args.exp, exps[args.exp])]

    if args.mode == 'global':
        loader = DataLoader(
            CircRNABindingSitesDataset(df=test_df, max_len=1022, k=1),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
        )

        for exp_name, cfg_list in run_exps:
            print(f"\n{'='*60}\n{exp_name.upper()}: Global embedding analysis\n{'='*60}")
            all_data = []
            for cfg in cfg_list:
                res = extract_global(cfg, loader)
                if res: all_data.append(res)

            for method in methods:
                fn  = run_umap if method == 'umap' else run_tsne
                red = [(lb, fn(e), l) for lb, e, l in all_data]
                plot_comparison(red, str(OUT / f'umap_{exp_name}_{method}'),
                                method.upper(), N_SAMPLES)

    else:  # per-pair (case study)
        # Load rnabert target model once if cross_attn mode
        target_rnabert = None
        if args.level == 'cross_attn':
            from multimolecule.models import RnaBertModel
            print("Loading rnabert target model for cross-attention...")
            target_rnabert = RnaBertModel.from_pretrained('multimolecule/rnabert').to(DEVICE)
            target_rnabert.eval()
            for p in target_rnabert.parameters():
                p.requires_grad_(False)

        # level_tag: file suffix
        level_tag   = '_w_mirna'  if args.level == 'cross_attn' else '_wo_mirna'
        # human-readable label for plot titles
        level_label = 'w/ miRNA'  if args.level == 'cross_attn' else 'w/o miRNA'

        for exp_name, cfg_list in run_exps:
            print(f"\n{'='*60}\n{exp_name.upper()}: Per-pair ({level_label}) chr4 case\n{'='*60}")
            model_results = {}
            for cfg in cfg_list:
                res = extract_pair(cfg, CHR4_ISOFORM, test_df,
                                   level=args.level,
                                   target_model_rnabert=target_rnabert)
                if res:
                    label, pairs = res
                    model_results[label] = pairs

            for method in methods:
                fn = run_umap if method == 'umap' else run_tsne
                reduced = {}
                for label, pairs in model_results.items():
                    if args.level == 'cross_attn':
                        # Each pair has its own embedding (miRNA-specific)
                        reduced_pairs = []
                        for mirna, emb, lbls, n_bind in pairs:
                            print(f"  [{label}] {mirna} ({method})...")
                            coords = fn(emb)
                            reduced_pairs.append((mirna, coords, lbls, n_bind))
                        reduced[label] = reduced_pairs
                    else:
                        # Backbone: all pairs share same embedding, UMAP once
                        emb = pairs[0][1]
                        coords = fn(emb)
                        reduced[label] = [
                            (mirna, coords, lbls, n_bind)
                            for mirna, _, lbls, n_bind in pairs
                        ]
                plot_per_pair_grid(
                    reduced, CHR4_ISOFORM,
                    f'{method.upper()} · {level_label}',
                    str(OUT / f'umap_{exp_name}_{method}_pair{level_tag}')
                )

    print("\nAll done!")


if __name__ == '__main__':
    main()
