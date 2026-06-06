#!/usr/bin/env python3
"""
generate_embedding_cache.py
Extract position-level embeddings (w/o miRNA, w/ miRNA) for new case studies
and save UMAP-reduced coordinates to figures_paper/embedding_cache/.

Usage:
    python figures_paper/fig12/generate_embedding_cache.py [--device 0]
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from umap import UMAP
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from trainer import Trainer
from data import CircRNABindingSitesDataset, KmerTokenizer
from utils import get_device
from utils_config import get_model_config

CACHE_DIR = ROOT / 'figures_paper' / 'embedding_cache'
MODELS_DIR = ROOT / 'models_for_viz'

# ── Case definitions ──────────────────────────────────────────────────────────
CASES = {
    'huwe1': dict(
        csv=ROOT / 'figures_paper/Fig_Case_HUWE1/data_predictions.csv',
        isoform='chrX|53645311,53647368,53648212,53648573,53654063|53645463,53647574,53648310,53649000,53654131|-',
        mirna='hsa-miR-29b-3p',
    ),
    'rad23b': dict(
        csv=ROOT / 'figures_paper/Fig_Case_RAD23B/data_predictions.csv',
        isoform='chr9|107302035,107306379,107311682,107318752,107321983,107323890,107324834|107302114,107306647,107311737,107318879,107322118,107324017,107325004|+',
        mirna='hsa-miR-3184-5p',
    ),
    'fbxo7': dict(
        csv=ROOT / 'figures_paper/Fig_Case_FBXO7/data_predictions.csv',
        isoform='chr22|32478981,32483897,32485068,32487745|32479275,32484124,32485209,32487828|+',
        mirna='hsa-miR-21-3p',
    ),
    'nfib': dict(
        csv=ROOT / 'figures_paper/Fig_Case_NFIB/data_predictions.csv',
        isoform='chr9|14120440,14125632,14146689,14150145,14155825,14179727|14120624,14125766,14146807,14150265,14155893,14179780|-',
        mirna='hsa-miR-373-3p',
    ),
    'pt_chr17': dict(
        csv=ROOT / 'figures_paper/Fig_Case_PT_CHR17/data_predictions.csv',
        isoform='chr17|63193991,63200771|63194139,63200957|+',
        mirna='hsa-miR-4732-5p',
    ),
    'cdyl2': dict(
        csv=ROOT / 'figures_paper/Fig_Case_CDYL2/data_predictions.csv',
        isoform='chr4|84678168,84679116|84678259,84679242|-',
        mirna='hsa-miR-34c-5p',
    ),
    'mapk1': dict(
        csv=ROOT / 'figures_paper/Fig_Case_MAPK1/data_predictions.csv',
        isoform='chr22|21799012,21805850,21807664|21799128,21806039,21807846|-',
        mirna='hsa-miR-12119',
    ),
    'app': dict(
        csv=ROOT / 'figures_paper/Fig_Case_APP/data_predictions.csv',
        isoform='chr21|25954590,25955627,25975070,25975954,25982344|25954689,25955755,25975228,25976028,25982477|-',
        mirna='hsa-miR-5001-3p',
    ),
}

# ── Model specs: (cache_name, model_name, exp_name, frozen_rna_lm) ────────────
MODEL_SPECS = [
    ('CircMAC',            'circmac',     'v2_abl_full',                False),
    ('Transformer',        'transformer', 'v2_enc_transformer',         False),
    ('Mamba',              'mamba',       'v2_enc_mamba',               False),
    ('LSTM',               'lstm',        'v2_enc_lstm',                False),
    ('Hymba',              'hymba',       'v2_enc_hymba',               False),
    # Frozen backbone
    ('RNABert (frozen)',   'rnabert',     'exp1_fair_frozen_rnabert',   True),
    ('RNAErnie (frozen)',  'rnaernie',    'exp1_fair_frozen_rnaernie',  True),
    ('RNA-MSM (frozen)',   'rnamsm',      'exp1_fair_frozen_rnamsm',    True),
    ('RNA-FM (frozen)',    'rnafm',       'exp1_fair_frozen_rnafm',     True),
    # Fine-tuned
    ('RNABert (ft)',       'rnabert',     'exp1_fair_trainable_rnabert', False),
    ('RNAErnie (ft)',      'rnaernie',    'exp1_fair_trainable_rnaernie',False),
    ('RNA-MSM (ft)',       'rnamsm',      'exp1_fair_trainable_rnamsm',  False),
    ('RNA-FM (ft)',        'rnafm',       'exp1_fair_trainable_rnafm',   False),
]

# Per-model max sequence length (positional embedding limit)
MODEL_MAX_LEN = {
    'rnabert':  436,
    'rnaernie': 511,
}

SEED = 1
MAX_LEN = 1022


def get_ckpt_vocab_size(model_name, exp_name, seed):
    exp_folder = f'{exp_name}_s{seed}'
    ckpt_path = MODELS_DIR / model_name / exp_folder / str(seed) / 'train' / 'model.pth'
    if not ckpt_path.exists():
        return None
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        if 'embedding.word_embeddings.weight' in ckpt:
            return ckpt['embedding.word_embeddings.weight'].shape[0]
    except Exception:
        pass
    return None


def make_dataset(df, max_len, ckpt_vocab_size):
    ds = CircRNABindingSitesDataset(
        df.reset_index(drop=True),
        max_len=max_len,
        target_type='mirna',
        k=1,
        k_target=1,
    )
    if ckpt_vocab_size == 11 and ds.vocab_size != 11:
        ds.circrna_tokenizer = KmerTokenizer(1)
        ds.vocab_size = 11
    return ds


def setup_trainer(model_name, exp_name, seed, device, frozen_rna_lm, vocab_size):
    exp_folder = f'{exp_name}_s{seed}'
    trainer = Trainer(
        seed=seed, device=device,
        dir_save=str(MODELS_DIR),
        experiment_name=exp_folder,
        verbose=False,
    )
    trainer.task = 'sites'
    trainer.rc = False
    trainer.use_unified_head = False
    trainer.interaction = 'cross_attention'

    config = get_model_config(model_name=model_name, d_model=128, n_layer=6,
                               vocab_size=vocab_size)
    if model_name in ['rnabert', 'rnaernie', 'rnafm', 'rnamsm']:
        config.trainable = not frozen_rna_lm

    trainer.define_model(config=config, model_name=model_name, pretrain=False,
                         is_cross_attention=True, interaction='cross_attention',
                         site_head_type='conv1d')

    if model_name in ['rnabert', 'rnaernie', 'rnafm', 'rnamsm'] and frozen_rna_lm:
        trainer.define_pretrained_model(model_name=model_name)

    trainer.set_pretrained_target(target='mirna', rna_model='rnabert')

    ckpt_path = MODELS_DIR / model_name / exp_folder / str(seed) / 'train' / 'model.pth'
    if ckpt_path.exists():
        trainer.load_model_from_path(str(ckpt_path), verbose=False)
    else:
        trainer.load_model(epoch=None, pretrain=False, verbose=False)
    trainer.model.eval()
    return trainer


def extract_embeddings(trainer, dataset):
    """Return emb_wo [L,D] and emb_wi [L,D] for a SINGLE-pair dataset (1 row)."""
    assert len(dataset) == 1, f"Expected 1 sample, got {len(dataset)}"
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    with torch.no_grad():
        data = next(iter(loader))
        L = int(data['length'][0].item())

        # w/o miRNA: backbone encoder output
        emb, mask = trainer.forward(data)            # [1, seq_tokens, D]
        # CLS at 0, actual positions at 1..L
        emb_wo = emb[0, 1:L+1, :].cpu().numpy()     # [L, D]

        # w/ miRNA: after cross-attention
        target, target_mask = trainer.forward_target(data)
        emb_wi_t, _ = trainer.forward_cross_attention(emb, target, target_mask)
        emb_wi = emb_wi_t[0, 1:L+1, :].cpu().numpy()  # [L, D]

    return emb_wo, emb_wi


def run_umap(emb, n_neighbors=15, min_dist=0.1, seed=42):
    reducer = UMAP(n_components=2, n_neighbors=n_neighbors,
                   min_dist=min_dist, random_state=seed)
    return reducer.fit_transform(emb)


def run_tsne(emb, perplexity=30, seed=42):
    n = len(emb)
    perp = min(perplexity, max(5, n // 4))
    reducer = TSNE(n_components=2, perplexity=perp, random_state=seed, max_iter=1000)
    return reducer.fit_transform(emb)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    device = get_device(args.device)
    print(f'Device: {device}')

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    df_test = pd.read_pickle(ROOT / 'data' / 'df_test_final.pkl')

    for case_key, case_cfg in CASES.items():
        print(f'\n=== {case_key.upper()} ===')

        # Load ONE row from df_test_final (full sequence + sites)
        mask = (
            (df_test['isoform_ID'] == case_cfg['isoform']) &
            (df_test['miRNA_ID']   == case_cfg['mirna']) &
            (df_test['binding']    == 1)
        )
        df_pair = df_test[mask].reset_index(drop=True)

        if df_pair.empty:
            print('  No data — skipping'); continue

        L       = len(df_pair['circRNA'].iloc[0])
        lbls    = np.array(df_pair['sites'].iloc[0])
        pos_idx = np.arange(L)
        print(f'  L={L}, pos={lbls.sum()}')

        emb_data    = {}
        coords_data = {}
        tsne_data   = {}

        for cache_name, model_name, exp_name, frozen_rna_lm in MODEL_SPECS:
            print(f'  [{cache_name}]', end=' ', flush=True)
            # Skip if sequence exceeds the model's positional embedding limit
            model_max = MODEL_MAX_LEN.get(model_name, MAX_LEN)
            if L > model_max:
                print(f'SKIP (L={L} > max={model_max})')
                continue
            try:
                ckpt_vocab = get_ckpt_vocab_size(model_name, exp_name, SEED)
                ds = make_dataset(df_pair, model_max + 2, ckpt_vocab)
                vocab_size = ds.vocab_size

                trainer = setup_trainer(
                    model_name, exp_name, SEED, device,
                    frozen_rna_lm, vocab_size
                )

                emb_wo, emb_wi = extract_embeddings(trainer, ds)
                n = min(len(emb_wo), L)

                emb_data[cache_name] = {
                    'emb_wo':  emb_wo[:n],
                    'emb_wi':  emb_wi[:n],
                    'lbls':    lbls[:n].astype(int),
                    'pos_idx': pos_idx[:n],
                    'seq_len': int(L),
                }

                coords_wo = run_umap(emb_wo[:n])
                coords_wi = run_umap(emb_wi[:n])
                coords_data[cache_name] = {
                    'coords_wo': coords_wo,
                    'coords_wi': coords_wi,
                }

                tsne_wo = run_tsne(emb_wo[:n])
                tsne_wi = run_tsne(emb_wi[:n])
                tsne_data[cache_name] = {
                    'coords_wo': tsne_wo,
                    'coords_wi': tsne_wi,
                }

                del trainer
                torch.cuda.empty_cache()
                print(f'OK  (emb_wo={emb_wo.shape}, emb_wi={emb_wi.shape})')

            except Exception as e:
                print(f'ERROR: {e}')

        emb_path    = CACHE_DIR / f'case_{case_key}_embeddings.pkl'
        coords_path = CACHE_DIR / f'case_{case_key}_umap_coords.pkl'
        tsne_path   = CACHE_DIR / f'case_{case_key}_tsne_coords.pkl'

        with open(emb_path, 'wb') as f:
            pickle.dump(emb_data, f)
        with open(coords_path, 'wb') as f:
            pickle.dump(coords_data, f)
        with open(tsne_path, 'wb') as f:
            pickle.dump(tsne_data, f)

        print(f'  Saved → {emb_path.name}, {coords_path.name}, {tsne_path.name}')

    print('\nDone.')


if __name__ == '__main__':
    main()
