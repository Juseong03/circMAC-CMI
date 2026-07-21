#!/usr/bin/env python3
"""
extract_multisite_predictions.py
Run inference on df_multisite.pkl (multi-site binding cases) and save
per-nucleotide predictions to figures_paper/fig_multisite/data_predictions.csv.

Usage:
    python scripts/extract_multisite_predictions.py --device 0 --seed 1
    python scripts/extract_multisite_predictions.py --device 0 --seed 1 --skip_existing
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trainer import Trainer
from data import CircRNABindingSitesDataset, KmerTokenizer
from utils import get_device
from utils_config import get_model_config

# ── Constants ──────────────────────────────────────────────────────────────────
MAX_LEN  = 1022
BSJ_WIN  = 40

MODEL_MAX_LEN = {
    'rnabert':  436,
    'rnaernie': 511,
}

MODELS_DIR = ROOT / 'models_for_viz'
OUT_DIR    = ROOT / 'figures_paper' / 'fig_multisite'
OUT_CSV    = OUT_DIR / 'data_predictions.csv'

# Models to run: (col_name, model_name, exp_name, frozen_rna_lm)
MODEL_SPECS = [
    ('pred_circmac',     'circmac',     'v2_abl_full',           False),
    ('pred_mamba',       'mamba',       'v2_enc_mamba',          False),
    ('pred_lstm',        'lstm',        'v2_enc_lstm',           False),
    ('pred_transformer', 'transformer', 'v2_enc_transformer',    False),
    ('pred_hymba',       'hymba',       'v2_enc_hymba',          False),
    ('pred_rnamsm_ft',   'rnamsm',      'exp1_fair_trainable_rnamsm', False),
    ('pred_rnafm_ft',    'rnafm',       'exp1_fair_trainable_rnafm',  False),
    ('pred_circmac_nopt','circmac',     'v2_pt_nopt',            False),
]


def bsj_adjacent_flag(length: int, bsj_win: int = BSJ_WIN) -> np.ndarray:
    flags = np.zeros(length, dtype=int)
    flags[:bsj_win] = 1
    flags[max(0, length - bsj_win):] = 1
    return flags


def get_ckpt_vocab_size(model_name: str, exp_name: str, seed: int):
    exp_folder = f'{exp_name}_s{seed}'
    ckpt_path = MODELS_DIR / model_name / exp_folder / str(seed) / 'train' / 'model.pth'
    if not ckpt_path.exists():
        return None
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        emb_key = 'embedding.word_embeddings.weight'
        if emb_key in ckpt:
            return ckpt[emb_key].shape[0]
    except Exception:
        pass
    return None


def make_dataset(df: pd.DataFrame, max_len: int, ckpt_vocab_size):
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


def run_inference(trainer, dataset, device):
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    all_preds = []
    with torch.no_grad():
        for data in loader:
            target, target_mask = trainer.forward_target(data)
            emb, mask = trainer.forward(data)
            emb, _ = trainer.forward_cross_attention(emb, target, target_mask)
            logits = trainer.forward_task(emb, target, task='sites')
            prob = torch.softmax(logits, dim=-1)[..., 1]
            L = int(data['length'][0].item())
            all_preds.append(prob[0, :L].cpu().numpy())
    return all_preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed',   type=int, default=1)
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip models whose column already exists in output CSV')
    args = parser.parse_args()

    device = get_device(args.device)
    print(f'Device: {device}')

    # ── Load multisite data ───────────────────────────────────────────────────
    df_multi = pd.read_pickle(ROOT / 'data' / 'df_multisite.pkl')
    print(f'Loaded df_multisite: {len(df_multi)} pairs')

    # Add miRNA sequences
    mirna_csv = ROOT / 'data' / 'binding_miRNA_seq.csv'
    if mirna_csv.exists():
        df_mirna = pd.read_csv(mirna_csv)
        mirna_seq_map = df_mirna.groupby('miRNA_ID')['miRNA'].first().to_dict()
        df_multi['miRNA'] = df_multi['miRNA_ID'].map(mirna_seq_map)
        missing = df_multi['miRNA'].isna().sum()
        if missing > 0:
            print(f'  Warning: {missing} rows missing miRNA sequence → using empty string')
            df_multi['miRNA'] = df_multi['miRNA'].fillna('')
    else:
        print(f'  Warning: {mirna_csv} not found — using empty miRNA sequences')
        df_multi['miRNA'] = ''

    # ── Build per-nucleotide output skeleton ──────────────────────────────────
    rows = []
    for _, row in df_multi.iterrows():
        seq   = row['circRNA']
        sites = np.array(row['sites'])
        L     = len(seq)
        bsj   = bsj_adjacent_flag(L)
        for pos in range(L):
            rows.append({
                'isoform_ID':   row['isoform_ID'],
                'miRNA_ID':     row['miRNA_ID'],
                'gene_name':    row.get('gene_name', ''),
                'length':       L,
                'n_binding_site': int(row['n_binding_site']),
                'ratio_binding_site': float(row['ratio_binding_site']),
                'position':     pos,
                'nucleotide':   seq[pos],
                'ground_truth': int(sites[pos]),
                'bsj_adjacent': int(bsj[pos]),
            })
    df_out = pd.DataFrame(rows)

    # ── Load existing columns if --skip_existing ──────────────────────────────
    existing_cols = set()
    if args.skip_existing and OUT_CSV.exists():
        df_existing = pd.read_csv(OUT_CSV)
        existing_cols = set(df_existing.columns)
        # Merge existing prediction columns
        for col in existing_cols:
            if col.startswith('pred_') and col not in df_out.columns:
                df_out[col] = df_existing[col].values
        print(f'  Loaded {len(existing_cols)} existing columns: {existing_cols & set(c for c in existing_cols if c.startswith("pred_"))}')

    # ── Run inference for each model ──────────────────────────────────────────
    for col_name, model_name, exp_name, frozen_rna_lm in MODEL_SPECS:
        if args.skip_existing and col_name in existing_cols:
            print(f'  [{col_name}] SKIP (already exists)')
            continue

        print(f'  [{col_name}] loading...', end=' ', flush=True)

        model_max_len = MODEL_MAX_LEN.get(model_name, MAX_LEN) + 2
        max_seq_len   = df_multi['circRNA'].apply(len).max()
        if max_seq_len > model_max_len - 2:
            print(f'SKIP (max_seq_len={max_seq_len} > model_max={model_max_len - 2})')
            df_out[col_name] = None
            continue

        try:
            ckpt_vocab = get_ckpt_vocab_size(model_name, exp_name, args.seed)
            tmp_ds = make_dataset(df_multi.head(1), model_max_len, ckpt_vocab)
            vocab_size = tmp_ds.vocab_size

            trainer = setup_trainer(model_name, exp_name, args.seed, device,
                                    frozen_rna_lm, vocab_size)
            inf_ds  = make_dataset(df_multi, model_max_len, ckpt_vocab)
            preds   = run_inference(trainer, inf_ds, device)
            flat    = np.concatenate(preds)
            df_out[col_name] = np.round(flat, 6)
            print(f'OK  (range [{flat.min():.3f}, {flat.max():.3f}])')
            del trainer
            torch.cuda.empty_cache()
        except Exception as e:
            print(f'ERROR: {e}')
            df_out[col_name] = None

    # ── Save ──────────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_CSV, index=False)
    print(f'\nSaved → {OUT_CSV}  ({len(df_out)} rows, {len(df_multi)} pairs)')


if __name__ == '__main__':
    main()
