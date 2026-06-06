#!/usr/bin/env python3
"""
extract_case_predictions.py
Run inference with all 10 models on specific circRNA-miRNA pairs
and save position-level predictions to data_predictions.csv.

Usage:
    python extract_case_predictions.py [--device 0] [--seed 1] [--case all|HUWE1|RAD23B|FBXO7|NFIB]
"""

import argparse
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from trainer import Trainer
from data import CircRNABindingSitesDataset, KmerTokenizer
from utils import get_device
from utils_config import get_model_config

# ── Constants ──────────────────────────────────────────────────────────────────
MAX_LEN   = 1022
BSJ_WIN   = 40   # positions within ±BSJ_WIN nt of BSJ are flagged

# Per-model max sequence length constraints (positional embedding size)
MODEL_MAX_LEN = {
    'rnabert':  436,   # 440 positional embeddings - 2 (CLS/SEP)
    'rnaernie': 511,   # 513 positional embeddings - 2
}

MODELS_DIR = ROOT / 'models_for_viz'

# Model definitions: (col_name, model_name, exp_name, frozen_rna_lm)
MODEL_SPECS = [
    ('pred_circmac',          'circmac',     'v2_abl_full',                  False),
    ('pred_mamba',            'mamba',       'v2_enc_mamba',                 False),
    ('pred_lstm',             'lstm',        'v2_enc_lstm',                  False),
    ('pred_transformer',      'transformer', 'v2_enc_transformer',           False),
    ('pred_hymba',            'hymba',       'v2_enc_hymba',                 False),
    # Frozen pretrained models (backbone fixed, only downstream head trained)
    ('pred_rnabert_frozen',   'rnabert',     'exp1_fair_frozen_rnabert',     True),
    ('pred_rnaernie_frozen',  'rnaernie',    'exp1_fair_frozen_rnaernie',    True),
    ('pred_rnamsm_frozen',    'rnamsm',      'exp1_fair_frozen_rnamsm',      True),
    ('pred_rnafm_frozen',     'rnafm',       'exp1_fair_frozen_rnafm',       True),
    # Fine-tuned pretrained models (full fine-tuning)
    ('pred_rnabert_ft',       'rnabert',     'exp1_fair_trainable_rnabert',  False),
    ('pred_rnaernie_ft',      'rnaernie',    'exp1_fair_trainable_rnaernie', False),
    ('pred_rnamsm_ft',        'rnamsm',      'exp1_fair_trainable_rnamsm',   False),
    ('pred_rnafm_ft',         'rnafm',       'exp1_fair_trainable_rnafm',    False),
    ('pred_circmac_mlm',      'circmac',     'v2_pt_mlm',                    False),
]

# Case definitions: (case_name, output_dir, isoform_prefixes)
CASES = {
    # Pretrained model comparison cases (L<=436, circmac clearly best)
    # Case 1: chr4 circRNA, binding site in middle (L=261, mac=1.0 vs pretrained<=0.24)
    'PT_CHR4': (
        ROOT / 'figures_paper' / 'Fig_Case_PT_CHR4',
        ['chr4|82878730,82880799,82881858|'],
    ),
    # Case 2: chr17 circRNA, binding site near end (L=344, mac=1.0 vs pretrained<=0.33)
    'PT_CHR17': (
        ROOT / 'figures_paper' / 'Fig_Case_PT_CHR17',
        ['chr17|63193991,63200771|'],
    ),
    # Case 3: chr9 circRNA, binding site near start (L=377, mac=0.975 vs pretrained<=0.15)
    'PT_CHR9': (
        ROOT / 'figures_paper' / 'Fig_Case_PT_CHR9',
        ['chr9|128185547,128187863|'],
    ),
    # Main text cases (need re-extraction with trainable models)
    'CDYL2': (
        ROOT / 'figures_paper' / 'Fig_Case_CDYL2',
        ['chr4|84678168,84679116|'],
    ),
    'MAPK1': (
        ROOT / 'figures_paper' / 'Fig_Case_MAPK1',
        ['chr22|21799012,21805850,21807664|'],
    ),
    'APP': (
        ROOT / 'figures_paper' / 'Fig_Case_APP',
        ['chr21|25954590,25955627,25975070,25975954,25982344|'],
    ),
    # Encoder comparison cases (L<=436 for fairness, or pick specific isoforms)
    'HUWE1': (
        ROOT / 'figures_paper' / 'Fig_Case_HUWE1',
        ['chrX|53645'],
    ),
    'RAD23B': (
        ROOT / 'figures_paper' / 'Fig_Case_RAD23B',
        ['chr9|107302'],
    ),
    'FBXO7': (
        ROOT / 'figures_paper' / 'Fig_Case_FBXO7',
        ['chr22|32478'],
    ),
    'NFIB': (
        ROOT / 'figures_paper' / 'Fig_Case_NFIB',
        ['chr9|14120'],
    ),
    # Option A: same circRNA, 2 miRNAs at distant positions (chr3/PLXNA1)
    'MULTI_MIR': (
        ROOT / 'figures_paper' / 'Fig_Case_MULTI_MIR',
        ['chr3|122449574'],
    ),
    # Option B: binding site at BSJ end region (chr16/MLYCD)
    'BSJ': (
        ROOT / 'figures_paper' / 'Fig_Case_BSJ',
        ['chr16|10969161,10971125,10972938|'],
    ),
}


def bsj_adjacent_flag(length: int, bsj_win: int = BSJ_WIN) -> np.ndarray:
    """Return 1 for BSJ-adjacent positions (start or end region), 0 otherwise."""
    flags = np.zeros(length, dtype=int)
    flags[:bsj_win] = 1
    flags[max(0, length - bsj_win):] = 1
    return flags


def get_ckpt_vocab_size(model_name: str, exp_name: str, seed: int) -> int:
    """Peek at the checkpoint's embedding vocab size."""
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


def make_dataset_with_vocab(df: pd.DataFrame, max_len: int, ckpt_vocab_size: int):
    """Create CircRNABindingSitesDataset, forcing KmerTokenizer when ckpt uses vocab_size=11."""
    ds = CircRNABindingSitesDataset(
        df.reset_index(drop=True),
        max_len=max_len,
        target_type='mirna',
        k=1,
        k_target=1,
    )
    # If checkpoint was trained with KmerTokenizer(k=1) (vocab=11) but current env
    # uses RnaTokenizer (vocab=26), override the tokenizer to match.
    if ckpt_vocab_size == 11 and ds.vocab_size != 11:
        ds.circrna_tokenizer = KmerTokenizer(1)
        ds.vocab_size = 11
    return ds


def setup_trainer(model_name: str, exp_name: str, seed: int, device,
                  frozen_rna_lm: bool, vocab_size: int) -> Trainer:
    """Initialize and load a Trainer for inference."""
    exp_folder = f'{exp_name}_s{seed}'

    trainer = Trainer(
        seed=seed,
        device=device,
        dir_save=str(MODELS_DIR),
        experiment_name=exp_folder,
        verbose=False,
    )
    trainer.task = 'sites'
    trainer.rc = False
    trainer.use_unified_head = False
    trainer.interaction = 'cross_attention'

    config = get_model_config(
        model_name=model_name,
        d_model=128,
        n_layer=6,
        vocab_size=vocab_size,
    )

    # For frozen RNA LMs, trainable=False
    if model_name in ['rnabert', 'rnaernie', 'rnafm', 'rnamsm']:
        config.trainable = not frozen_rna_lm

    trainer.define_model(
        config=config,
        model_name=model_name,
        pretrain=False,
        is_cross_attention=True,
        interaction='cross_attention',
        site_head_type='conv1d',
    )

    # Load frozen RNA LM backbone for circRNA encoding
    if model_name in ['rnabert', 'rnaernie', 'rnafm', 'rnamsm'] and frozen_rna_lm:
        trainer.define_pretrained_model(model_name=model_name)

    # Load target model (rnabert for miRNA encoding)
    trainer.set_pretrained_target(target='mirna', rna_model='rnabert')

    # Load trained weights using partial matching to handle any key mismatches
    exp_folder_path = MODELS_DIR / model_name / exp_folder / str(seed) / 'train' / 'model.pth'
    if exp_folder_path.exists():
        trainer.load_model_from_path(str(exp_folder_path), verbose=False)
    else:
        trainer.load_model(epoch=None, pretrain=False, verbose=False)
    trainer.model.eval()

    return trainer


def run_inference_ds(trainer: Trainer, dataset, device) -> list:
    """Run inference on a pre-built dataset, return list of per-sample predictions."""
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    all_preds = []
    with torch.no_grad():
        for data in loader:
            target, target_mask = trainer.forward_target(data)
            emb, mask = trainer.forward(data)
            emb, _ = trainer.forward_cross_attention(emb, target, target_mask)
            logits = trainer.forward_task(emb, target, task='sites')  # [1, L, 2]
            prob = torch.softmax(logits, dim=-1)[..., 1]              # [1, L]
            L = int(data['length'][0].item())
            all_preds.append(prob[0, :L].cpu().numpy())

    return all_preds


def extract_case(case_name: str, out_dir: Path, prefixes: list,
                 df_test: pd.DataFrame, device, seed: int):
    print(f'\n=== {case_name} ===')

    # Filter pairs
    mask = pd.Series([False] * len(df_test))
    for pfx in prefixes:
        mask = mask | df_test['isoform_ID'].str.startswith(pfx)
    df_case = df_test[mask & (df_test['binding'] == 1)].reset_index(drop=True)
    print(f'  {len(df_case)} pairs found')

    if len(df_case) == 0:
        print('  No pairs — skipping')
        return

    # Build result rows per sample
    # First pass: collect metadata
    rows = []
    for _, row in df_case.iterrows():
        seq    = row['circRNA']
        sites  = np.array(row['sites'])
        L      = len(seq)
        bsj    = bsj_adjacent_flag(L)
        isoform_id = row['isoform_ID']
        mirna_id   = row['miRNA_ID']
        for pos in range(L):
            rows.append({
                'isoform_ID':    isoform_id,
                'miRNA_ID':      mirna_id,
                'length':        L,
                'position':      pos,
                'nucleotide':    seq[pos],
                'ground_truth':  int(sites[pos]),
                'bsj_adjacent':  int(bsj[pos]),
            })
    df_out = pd.DataFrame(rows)

    # Second pass: run each model
    for col_name, model_name, exp_name, frozen_rna_lm in MODEL_SPECS:
        print(f'  [{col_name}] loading model...', end=' ', flush=True)

        # Use model-specific max_len to respect positional embedding constraints
        model_max_len = MODEL_MAX_LEN.get(model_name, MAX_LEN) + 2

        # Skip pairs where any sequence exceeds the model's max length
        max_seq_len = df_case['circRNA'].apply(len).max()
        if max_seq_len > model_max_len - 2:
            print(f'SKIP (seq_len={max_seq_len} > model_max={model_max_len - 2})')
            continue

        try:
            ckpt_vocab = get_ckpt_vocab_size(model_name, exp_name, seed)
            tmp_ds = make_dataset_with_vocab(
                df_case.head(1), model_max_len, ckpt_vocab
            )
            vocab_size = tmp_ds.vocab_size

            trainer = setup_trainer(
                model_name=model_name,
                exp_name=exp_name,
                seed=seed,
                device=device,
                frozen_rna_lm=frozen_rna_lm,
                vocab_size=vocab_size,
            )
            # Build inference dataset with matching tokenizer
            inf_ds = make_dataset_with_vocab(df_case, model_max_len, ckpt_vocab)
            preds_list = run_inference_ds(trainer, inf_ds, device)
            # Flatten into the result df
            flat_preds = np.concatenate(preds_list)
            df_out[col_name] = np.round(flat_preds, 6)
            print(f'OK  (range [{flat_preds.min():.3f}, {flat_preds.max():.3f}])')
            # Free GPU memory
            del trainer
            torch.cuda.empty_cache()
        except Exception as e:
            print(f'ERROR: {e}')
            df_out[col_name] = None

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'data_predictions.csv'
    df_out.to_csv(out_path, index=False)
    print(f'  Saved → {out_path}  ({len(df_out)} rows)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed',   type=int, default=1)
    parser.add_argument('--case',   type=str, default='all',
                        help='all | HUWE1 | RAD23B | FBXO7 | NFIB')
    args = parser.parse_args()

    device = get_device(args.device)
    print(f'Device: {device}')

    df_test = pd.read_pickle(ROOT / 'data' / 'df_test_final.pkl')
    df_test['length'] = df_test['circRNA'].apply(len)

    # Merge miRNA sequences — required for cross-attention target encoding
    df_mirna = pd.read_csv(ROOT / 'data' / 'binding_miRNA_seq.csv')
    mirna_seq_map = df_mirna.groupby('miRNA_ID')['miRNA'].first().to_dict()
    df_test['miRNA'] = df_test['miRNA_ID'].map(mirna_seq_map)
    missing = df_test['miRNA'].isna().sum()
    if missing > 0:
        print(f'Warning: {missing} rows have no miRNA sequence (will use empty target)')
        df_test['miRNA'] = df_test['miRNA'].fillna('')

    cases_to_run = CASES if args.case == 'all' else {args.case: CASES[args.case]}

    for case_name, (out_dir, prefixes) in cases_to_run.items():
        extract_case(
            case_name=case_name,
            out_dir=out_dir,
            prefixes=prefixes,
            df_test=df_test,
            device=device,
            seed=args.seed,
        )

    print('\nDone.')


if __name__ == '__main__':
    main()
