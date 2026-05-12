#!/usr/bin/env python3
"""
Step 1: Extract embeddings for each case × model, save to disk.

Saved files:
  figures_claude/emb_cache/case_{key}_embeddings.pkl
    → {model_label: {'emb_wo': ndarray [L,D], 'emb_wi': ndarray or None, 'lbls': ndarray, 'seq_len': int}}

  figures_claude/emb_cache/case_{key}_{method}_coords.pkl
    → {model_label: {'coords_wo': ndarray [L,2], 'coords_wi': ndarray or None}}

Usage:
    python docs/paper_cmi/extract_case_embeddings.py            # all cases, both methods
    python docs/paper_cmi/extract_case_embeddings.py --case cdyl2
    python docs/paper_cmi/extract_case_embeddings.py --method umap
    python docs/paper_cmi/extract_case_embeddings.py --skip_reduce  # embeddings only
"""

import argparse, sys, pickle, torch
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data import CircRNABindingSitesDataset
from models.model import ModelWrapper
from utils_config import get_model_config

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

CASES = {
    'cdyl2': (
        'chr4|84678168,84679116|84678259,84679242|-',
        'hsa-miR-449a',
        'circCDYL2'
    ),
    'mapk1': (
        'chr22|21799012,21805850,21807664|21799128,21806039,21807846|-',
        'hsa-miR-12119',
        'circMAPK1'
    ),
    'app': (
        'chr21|25954590,25955627,25975070,25975954,25982344|25954689,25955755,25975228,25976028,25982477|-',
        'hsa-miR-5001-3p',
        'circAPP'
    ),
}

MODELS = [
    ('circmac',     'v2_pt_pairing_s1',                1, 'CircMAC',            'seq'),
    ('transformer', 'v2_enc_transformer_s1',            1, 'Transformer',        'seq'),
    ('mamba',       'v2_enc_mamba_s1',                  1, 'Mamba',              'seq'),
    ('lstm',        'v2_enc_lstm_s1',                   1, 'LSTM',               'seq'),
    ('hymba',       'v2_enc_hymba_s1',                  1, 'Hymba',              'seq'),
    ('rnabert',  'exp1_fair_frozen_rnabert_s1',         1, 'RNABert (frozen)',   'frozen_rna', 'rnabert',   440),
    ('rnabert',  'exp1_fair_trainable_rnabert_s1',      1, 'RNABert (train)',    'train_rna',  'rnabert',   440),
    ('rnaernie', 'exp1_fair_frozen_rnaernie_s1',        1, 'RNAErnie (frozen)',  'frozen_rna', 'rnaernie',  512),
    ('rnaernie', 'exp1_fair_trainable_rnaernie_s1',     1, 'RNAErnie (train)',   'train_rna',  'rnaernie',  512),
    ('rnafm',    'exp1_fair_frozen_rnafm_s1',           1, 'RNA-FM (frozen)',    'frozen_rna', 'rnafm',    1024),
    ('rnafm',    'exp1_fair_trainable_rnafm_s1',        1, 'RNA-FM (train)',     'train_rna',  'rnafm',    1024),
    ('rnamsm',   'exp1_fair_frozen_rnamsm_s1',          1, 'RNA-MSM (frozen)',   'frozen_rna', 'rnamsm',    512),
    ('rnamsm',   'exp1_fair_trainable_rnamsm_s1',       1, 'RNA-MSM (train)',    'train_rna',  'rnamsm',    512),
]

# ── RNA LM cache ──────────────────────────────────────────────────────────────
_rna_lm_cache = {}
def get_rna_lm(rna_name):
    if rna_name in _rna_lm_cache:
        return _rna_lm_cache[rna_name]
    from multimolecule.models import RnaBertModel, RnaFmModel, RnaErnieModel, RnaMsmModel
    _hub = {'rnabert': 'multimolecule/rnabert', 'rnafm': 'multimolecule/rnafm',
            'rnaernie': 'multimolecule/rnaernie', 'rnamsm': 'multimolecule/rnamsm'}
    _cls = {'rnabert': RnaBertModel, 'rnafm': RnaFmModel,
            'rnaernie': RnaErnieModel, 'rnamsm': RnaMsmModel}
    print(f"    Loading RNA LM: {rna_name}...")
    lm = _cls[rna_name].from_pretrained(_hub[rna_name]).to(DEVICE)
    lm.eval()
    for p in lm.parameters(): p.requires_grad_(False)
    _rna_lm_cache[rna_name] = lm
    return lm


# ── Model loader ──────────────────────────────────────────────────────────────
def load_model(cfg):
    model_name = cfg[0]; exp_name = cfg[1]; seed = cfg[2]
    mtype      = cfg[4]
    rna_name   = cfg[5] if len(cfg) > 5 else None
    max_pos    = cfg[6] if len(cfg) > 6 else 1022

    path = ROOT / 'saved_models' / model_name / exp_name / str(seed) / 'train' / 'model.pth'
    if not path.exists():
        return None, None, False, mtype, max_pos

    sd = torch.load(str(path), map_location=DEVICE, weights_only=False)
    has_ca = 'cross_attention.q_proj.weight' in sd

    if mtype == 'seq':
        config = get_model_config(model_name, d_model=128, n_layer=6,
                                  verbose=False, vocab_size=11)
        model = ModelWrapper(config=config, name=model_name, device=DEVICE).to(DEVICE)
        model_pt = None
    elif mtype == 'train_rna':
        config = get_model_config(model_name, d_model=128, n_layer=0, verbose=False)
        config.rc = False; config.trainable = True
        model = ModelWrapper(config=config, name=model_name, device=DEVICE).to(DEVICE)
        model_pt = None
    elif mtype == 'frozen_rna':
        model_pt = get_rna_lm(rna_name)
        config = get_model_config(model_name, d_model=128, n_layer=0, verbose=False)
        config.rc = False
        model = ModelWrapper(config=config, name=model_name, device=DEVICE).to(DEVICE)

    if 'proj_target.0.weight' in sd:
        model._set_proj_target(d_target=sd['proj_target.0.weight'].shape[1])
    if has_ca:
        model._set_cross_attention()
        model = model.to(DEVICE)

    ms   = model.state_dict()
    filt = {k: v for k, v in sd.items() if k in ms and v.shape == ms[k].shape}
    ms.update(filt); model.load_state_dict(ms)
    print(f"    Loaded {len(filt)}/{len(sd)} params  ca={'yes' if has_ca else 'no'}")
    model.eval()
    return model, model_pt, has_ca, mtype, max_pos


# ── Embedding extraction ──────────────────────────────────────────────────────
@torch.no_grad()
def extract_embeddings(model, model_pt, has_ca, mtype, max_pos,
                       circ_t, circ_m, tgt_t, tgt_m, seq_len, target_rnabert):
    if mtype == 'seq':
        x_emb = model.embedding(circ_t)
        h, _ = model.backbone(x_emb, circ_m, None, None)
    elif mtype == 'train_rna':
        h, _ = model.backbone(circ_t[:, :max_pos], circ_m[:, :max_pos], None, None)
    elif mtype == 'frozen_rna':
        out  = model_pt(input_ids=circ_t[:, :max_pos], attention_mask=circ_m[:, :max_pos])
        hs   = out['last_hidden_state']
        h, _ = model.backbone(hs, circ_m[:, :max_pos], None, None)

    actual_len = min(h.shape[1] - 1, seq_len)
    emb_wo = h[0, 1:1+actual_len].cpu().float().numpy()

    if not has_ca or target_rnabert is None:
        return emb_wo, None

    tgt_out  = target_rnabert(tgt_t, tgt_m)
    tgt_hs   = tgt_out['last_hidden_state']
    tgt_proj = model.get_target_projected(tgt_hs, mode='None')
    h_ca, _  = model.cross_attention(h, tgt_proj, tgt_proj, tgt_m)
    emb_wi   = h_ca[0, 1:1+actual_len].cpu().float().numpy()
    return emb_wo, emb_wi


# ── Dim reduction ─────────────────────────────────────────────────────────────
def run_umap(embs):
    import umap as _umap
    n = len(embs)
    embs = embs + np.random.default_rng(42).normal(0, 1e-6, embs.shape)
    return _umap.UMAP(n_components=2, random_state=42,
                      n_neighbors=min(15, max(5, n//10)),
                      min_dist=0.1, metric='cosine').fit_transform(embs)

def run_tsne(embs):
    from sklearn.manifold import TSNE
    n = len(embs)
    return TSNE(n_components=2, random_state=42,
                perplexity=min(30, max(5, n//10)),
                learning_rate='auto', init='pca', n_jobs=-1).fit_transform(embs)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case',   default='all', help='cdyl2|mapk1|app|all')
    parser.add_argument('--method', default='both', choices=['umap','tsne','both'])
    parser.add_argument('--skip_reduce', action='store_true',
                        help='Save embeddings only, skip UMAP/t-SNE reduction')
    args = parser.parse_args()

    CACHE = ROOT / 'figures_claude' / 'emb_cache'
    CACHE.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    test_df = pickle.load(open(ROOT / 'data/df_test_final.pkl', 'rb'))

    print("Loading rnabert target model...")
    from multimolecule.models import RnaBertModel
    target_rnabert = RnaBertModel.from_pretrained('multimolecule/rnabert').to(DEVICE)
    target_rnabert.eval()
    for p in target_rnabert.parameters(): p.requires_grad_(False)

    cases_to_run = CASES if args.case == 'all' else {args.case: CASES[args.case]}
    methods = ['umap','tsne'] if args.method == 'both' else [args.method]

    for case_key, (isoform_id, mirna_id, case_name) in cases_to_run.items():
        print(f"\n{'='*65}")
        print(f"  Case: {case_name}  |  {mirna_id}")
        print(f"{'='*65}")

        emb_cache_path = CACHE / f'case_{case_key}_embeddings.pkl'

        # ── Load or extract embeddings ────────────────────────────────────────
        if emb_cache_path.exists():
            print(f"  [CACHE] Loading embeddings from {emb_cache_path}")
            with open(emb_cache_path, 'rb') as f:
                emb_data = pickle.load(f)
        else:
            row = test_df[(test_df['isoform_ID'] == isoform_id) &
                          (test_df['miRNA_ID']   == mirna_id)]
            if row.empty:
                print(f"  [SKIP] Not found"); continue

            ds   = CircRNABindingSitesDataset(df=row.reset_index(drop=True),
                                              max_len=1022, k=1)
            item = ds[0]
            circ_t  = item['circRNA'].unsqueeze(0).to(DEVICE)
            circ_m  = item['circRNA_mask'].unsqueeze(0).to(DEVICE)
            tgt_t   = item['target'].unsqueeze(0).to(DEVICE)
            tgt_m   = item['target_mask'].unsqueeze(0).to(DEVICE)
            seq_len = int(item['length'].item())
            sites   = item['sites'].numpy()
            lbls_all = sites[1:1+seq_len]
            valid   = lbls_all != -100
            lbls    = lbls_all[valid].astype(int)
            print(f"  seq_len={seq_len}  n_binding={lbls.sum()}  n_valid={valid.sum()}")

            emb_data = {}  # label → dict
            for cfg in MODELS:
                label = cfg[3]
                print(f"\n  [{label}]")
                model, model_pt, has_ca, mtype, max_pos = load_model(cfg)
                if model is None:
                    print(f"    SKIP: not found"); continue

                ewo, ewi = extract_embeddings(
                    model, model_pt, has_ca, mtype, max_pos,
                    circ_t, circ_m, tgt_t, tgt_m, seq_len, target_rnabert)
                del model; torch.cuda.empty_cache()

                min_len  = min(len(ewo), len(valid))
                ewo_v    = ewo[:min_len][valid[:min_len]]
                ewi_v    = (ewi[:min_len][valid[:min_len]]
                            if ewi is not None else None)
                lbl_v    = lbls[:len(ewo_v)]
                pos_idx  = np.where(valid[:min_len])[0]  # original position indices

                emb_data[label] = {
                    'emb_wo':  ewo_v,
                    'emb_wi':  ewi_v,
                    'lbls':    lbl_v,
                    'pos_idx': pos_idx,
                    'seq_len': seq_len,
                }
                print(f"    emb_wo={ewo_v.shape}  "
                      f"emb_wi={'None' if ewi_v is None else ewi_v.shape}")

            with open(emb_cache_path, 'wb') as f:
                pickle.dump(emb_data, f)
            print(f"\n  [SAVED] embeddings → {emb_cache_path}")

        if args.skip_reduce:
            continue

        # ── Dim reduction ─────────────────────────────────────────────────────
        for method in methods:
            coord_path = CACHE / f'case_{case_key}_{method}_coords.pkl'
            if coord_path.exists():
                print(f"  [CACHE] {method} coords already exist: {coord_path}")
                continue

            fn = run_umap if method == 'umap' else run_tsne
            print(f"\n  Running {method.upper()}...")
            coords_data = {}
            for label, d in emb_data.items():
                print(f"    [{label}] w/o miRNA ({method})...")
                cwo = fn(d['emb_wo'])
                cwi = fn(d['emb_wi']) if d['emb_wi'] is not None else None
                if cwi is not None:
                    print(f"    [{label}] w/ miRNA ({method})...")
                coords_data[label] = {'coords_wo': cwo, 'coords_wi': cwi}

            with open(coord_path, 'wb') as f:
                pickle.dump(coords_data, f)
            print(f"  [SAVED] {method} coords → {coord_path}")

    print("\nAll done!")


if __name__ == '__main__':
    main()
