#!/usr/bin/env python3
"""
logs2csv.py  —  logs_0512 training.json → figure CSV files

Reads final test scores from logs_0512/{model}/{exp}_s{seed}/{seed}/training.json
and writes per-figure data CSVs used by fig1~fig5 plotting scripts.

Usage:
    python3.10 scripts/logs2csv.py [--logs_dir logs_0512]
"""

import json, argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

METRICS = ['f1_macro', 'roc_auc', 'auprc']


def load_scores(model_name: str, exp_name: str, seed: int, logs_dir: Path):
    """Load final test scores from training.json. Returns dict or None."""
    path = logs_dir / model_name / f'{exp_name}_s{seed}' / str(seed) / 'training.json'
    if not path.exists():
        print(f'  [MISS] {path}')
        return None
    try:
        d = json.load(open(path))
        final = d.get('final', {})
        if not final:
            print(f'  [EMPTY final] {path}')
            return None
        ep = list(final.keys())[0]
        scores = final[ep]['scores'].get('sites', {})
        if not scores:
            print(f'  [NO sites] {path}')
            return None
        return {m: scores[m] for m in METRICS if m in scores}
    except Exception as e:
        print(f'  [ERR] {path}: {e}')
        return None


def make_rows(model_label, extra_cols, model_name, exp_name, seeds, logs_dir):
    """Return list of row dicts for given experiment across seeds."""
    rows = []
    for seed in seeds:
        sc = load_scores(model_name, exp_name, seed, logs_dir)
        if sc is None:
            continue
        row = {'seed': seed, **extra_cols}
        row.update(sc)
        rows.append(row)
    return rows


# ─────────────────────────────────────────────
# Fig 1: RNA-LM comparison
# ─────────────────────────────────────────────
FIG1_EXPS = [
    # (display_label, mode, model_name, exp_prefix)
    ('RNABERT',  'frozen',      'rnabert',  'exp1_fair_frozen_rnabert'),
    ('RNABERT',  'fine-tuned',  'rnabert',  'exp1_fair_trainable_rnabert'),
    ('RNAErnie', 'frozen',      'rnaernie', 'exp1_fair_frozen_rnaernie'),
    ('RNAErnie', 'fine-tuned',  'rnaernie', 'exp1_fair_trainable_rnaernie'),
    ('RNAMSM',   'frozen',      'rnamsm',   'exp1_fair_frozen_rnamsm'),
    ('RNAMSM',   'fine-tuned',  'rnamsm',   'exp1_fair_trainable_rnamsm'),
    ('RNA-FM',   'frozen',      'rnafm',    'exp1_fair_frozen_rnafm'),
    ('RNA-FM',   'fine-tuned',  'rnafm',    'exp1_fair_trainable_rnafm'),
    ('CircMAC',  'proposed',    'circmac',  'v2_pt_pairing'),
]

# ─────────────────────────────────────────────
# Fig 2: Pretraining strategy
# ─────────────────────────────────────────────
FIG2_EXPS = [
    ('No PT',   'baseline', 'circmac', 'v2_pt_nopt'),
    ('MLM',     'single',   'circmac', 'v2_pt_mlm'),
    ('NTP',     'single',   'circmac', 'v2_pt_ntp'),
    ('SSP',     'single',   'circmac', 'v2_pt_ssp'),
    ('Pairing', 'single',   'circmac', 'v2_pt_pairing'),
]

# ─────────────────────────────────────────────
# Fig 3: Encoder comparison
# ─────────────────────────────────────────────
FIG3_EXPS = [
    ('LSTM',        'lstm',        'v2_enc_lstm'),
    ('Transformer', 'transformer', 'v2_enc_transformer'),
    ('Mamba',       'mamba',       'v2_enc_mamba'),
    ('Hymba',       'hymba',       'v2_enc_hymba'),
    ('CircMAC',     'circmac',     'v2_enc_circmac'),
]

# ─────────────────────────────────────────────
# Fig 4: Module ablation
# ─────────────────────────────────────────────
FIG4_EXPS = [
    ('CircMAC',    'full',   'circmac', 'v2_abl_full'),
    ('w/o Attn',   'remove', 'circmac', 'v2_abl_no_attn'),
    ('w/o Conv',   'remove', 'circmac', 'v2_abl_no_conv'),
    ('w/o Mamba',  'remove', 'circmac', 'v2_abl_no_mamba'),
    ('Mamba Only', 'single', 'circmac', 'v2_abl_mamba_only'),
    ('CNN Only',   'single', 'circmac', 'v2_abl_cnn_only'),
    ('Attn Only',  'single', 'circmac', 'v2_abl_attn_only'),
]

# ─────────────────────────────────────────────
# Fig 5: Interaction & Head ablation
# ─────────────────────────────────────────────
FIG5_INT_EXPS = [
    ('Cross-Attn',  'interaction', 'circmac', 'v2_int_cross_attn'),
    ('Concat',      'interaction', 'circmac', 'v2_int_concat'),
    ('Elementwise', 'interaction', 'circmac', 'v2_int_elementwise'),
]
FIG5_HEAD_EXPS = [
    ('Conv1D', 'head', 'circmac', 'v2_head_conv1d'),
    ('Linear', 'head', 'circmac', 'v2_head_linear'),
]


def build_fig1(logs_dir, seeds):
    rows = []
    for label, mode, model_name, exp in FIG1_EXPS:
        s_list = seeds if model_name == 'circmac' else seeds
        for seed in s_list:
            sc = load_scores(model_name, exp, seed, logs_dir)
            if sc is None:
                continue
            rows.append({'model': label, 'mode': mode, 'seed': seed, **sc})
    return pd.DataFrame(rows)


def build_fig2(logs_dir, seeds):
    rows = []
    for label, group, model_name, exp in FIG2_EXPS:
        for seed in seeds:
            sc = load_scores(model_name, exp, seed, logs_dir)
            if sc is None:
                continue
            rows.append({'model': label, 'group': group, 'seed': seed, **sc})
    return pd.DataFrame(rows)


def build_fig3(logs_dir, seeds):
    rows = []
    for label, model_name, exp in FIG3_EXPS:
        for seed in seeds:
            sc = load_scores(model_name, exp, seed, logs_dir)
            if sc is None:
                continue
            rows.append({'model': label, 'seed': seed, **sc})
    return pd.DataFrame(rows)


def build_fig4(logs_dir, seeds):
    rows = []
    for label, group, model_name, exp in FIG4_EXPS:
        for seed in seeds:
            sc = load_scores(model_name, exp, seed, logs_dir)
            if sc is None:
                continue
            rows.append({'model': label, 'group': group, 'seed': seed, **sc})
    return pd.DataFrame(rows)


def build_fig5(logs_dir, seeds):
    rows = []
    for label, ablation, model_name, exp in FIG5_INT_EXPS + FIG5_HEAD_EXPS:
        for seed in seeds:
            sc = load_scores(model_name, exp, seed, logs_dir)
            if sc is None:
                continue
            rows.append({'model': label, 'ablation': ablation, 'seed': seed, **sc})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logs_dir', default='logs_0512')
    parser.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3])
    args = parser.parse_args()

    logs_dir = ROOT / args.logs_dir
    seeds = args.seeds

    OUT = {
        'fig1': ROOT / 'figures_paper/fig1_rna_lm/fig1_rna_lm_data.csv',
        'fig2': ROOT / 'figures_paper/fig2_pretraining/fig2_pretraining_data.csv',
        'fig3': ROOT / 'figures_paper/fig3_encoder/fig3_encoder_data.csv',
        'fig4': ROOT / 'figures_paper/fig4_ablation_modules/fig4_ablation_modules_data.csv',
        'fig5': ROOT / 'figures_paper/fig5_ablation_int_head/fig5_ablation_int_head_data.csv',
    }

    builders = {
        'fig1': build_fig1,
        'fig2': build_fig2,
        'fig3': build_fig3,
        'fig4': build_fig4,
        'fig5': build_fig5,
    }

    for key, builder in builders.items():
        print(f'\n=== {key} ===')
        df = builder(logs_dir, seeds)
        if df.empty:
            print(f'  WARNING: no data for {key}')
            continue
        out_path = OUT[key]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f'  Saved {len(df)} rows → {out_path}')
        print(df.groupby([c for c in ['model','mode','ablation','group'] if c in df.columns])[METRICS].mean().round(4).to_string())


if __name__ == '__main__':
    main()
