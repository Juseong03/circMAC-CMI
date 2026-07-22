#!/usr/bin/env python3
"""
logs2csv.py  —  logs/ training.json → figure CSV files

Reads final test scores from logs/{model}/{exp}_s{seed}/{seed}/training.json
and writes per-figure data CSVs used by fig1~fig5 and fig_disjoint scripts.

Usage:
    python scripts/logs2csv.py                        # pair split (fig1-5)
    python scripts/logs2csv.py --logs_dir logs        # explicit dir
    python scripts/logs2csv.py --disjoint             # also build iso/bsj CSV
    python scripts/logs2csv.py --only disjoint        # only disjoint
"""

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT    = Path(__file__).resolve().parents[1]
METRICS = ['f1_macro', 'roc_auc', 'auprc', 'f1_pos', 'prec_pos', 'rec_pos', 'mcc']
SEEDS   = [1, 2, 3]


def load_scores(model_name: str, exp_name: str, seed: int, logs_dir: Path):
    """Load final test scores from training.json. Returns dict or None."""
    path = logs_dir / model_name / f'{exp_name}_s{seed}' / str(seed) / 'training.json'
    if not path.exists():
        print(f'  [MISS] {path.relative_to(ROOT)}')
        return None
    try:
        d = json.load(open(path))
        final = d.get('final', {})
        if not final:
            print(f'  [EMPTY final] {path.relative_to(ROOT)}')
            return None
        ep     = list(final.keys())[0]
        scores = final[ep]['scores'].get('sites', {})
        if not scores:
            print(f'  [NO sites] {path.relative_to(ROOT)}')
            return None
        out = {}
        for m in METRICS:
            if m in scores:
                out[m] = scores[m]
        # alias: roc_auc → auroc
        if 'auroc' in scores and 'roc_auc' not in out:
            out['roc_auc'] = scores['auroc']
        return out
    except Exception as e:
        print(f'  [ERR] {path.relative_to(ROOT)}: {e}')
        return None


# ─────────────────────────────────────────────
# Fig 1: RNA-LM comparison (pair split)
# ─────────────────────────────────────────────
FIG1_EXPS = [
    ('RNABERT',  'frozen',     'rnabert',  'exp1_fair_frozen_rnabert'),
    ('RNABERT',  'fine-tuned', 'rnabert',  'exp1_fair_trainable_rnabert'),
    ('RNAErnie', 'frozen',     'rnaernie', 'exp1_fair_frozen_rnaernie'),
    ('RNAErnie', 'fine-tuned', 'rnaernie', 'exp1_fair_trainable_rnaernie'),
    ('RNAMSM',   'frozen',     'rnamsm',   'exp1_fair_frozen_rnamsm'),
    ('RNAMSM',   'fine-tuned', 'rnamsm',   'exp1_fair_trainable_rnamsm'),
    ('RNA-FM',   'frozen',     'rnafm',    'exp1_fair_frozen_rnafm'),
    ('RNA-FM',   'fine-tuned', 'rnafm',    'exp1_fair_trainable_rnafm'),
    ('CircMAC',  'proposed',   'circmac',  'v2_pt_pairing'),
]

# ─────────────────────────────────────────────
# Fig 2: Pretraining strategy (pair split)
# ─────────────────────────────────────────────
FIG2_EXPS = [
    ('No PT',       'baseline',    'circmac', 'v2_pt_nopt'),
    ('MLM',         'single',      'circmac', 'v2_pt_mlm'),
    ('NTP',         'single',      'circmac', 'v2_pt_ntp'),
    ('SSP',         'single',      'circmac', 'v2_pt_ssp'),
    ('Pairing',     'single',      'circmac', 'v2_pt_pairing'),
    ('MLM+NTP',     'combination', 'circmac', 'v2_pt_mlm_ntp'),
    ('MLM+SSP',     'combination', 'circmac', 'v2_pt_mlm_ssp'),
    ('MLM+Pairing', 'combination', 'circmac', 'v2_pt_mlm_pairing'),
    ('SSP+Pairing', 'combination', 'circmac', 'v2_pt_ssp_pairing'),
    ('MLM+NTP+SSP', 'combination', 'circmac', 'v2_pt_mlm_ntp_ssp'),
    ('All',         'combination', 'circmac', 'v2_pt_all'),
]

# ─────────────────────────────────────────────
# Fig 3: Encoder comparison (pair split)
# ─────────────────────────────────────────────
FIG3_EXPS = [
    ('LSTM',        'lstm',        'v2_enc_lstm'),
    ('Transformer', 'transformer', 'v2_enc_transformer'),
    ('Mamba',       'mamba',       'v2_enc_mamba'),
    ('Hymba',       'hymba',       'v2_enc_hymba'),
    ('CircMAC',     'circmac',     'v2_abl_full'),
]

# ─────────────────────────────────────────────
# Fig 4: Module ablation (pair split)
# ─────────────────────────────────────────────
FIG4_EXPS = [
    ('CircMAC',      'full',   'circmac', 'v2_abl_full'),
    ('w/o Attn',     'remove', 'circmac', 'v2_abl_no_attn'),
    ('w/o Conv',     'remove', 'circmac', 'v2_abl_no_conv'),
    ('w/o Mamba',    'remove', 'circmac', 'v2_abl_no_mamba'),
    ('w/o CircBias', 'remove', 'circmac', 'v2_abl_no_circ_bias'),
    ('Mamba Only',   'single', 'circmac', 'v2_abl_mamba_only'),
    ('CNN Only',     'single', 'circmac', 'v2_abl_cnn_only'),
    ('Attn Only',    'single', 'circmac', 'v2_abl_attn_only'),
]

# ─────────────────────────────────────────────
# Fig 5: Interaction & Head ablation (pair split)
# ─────────────────────────────────────────────
FIG5_INT_EXPS = [
    ('Cross-Attn',  'interaction', 'circmac', 'v2_int_cross_attn'),
    ('Concat',      'interaction', 'circmac', 'v2_int_concat'),
    ('Elementwise', 'interaction', 'circmac', 'v2_int_elementwise'),
]
FIG5_HEAD_EXPS = [
    ('Conv1D', 'head', 'circmac', 'v2_int_cross_attn'),
    ('Linear', 'head', 'circmac', 'v2_head_linear'),
]

# ─────────────────────────────────────────────
# Disjoint splits: iso / bsj
# ─────────────────────────────────────────────
DISJOINT_EXPS = [
    # (split, label, group, model_name, exp_prefix)
    ('iso', 'CircMAC (NoPT)',    'pretraining', 'circmac', 'iso_pt_nopt'),
    ('iso', 'CircMAC (MLM)',     'pretraining', 'circmac', 'iso_pt_mlm'),
    ('iso', 'CircMAC (SSP)',     'pretraining', 'circmac', 'iso_pt_ssp'),
    ('iso', 'CircMAC (Pairing)', 'pretraining', 'circmac', 'iso_pt_pairing'),
    ('iso', 'Hymba',             'encoder',     'hymba',       'iso_hymba'),
    ('iso', 'Mamba',             'encoder',     'mamba',       'iso_mamba'),
    ('iso', 'LSTM',              'encoder',     'lstm',        'iso_lstm'),
    ('iso', 'Transformer',       'encoder',     'transformer', 'iso_transformer'),
    ('iso', 'RNABERT (ft)',      'encoder',     'rnabert',     'iso_rnabert_ft'),
    ('iso', 'RNAErnie (ft)',     'encoder',     'rnaernie',    'iso_rnaernie_ft'),
    ('iso', 'RNAMSM (ft)',       'encoder',     'rnamsm',      'iso_rnamsm_ft'),
    ('iso', 'RNA-FM (ft)',       'encoder',     'rnafm',       'iso_rnafm_ft'),
    ('bsj', 'CircMAC (NoPT)',    'pretraining', 'circmac', 'bsj_pt_nopt'),
    ('bsj', 'CircMAC (MLM)',     'pretraining', 'circmac', 'bsj_pt_mlm'),
    ('bsj', 'CircMAC (SSP)',     'pretraining', 'circmac', 'bsj_pt_ssp'),
    ('bsj', 'CircMAC (Pairing)', 'pretraining', 'circmac', 'bsj_pt_pairing'),
    ('bsj', 'Hymba',             'encoder',     'hymba',       'bsj_hymba'),
    ('bsj', 'Mamba',             'encoder',     'mamba',       'bsj_mamba'),
    ('bsj', 'LSTM',              'encoder',     'lstm',        'bsj_lstm'),
    ('bsj', 'Transformer',       'encoder',     'transformer', 'bsj_transformer'),
    ('bsj', 'RNABERT (ft)',      'encoder',     'rnabert',     'bsj_rnabert_ft'),
    ('bsj', 'RNAErnie (ft)',     'encoder',     'rnaernie',    'bsj_rnaernie_ft'),
    ('bsj', 'RNAMSM (ft)',       'encoder',     'rnamsm',      'bsj_rnamsm_ft'),
    ('bsj', 'RNA-FM (ft)',       'encoder',     'rnafm',       'bsj_rnafm_ft'),
]

# ─────────────────────────────────────────────
# LM Length-stratified: iso / bsj  (Option C)
# sub438: RNABERT/RNAErnie/RNAMSM/RNA-FM + CircMAC (max_len=438)
# sub511: RNAErnie/RNAMSM/RNA-FM + CircMAC (max_len=511)
# max   : RNAMSM/RNA-FM + CircMAC — reuse existing iso/bsj experiments
# ─────────────────────────────────────────────
LM_LENGTH_DISJOINT_EXPS = [
    # (split, label, model_name, exp_prefix, length_group)
    #
    # sub438:
    #   RNABERT  → reuse existing iso/bsj_rnabert_ft  (native max=438, same result)
    #   RNAErnie → new sub438 experiment
    # sub511:
    #   RNAErnie → reuse existing iso/bsj_rnaernie_ft (native max=511, same result)
    #   RNAMSM/RNA-FM/CircMAC → new sub511 experiment
    # max:
    #   RNAMSM/RNA-FM/CircMAC → reuse existing iso/bsj experiments
    #
    # ── sub438 ISO ──
    ('iso', 'RNABERT',  'rnabert',  'iso_rnabert_ft',             '438'),  # reuse (native max=438)
    ('iso', 'RNAErnie', 'rnaernie', 'sub438_iso_rnaernie_ft',     '438'),
    ('iso', 'RNAMSM',   'rnamsm',   'sub438_iso_rnamsm_ft',       '438'),
    ('iso', 'RNA-FM',   'rnafm',    'sub438_iso_rnafm_ft',        '438'),
    ('iso', 'CircMAC',  'circmac',  'sub438_iso_circmac_pairing', '438'),
    # ── sub511 ISO ──
    ('iso', 'RNAErnie', 'rnaernie', 'iso_rnaernie_ft',            '511'),  # reuse (native max=511)
    ('iso', 'RNAMSM',   'rnamsm',   'sub511_iso_rnamsm_ft',       '511'),
    ('iso', 'RNA-FM',   'rnafm',    'sub511_iso_rnafm_ft',        '511'),
    ('iso', 'CircMAC',  'circmac',  'sub511_iso_circmac_pairing', '511'),
    # ── max ISO (reuse existing) ──
    ('iso', 'RNAMSM',   'rnamsm',   'iso_rnamsm_ft',              'max'),
    ('iso', 'RNA-FM',   'rnafm',    'iso_rnafm_ft',               'max'),
    ('iso', 'CircMAC',  'circmac',  'iso_pt_pairing',             'max'),
    # ── sub438 BSJ ──
    ('bsj', 'RNABERT',  'rnabert',  'bsj_rnabert_ft',             '438'),  # reuse (native max=438)
    ('bsj', 'RNAErnie', 'rnaernie', 'sub438_bsj_rnaernie_ft',     '438'),
    ('bsj', 'RNAMSM',   'rnamsm',   'sub438_bsj_rnamsm_ft',       '438'),
    ('bsj', 'RNA-FM',   'rnafm',    'sub438_bsj_rnafm_ft',        '438'),
    ('bsj', 'CircMAC',  'circmac',  'sub438_bsj_circmac_pairing', '438'),
    # ── sub511 BSJ ──
    ('bsj', 'RNAErnie', 'rnaernie', 'bsj_rnaernie_ft',            '511'),  # reuse (native max=511)
    ('bsj', 'RNAMSM',   'rnamsm',   'sub511_bsj_rnamsm_ft',       '511'),
    ('bsj', 'RNA-FM',   'rnafm',    'sub511_bsj_rnafm_ft',        '511'),
    ('bsj', 'CircMAC',  'circmac',  'sub511_bsj_circmac_pairing', '511'),
    # ── max BSJ (reuse existing) ──
    ('bsj', 'RNAMSM',   'rnamsm',   'bsj_rnamsm_ft',              'max'),
    ('bsj', 'RNA-FM',   'rnafm',    'bsj_rnafm_ft',               'max'),
    ('bsj', 'CircMAC',  'circmac',  'bsj_pt_pairing',             'max'),
]


# ── builders ──────────────────────────────────────────────────────────────────

def build_fig1(logs_dir, seeds):
    rows = []
    for label, mode, model_name, exp in FIG1_EXPS:
        for seed in seeds:
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


def build_disjoint(logs_dir, seeds):
    """iso/bsj disjoint → disjoint_new_summary.csv 형식으로 반환."""
    raw_rows = []
    for split, label, group, model_name, exp_pfx in DISJOINT_EXPS:
        for seed in seeds:
            sc = load_scores(model_name, exp_pfx, seed, logs_dir)
            if sc is None:
                continue
            raw_rows.append({
                'split': split, 'label': label, 'group': group,
                'seed': seed, **sc,
            })

    if not raw_rows:
        return pd.DataFrame(), pd.DataFrame()

    df_raw = pd.DataFrame(raw_rows)

    # summary: mean ± std per (split, label)
    metric_cols = [c for c in ['f1_macro', 'roc_auc', 'auprc', 'f1_pos', 'prec_pos', 'rec_pos', 'mcc']
                   if c in df_raw.columns]
    summary_rows = []
    for (split, label), grp in df_raw.groupby(['split', 'label']):
        group_val = grp['group'].iloc[0]
        row = {'split': split, 'label': label, 'group': group_val,
               'n_seeds': len(grp)}
        for m in metric_cols:
            row[f'{m}_mean'] = round(grp[m].mean(), 4)
            row[f'{m}_std']  = round(grp[m].std(),  4)
        # alias roc_auc → auroc
        if 'roc_auc_mean' in row:
            row['auroc_mean'] = row['roc_auc_mean']
            row['auroc_std']  = row['roc_auc_std']
        summary_rows.append(row)

    df_sum = pd.DataFrame(summary_rows)
    return df_raw, df_sum


def build_lm_length_disjoint(logs_dir, seeds):
    """
    iso/bsj sub438/sub511/max experiments →
    figures_paper/fig_lm_length_disjoint/fig_lm_length_disjoint_data.csv
    (same format as fig_lm_length_comparison_data.csv)
    """
    raw_rows = []
    for split, label, model_name, exp_pfx, length_group in LM_LENGTH_DISJOINT_EXPS:
        for seed in seeds:
            sc = load_scores(model_name, exp_pfx, seed, logs_dir)
            if sc is None:
                continue
            raw_rows.append({
                'split': split, 'model': label, 'group': length_group,
                'seed': seed, **sc,
            })

    if not raw_rows:
        return pd.DataFrame()

    df_raw = pd.DataFrame(raw_rows)

    metric_cols = [c for c in ['f1_macro', 'roc_auc', 'auprc', 'f1_pos']
                   if c in df_raw.columns]
    summary_rows = []
    for (split, model, group), grp in df_raw.groupby(['split', 'model', 'group']):
        row = {'split': split, 'model': model, 'group': group, 'n_seeds': len(grp)}
        for m in metric_cols:
            row[f'{m}_mean'] = round(grp[m].mean(), 4)
            row[f'{m}_std']  = round(grp[m].std(),  4)
        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logs_dir', default='logs',
                        help='logs directory (default: logs)')
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    parser.add_argument('--disjoint', action='store_true',
                        help='also build iso/bsj disjoint CSVs')
    parser.add_argument('--only', choices=['pair', 'disjoint', 'lm_length', 'all'],
                        default='all')
    args = parser.parse_args()

    logs_dir = ROOT / args.logs_dir
    seeds    = args.seeds

    print(f'logs_dir : {logs_dir}')
    print(f'seeds    : {seeds}')

    do_pair      = args.only in ('pair', 'all')
    do_disjoint  = args.only in ('disjoint', 'all') or args.disjoint
    do_lm_length = args.only in ('lm_length', 'all')

    # ── Pair split figures ────────────────────────────────────────────────────
    if do_pair:
        OUT_FIG = {
            'fig1': ROOT / 'figures_paper/fig1_rna_lm/fig1_rna_lm_data.csv',
            'fig2': ROOT / 'figures_paper/fig2_pretraining/fig2_pretraining_data.csv',
            'fig3': ROOT / 'figures_paper/fig3_encoder/fig3_encoder_data.csv',
            'fig4': ROOT / 'figures_paper/fig4_ablation_modules/fig4_ablation_modules_data.csv',
            'fig5': ROOT / 'figures_paper/fig5_ablation_int_head/fig5_ablation_int_head_data.csv',
        }
        builders = {
            'fig1': build_fig1, 'fig2': build_fig2, 'fig3': build_fig3,
            'fig4': build_fig4, 'fig5': build_fig5,
        }
        for key, builder in builders.items():
            print(f'\n=== {key} ===')
            df = builder(logs_dir, seeds)
            if df.empty:
                print(f'  WARNING: no data')
                continue
            OUT_FIG[key].parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(OUT_FIG[key], index=False)
            grp_cols = [c for c in ['model', 'mode', 'ablation', 'group'] if c in df.columns]
            print(f'  Saved {len(df)} rows → {OUT_FIG[key].name}')
            print(df.groupby(grp_cols)[[c for c in METRICS if c in df.columns]].mean().round(4).to_string())

    # ── Disjoint splits ───────────────────────────────────────────────────────
    if do_disjoint:
        print(f'\n=== disjoint (iso/bsj) ===')
        df_raw, df_sum = build_disjoint(logs_dir, seeds)
        if df_raw.empty:
            print('  WARNING: no disjoint data found')
        else:
            out_raw = ROOT / 'eval_results/disjoint_new_raw.csv'
            out_sum = ROOT / 'eval_results/disjoint_new_summary.csv'
            out_raw.parent.mkdir(parents=True, exist_ok=True)
            df_raw.to_csv(out_raw, index=False)
            df_sum.to_csv(out_sum, index=False)
            print(f'  raw    : {len(df_raw)} rows → {out_raw.name}')
            print(f'  summary: {len(df_sum)} rows → {out_sum.name}')

            def fmt(mean_key, std_key, r):
                m = r.get(mean_key, float('nan'))
                s = r.get(std_key,  float('nan'))
                if np.isnan(m): return '   —   '
                return f'{m:.4f}±{s:.4f}'

            for split in ['iso', 'bsj']:
                sub = df_sum[df_sum['split'] == split].sort_values('auroc_mean', ascending=False)
                print(f'\n  [{split.upper()}-DISJOINT]')
                print(f"  {'Model':<22}  {'AUROC':>14}  {'AUPRC':>14}  {'F1(pos)':>14}")
                print(f"  {'-'*68}")
                for _, r in sub.iterrows():
                    auroc_k = 'auroc_mean' if 'auroc_mean' in r else 'roc_auc_mean'
                    auroc_s = 'auroc_std'  if 'auroc_std'  in r else 'roc_auc_std'
                    print(f"  {r['label']:<22}  "
                          f"{fmt(auroc_k, auroc_s, r):>14}  "
                          f"{fmt('auprc_mean', 'auprc_std', r):>14}  "
                          f"{fmt('f1_pos_mean', 'f1_pos_std', r):>14}")

    # ── LM length-stratified disjoint ────────────────────────────────────────
    if do_lm_length:
        print(f'\n=== lm_length (iso/bsj sub438/sub511/max) ===')
        df_lm = build_lm_length_disjoint(logs_dir, seeds)
        if df_lm.empty:
            print('  WARNING: no data — run scripts/sub438_iso/, sub511_iso/, sub438_bsj/, sub511_bsj/ first')
        else:
            out_lm = ROOT / 'figures_paper/fig_lm_length_disjoint/fig_lm_length_disjoint_data.csv'
            out_lm.parent.mkdir(parents=True, exist_ok=True)
            df_lm.to_csv(out_lm, index=False)
            print(f'  Saved {len(df_lm)} rows → {out_lm.name}')

            def fmt_lm(mean_key, std_key, r):
                m = r.get(mean_key, float('nan'))
                s = r.get(std_key,  float('nan'))
                if np.isnan(m): return '   —   '
                return f'{m:.4f}±{s:.4f}'

            for split in ['iso', 'bsj']:
                sub = df_lm[df_lm['split'] == split]
                print(f'\n  [{split.upper()} — length-stratified]')
                print(f"  {'Model':<12}  {'Group':>5}  {'AUROC':>14}  {'AUPRC':>14}  {'F1(pos)':>14}")
                print(f"  {'-'*60}")
                for _, r in sub.sort_values(['group', 'model']).iterrows():
                    print(f"  {r['model']:<12}  {r['group']:>5}  "
                          f"{fmt_lm('roc_auc_mean', 'roc_auc_std', r):>14}  "
                          f"{fmt_lm('auprc_mean', 'auprc_std', r):>14}  "
                          f"{fmt_lm('f1_pos_mean', 'f1_pos_std', r):>14}")


if __name__ == '__main__':
    main()
