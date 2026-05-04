#!/usr/bin/env python3
"""
v2 실험 결과 확인 스크립트

사용법:
    python check_results.py                    # 전체 요약
    python check_results.py --exp enc          # Encoder 비교만
    python check_results.py --exp abl          # Ablation만
    python check_results.py --exp int          # Interaction만
    python check_results.py --exp head         # Site Head만
    python check_results.py --exp rna          # RNA LM 비교만
    python check_results.py --exp pt           # Pretraining만
    python check_results.py --all              # 모든 실험 (old포함)
    python check_results.py --logs logs_0427   # 다른 logs 폴더 지정
"""

import argparse
import json
import subprocess
import re
from pathlib import Path
from collections import defaultdict

# ── 색상 ────────────────────────────────────────────────────────────────────
G  = '\033[92m'   # green
Y  = '\033[93m'   # yellow
R  = '\033[91m'   # red
B  = '\033[94m'   # blue
C  = '\033[96m'   # cyan
M  = '\033[95m'   # magenta
BO = '\033[1m'    # bold
EN = '\033[0m'    # end

# ── v2 실험 정의 ─────────────────────────────────────────────────────────────
V2_EXP_GROUPS = {
    'enc': {
        'title': 'EXP1: Encoder Architecture Comparison',
        'models': {
            'circmac':     [('v2_enc_circmac_s{s}',     'circmac')],
            'mamba':       [('v2_enc_mamba_s{s}',        'mamba')],
            'lstm':        [('v2_enc_lstm_s{s}',         'lstm')],
            'transformer': [('v2_enc_transformer_s{s}',  'transformer')],
            'hymba':       [('v2_enc_hymba_s{s}',        'hymba')],
        },
    },
    'rna': {
        'title': 'EXP1: RNA LM Comparison',
        'models': {
            'rnabert_frozen':     [('exp1_fair_frozen_rnabert_s{s}',    'rnabert')],
            'rnaernie_frozen':    [('exp1_fair_frozen_rnaernie_s{s}',   'rnaernie')],
            'rnafm_frozen':       [('exp1_fair_frozen_rnafm_s{s}',      'rnafm')],
            'rnamsm_frozen':      [('exp1_fair_frozen_rnamsm_s{s}',     'rnamsm')],
            'rnabert_trainable':  [('exp1_fair_trainable_rnabert_s{s}', 'rnabert')],
            'rnaernie_trainable': [('exp1_fair_trainable_rnaernie_s{s}','rnaernie')],
            'rnafm_trainable':    [('exp1_fair_trainable_rnafm_s{s}',   'rnafm')],
            'rnamsm_trainable':   [('exp1_fair_trainable_rnamsm_s{s}',  'rnamsm')],
        },
    },
    'abl': {
        'title': 'EXP4: CircMAC Ablation Study',
        'models': {
            'full':          [('v2_abl_full_s{s}',          'circmac')],
            'no_attn':       [('v2_abl_no_attn_s{s}',       'circmac')],
            'no_mamba':      [('v2_abl_no_mamba_s{s}',      'circmac')],
            'no_conv':       [('v2_abl_no_conv_s{s}',       'circmac')],
            'no_circ_bias':  [('v2_abl_no_circ_bias_s{s}',  'circmac')],
            'attn_only':     [('v2_abl_attn_only_s{s}',     'circmac')],
            'mamba_only':    [('v2_abl_mamba_only_s{s}',    'circmac')],
            'cnn_only':      [('v2_abl_cnn_only_s{s}',      'circmac')],
        },
    },
    'int': {
        'title': 'EXP5: Interaction Mechanism',
        'models': {
            'cross_attention': [('v2_int_cross_attention_s{s}', 'circmac')],
            'concat':          [('v2_int_concat_s{s}',          'circmac')],
            'elementwise':     [('v2_int_elementwise_s{s}',     'circmac')],
        },
    },
    'head': {
        'title': 'EXP6: Site Prediction Head',
        'models': {
            'conv1d': [('v2_head_conv1d_s{s}', 'circmac')],
            'linear': [('v2_head_linear_s{s}', 'circmac')],
        },
    },
    'pt': {
        'title': 'EXP2: Pretraining Strategy',
        'models': {
            'nopt':    [('v2_pt_nopt_s{s}',    'circmac')],
            'mlm':     [('v2_pt_mlm_s{s}',     'circmac')],
            'ssp':     [('v2_pt_ssp_s{s}',     'circmac')],
            'pairing': [('v2_pt_pairing_s{s}', 'circmac')],
            'cpcl':    [('v2_pt_cpcl_s{s}',    'circmac')],
            'bsj':     [('v2_pt_bsj_s{s}',     'circmac')],
            'all':     [('v2_pt_all_s{s}',      'circmac')],
        },
    },
}

SEEDS = [1, 2, 3]


# ── JSON 파싱 ────────────────────────────────────────────────────────────────
def get_f1(json_path: Path) -> float:
    """training.json → best epoch F1 (sites f1_macro 우선, 없으면 bind roc_auc)"""
    try:
        data = json.loads(json_path.read_text())
        final = data.get('final', {})
        if not final:
            return None
        ep_data = list(final.values())[0]
        scores = ep_data.get('scores', {})
        f1 = scores.get('sites', {}).get('f1_macro')
        if f1 is None:
            f1 = scores.get('bind', {}).get('roc_auc')
        return round(float(f1), 4) if f1 is not None else None
    except Exception:
        return None


def find_json(exp_name: str, model: str, logs_root: Path) -> Path | None:
    """logs/{model}/{exp_name}/{seed}/training.json 탐색"""
    for seed in SEEDS:
        p = logs_root / model / exp_name / str(seed) / 'training.json'
        if p.exists():
            return p
    # seed 디렉토리 없이 바로 있는 경우
    p = logs_root / model / exp_name / 'training.json'
    if p.exists():
        return p
    return None


def get_status_and_score(exp_name: str, model: str, seed: int, logs_root: Path):
    """
    Returns (status, f1_score)
    status: 'done' | 'running' | 'pending'
    """
    json_path = logs_root / model / exp_name / str(seed) / 'training.json'
    if json_path.exists():
        f1 = get_f1(json_path)
        return ('done', f1)

    # log 파일이라도 있으면 running
    log_candidates = list((logs_root / model).glob(f'{exp_name}/**/*.log')) if \
        (logs_root / model).exists() else []
    if log_candidates:
        return ('running', None)

    return ('pending', None)


# ── running 프로세스 감지 ────────────────────────────────────────────────────
def get_running() -> set:
    try:
        out = subprocess.run(['ps', 'aux'], capture_output=True, text=True, timeout=10).stdout
        running = set()
        for line in out.split('\n'):
            if 'python' in line and '--exp' in line:
                m = re.search(r'--exp\s+(\S+)', line)
                if m:
                    running.add(m.group(1))
        return running
    except Exception:
        return set()


# ── 테이블 출력 ──────────────────────────────────────────────────────────────
def fmt_cell(status, f1, exp_name, running_set):
    is_running = exp_name in running_set
    if status == 'done' and f1 is not None:
        return f'{G}{f1:.4f}{EN}'
    elif status == 'done' and f1 is None:
        return f'{Y}done?{EN}'
    elif is_running or status == 'running':
        return f'{Y}run...{EN}'
    else:
        return f'{R}--{EN}'


def print_group(group_key: str, logs_root: Path, running_set: set):
    group = V2_EXP_GROUPS[group_key]
    print(f'\n{BO}{C}{"="*65}{EN}')
    print(f'{BO}{C}  {group["title"]}{EN}')
    print(f'{BO}{C}{"="*65}{EN}')
    print(f'  {"Variant":<22} | {"s1":^10} | {"s2":^10} | {"s3":^10} | {"Mean":^10} | {"Std":^8}')
    print(f'  {"-"*22}-+-{"-"*10}-+-{"-"*10}-+-{"-"*10}-+-{"-"*10}-+-{"-"*8}')

    for variant, entries in group['models'].items():
        exp_template, model = entries[0]
        scores = []
        cells  = []
        for seed in SEEDS:
            exp_name = exp_template.format(s=seed)
            status, f1 = get_status_and_score(exp_name, model, seed, logs_root)
            cells.append(fmt_cell(status, f1, exp_name, running_set))
            if f1 is not None:
                scores.append(f1)

        if len(scores) >= 2:
            import numpy as np
            mean_str = f'{G}{np.mean(scores):.4f}{EN}'
            std_str  = f'{np.std(scores):.4f}'
        elif len(scores) == 1:
            mean_str = f'{Y}{scores[0]:.4f}{EN}'
            std_str  = '--'
        else:
            mean_str = f'{R}--{EN}'
            std_str  = '--'

        print(f'  {variant:<22} | {cells[0]:^10} | {cells[1]:^10} | {cells[2]:^10} | {mean_str:^10} | {std_str:^8}')

    print()


def print_all_summary(logs_root: Path, running_set: set, exp_filter: str = None):
    import numpy as np
    groups = [exp_filter] if exp_filter else list(V2_EXP_GROUPS.keys())
    for g in groups:
        if g in V2_EXP_GROUPS:
            print_group(g, logs_root, running_set)
        else:
            print(f'{R}Unknown experiment type: {g}{EN}')
            print(f'Available: {list(V2_EXP_GROUPS.keys())}')


def scan_all_logs(logs_root: Path):
    """logs 폴더의 모든 training.json을 스캔 → seed 묶어서 출력"""
    import numpy as np
    print(f'\n{BO}{C}{"="*75}{EN}')
    print(f'{BO}{C}  All experiments in {logs_root}{EN}')
    print(f'{BO}{C}{"="*75}{EN}')
    print(f'  {"Experiment":<42} | {"s1":^8} | {"s2":^8} | {"s3":^8} | {"Mean":^8} | {"Std":^6}')
    print(f'  {"-"*42}-+-{"-"*8}-+-{"-"*8}-+-{"-"*8}-+-{"-"*8}-+-{"-"*6}')

    # exp_base → {seed: f1}
    results = defaultdict(dict)
    for json_path in sorted(logs_root.rglob('training.json')):
        parts = json_path.relative_to(logs_root).parts
        if len(parts) < 3:
            continue
        model, exp_name = parts[0], parts[1]
        seed_str = parts[2] if parts[2].isdigit() else None
        if seed_str is None:
            continue
        # exp_base: _s{seed} 접미사 제거
        base = re.sub(r'_s\d+$', '', exp_name)
        f1 = get_f1(json_path)
        results[base][int(seed_str)] = f1

    for base in sorted(results.keys()):
        sd = results[base]
        vals = [sd[s] for s in SEEDS if s in sd and sd[s] is not None]
        cells = []
        for s in SEEDS:
            v = sd.get(s)
            if v is not None:
                cells.append(f'{G}{v:.4f}{EN}')
            elif s in sd:
                cells.append(f'{Y}done?{EN}')
            else:
                cells.append(f'{R}--{EN}')

        if len(vals) >= 2:
            mean, std = np.mean(vals), np.std(vals)
            color = G if mean >= 0.74 else (Y if mean >= 0.70 else R)
            mean_s = f'{color}{mean:.4f}{EN}'
            std_s  = f'{std:.4f}'
        elif len(vals) == 1:
            mean_s = f'{Y}{vals[0]:.4f}{EN}'
            std_s  = '--'
        else:
            mean_s = f'{R}--{EN}'
            std_s  = '--'

        print(f'  {base:<42} | {cells[0]:^8} | {cells[1]:^8} | {cells[2]:^8} | {mean_s:^8} | {std_s:^6}')


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='v2 실험 결과 확인')
    parser.add_argument('--exp', type=str, default=None,
                        choices=list(V2_EXP_GROUPS.keys()),
                        help='enc | abl | int | head | rna | pt')
    parser.add_argument('--logs', type=str, default='logs',
                        help='logs 폴더 경로 (default: logs)')
    parser.add_argument('--scan', action='store_true',
                        help='logs 폴더 전체 스캔 (모든 training.json)')
    args = parser.parse_args()

    logs_root = Path(args.logs)
    if not logs_root.exists():
        print(f'{R}logs 폴더 없음: {logs_root}{EN}')
        return

    running_set = get_running()
    if running_set:
        print(f'\n{Y}{BO}현재 실행 중: {", ".join(sorted(running_set))}{EN}')

    if args.scan:
        scan_all_logs(logs_root)
    else:
        print_all_summary(logs_root, running_set, args.exp)


if __name__ == '__main__':
    main()
