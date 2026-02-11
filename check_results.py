#!/usr/bin/env python
"""
Experiment Results Checker

Usage:
    python check_results.py                    # Show all experiments
    python check_results.py --exp 1            # Show only Exp1
    python check_results.py --model circmac    # Filter by model
    python check_results.py --task binding     # Filter by task
    python check_results.py --running          # Show only running experiments
    python check_results.py --summary          # Show summary table
"""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from collections import defaultdict
from typing import Optional, Dict, List, Tuple

# ANSI colors
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

def get_running_experiments() -> Dict[str, str]:
    """Get currently running experiment names and their GPU."""
    try:
        result = subprocess.run(
            ['ps', 'aux'], capture_output=True, text=True, timeout=10
        )
        running = {}
        for line in result.stdout.split('\n'):
            if 'python' in line and 'training' in line and '--exp' in line:
                # Extract experiment name
                match = re.search(r'--exp\s+(\S+)', line)
                device_match = re.search(r'--device\s+(\d+)', line)
                if match:
                    exp_name = match.group(1)
                    device = device_match.group(1) if device_match else '?'
                    running[exp_name] = device
        return running
    except Exception:
        return {}

def parse_log_file(log_path: str) -> Dict:
    """Parse a log file to extract experiment status and metrics."""
    result = {
        'status': 'unknown',
        'current_epoch': 0,
        'best_epoch': 0,
        'best_score': 0.0,
        'patience': 0,
        'max_patience': 20,
        'test_scores': {},
    }

    try:
        with open(log_path, 'r') as f:
            content = f.read()

        # Check if completed
        if 'Training completed' in content or 'Training Complete' in content:
            result['status'] = 'completed'
        elif 'Early stopping' in content:
            result['status'] = 'completed'
        else:
            result['status'] = 'running'

        # Extract best epoch and score
        best_match = re.search(r'Best epoch:\s*(\d+),\s*score:\s*([\d.]+)', content)
        if best_match:
            result['best_epoch'] = int(best_match.group(1))
            result['best_score'] = float(best_match.group(2))

        # Extract current epoch from last "=== Epoch X/Y ==="
        epoch_matches = list(re.finditer(r'=== Epoch (\d+)/(\d+) ===', content))
        if epoch_matches:
            result['current_epoch'] = int(epoch_matches[-1].group(1))
            result['max_epochs'] = int(epoch_matches[-1].group(2))

        # Extract patience
        patience_matches = list(re.finditer(r'Patience:\s*(\d+)/(\d+)', content))
        if patience_matches:
            result['patience'] = int(patience_matches[-1].group(1))
            result['max_patience'] = int(patience_matches[-1].group(2))

        # Extract best score from "Best model updated" or "Best: X.XXXX"
        best_updated = list(re.finditer(r'Best model updated.*?score\s*([\d.]+)', content))
        if best_updated:
            result['best_score'] = float(best_updated[-1].group(1))

        best_at = list(re.finditer(r'Best:\s*([\d.]+)\s*at epoch\s*(\d+)', content))
        if best_at:
            result['best_score'] = float(best_at[-1].group(1))
            result['best_epoch'] = int(best_at[-1].group(2))

    except Exception as e:
        result['status'] = f'error: {e}'

    return result

def parse_json_log(json_path: str) -> Dict:
    """Parse JSON training log for detailed metrics."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        result = {
            'train_epochs': len(data.get('train', {})),
            'valid_epochs': len(data.get('valid', {})),
            'test': data.get('test', {}),
        }

        # Get best validation score
        valid_data = data.get('valid', {})
        if valid_data:
            best_epoch = None
            best_score = 0
            for epoch, metrics in valid_data.items():
                scores = metrics.get('scores', {})
                # Check binding AUROC
                bind_auroc = scores.get('bind', {}).get('roc_auc', 0)
                sites_f1 = scores.get('sites', {}).get('f1_macro', 0)
                score = max(bind_auroc, sites_f1)
                if score > best_score:
                    best_score = score
                    best_epoch = epoch
            result['best_valid_epoch'] = best_epoch
            result['best_valid_score'] = best_score

        # Get test scores
        if 'test' in data and data['test']:
            test_data = list(data['test'].values())[0] if isinstance(data['test'], dict) else data['test']
            if isinstance(test_data, dict):
                result['test_scores'] = test_data.get('scores', {})

        return result
    except Exception:
        return {}

def get_experiment_info(exp_name: str) -> Dict:
    """Extract model, task, seed from experiment name."""
    # Pattern: exp{N}_{model}_{task}_s{seed}
    match = re.match(r'exp(\d+)_(\w+)_(\w+)_s(\d+)', exp_name)
    if match:
        return {
            'exp_num': int(match.group(1)),
            'model': match.group(2),
            'task': match.group(3),
            'seed': int(match.group(4)),
        }
    return {'exp_num': 0, 'model': 'unknown', 'task': 'unknown', 'seed': 0}

def find_json_log(exp_name: str, model: str, seed: int) -> Optional[str]:
    """Find the JSON log file for an experiment."""
    possible_paths = [
        f'logs/{model}/{exp_name}/{seed}/training.json',
        f'logs/{model}/{exp_name}/training.json',
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

def print_experiment_status(experiments: List[Dict], running: Dict[str, str]):
    """Print formatted experiment status."""
    # Group by model
    by_model = defaultdict(list)
    for exp in experiments:
        by_model[exp['model']].append(exp)

    for model in sorted(by_model.keys()):
        print(f"\n{Colors.BOLD}{Colors.CYAN}=== {model.upper()} ==={Colors.END}")

        # Group by task
        by_task = defaultdict(list)
        for exp in by_model[model]:
            by_task[exp['task']].append(exp)

        for task in ['binding', 'sites', 'both']:
            if task not in by_task:
                continue

            print(f"\n  {Colors.BLUE}{task}:{Colors.END}")
            for exp in sorted(by_task[task], key=lambda x: x['seed']):
                name = exp['name']
                status = exp['status']

                # Status color
                if status == 'completed':
                    status_str = f"{Colors.GREEN}DONE{Colors.END}"
                elif name in running:
                    gpu = running[name]
                    status_str = f"{Colors.YELLOW}RUNNING (GPU {gpu}){Colors.END}"
                else:
                    status_str = f"{Colors.RED}INCOMPLETE{Colors.END}"

                # Score
                score_str = f"Best: {exp['best_score']:.4f}" if exp['best_score'] > 0 else ""
                epoch_str = f"Epoch {exp['current_epoch']}" if exp['current_epoch'] > 0 else ""

                print(f"    s{exp['seed']}: {status_str} {epoch_str} {score_str}")

def print_summary_table(experiments: List[Dict], running: Dict[str, str]):
    """Print a summary table with results."""
    print(f"\n{Colors.BOLD}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}EXPERIMENT 1 RESULTS SUMMARY{Colors.END}")
    print(f"{'='*80}")

    # Create summary by model and task
    models = ['lstm', 'transformer', 'mamba', 'hymba', 'circmac']
    tasks = ['binding', 'sites', 'both']

    # Header
    print(f"\n{'Model':<12} | {'Task':<8} | {'s1':^12} | {'s2':^12} | {'s3':^12} | {'Mean':^10}")
    print("-" * 75)

    for model in models:
        for task in tasks:
            scores = []
            cells = []
            for seed in [1, 2, 3]:
                exp_name = f"exp1_{model}_{task}_s{seed}"
                exp = next((e for e in experiments if e['name'] == exp_name), None)

                if exp and exp['status'] == 'completed' and exp['best_score'] > 0:
                    scores.append(exp['best_score'])
                    cells.append(f"{exp['best_score']:.4f}")
                elif exp and exp_name in running:
                    cells.append(f"{Colors.YELLOW}running{Colors.END}")
                elif exp:
                    cells.append(f"{Colors.RED}--{Colors.END}")
                else:
                    cells.append("--")

            mean_str = f"{sum(scores)/len(scores):.4f}" if scores else "--"
            print(f"{model:<12} | {task:<8} | {cells[0]:^12} | {cells[1]:^12} | {cells[2]:^12} | {mean_str:^10}")
        print("-" * 75)

def main():
    parser = argparse.ArgumentParser(description="Check experiment results")
    parser.add_argument('--exp', type=int, help='Experiment number (1, 2, 3, 4)')
    parser.add_argument('--model', type=str, help='Filter by model name')
    parser.add_argument('--task', type=str, help='Filter by task (binding, sites, both)')
    parser.add_argument('--running', action='store_true', help='Show only running experiments')
    parser.add_argument('--summary', action='store_true', help='Show summary table')
    parser.add_argument('--completed', action='store_true', help='Show only completed experiments')
    args = parser.parse_args()

    # Get running experiments
    running = get_running_experiments()

    # Find all log files
    log_dir = Path('logs/exp1')
    experiments = []

    if log_dir.exists():
        for log_file in log_dir.glob('*.log'):
            exp_name = log_file.stem
            info = get_experiment_info(exp_name)
            log_data = parse_log_file(str(log_file))

            exp = {
                'name': exp_name,
                **info,
                **log_data,
            }

            # Find and parse JSON log
            json_path = find_json_log(exp_name, info['model'], info['seed'])
            if json_path:
                json_data = parse_json_log(json_path)
                exp.update(json_data)

            experiments.append(exp)

    # Apply filters
    if args.exp:
        experiments = [e for e in experiments if e['exp_num'] == args.exp]
    if args.model:
        experiments = [e for e in experiments if args.model.lower() in e['model'].lower()]
    if args.task:
        experiments = [e for e in experiments if e['task'] == args.task]
    if args.running:
        experiments = [e for e in experiments if e['name'] in running]
    if args.completed:
        experiments = [e for e in experiments if e['status'] == 'completed']

    # Sort experiments
    experiments.sort(key=lambda x: (x['model'], x['task'], x['seed']))

    # Print results
    if args.summary:
        print_summary_table(experiments, running)
    else:
        # Print overall stats
        total = 54  # 6 models * 3 tasks * 3 seeds
        completed = sum(1 for e in experiments if e['status'] == 'completed')
        running_count = len(running)

        print(f"\n{Colors.BOLD}Experiment 1 Progress: {completed}/{total} completed")
        if running_count > 0:
            print(f"Currently running: {running_count} experiments{Colors.END}")
            for name, gpu in running.items():
                print(f"  - {name} on GPU {gpu}")
        print()

        print_experiment_status(experiments, running)

    print()

if __name__ == '__main__':
    main()
