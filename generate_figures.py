#!/usr/bin/env python
"""
Paper Figures Generator for CircMAC
=====================================

Usage:
    python generate_figures.py                      # Generate all figures (PNG)
    python generate_figures.py --format jpg         # Generate all figures (JPG)
    python generate_figures.py --fig 1              # Generate specific figure
    python generate_figures.py --output my_figures  # Custom output directory
    python generate_figures.py --list               # List available figures

Examples:
    python generate_figures.py --format png --dpi 300
    python generate_figures.py --fig 1 2 3 --format pdf
    python generate_figures.py --fig all --format jpg --dpi 150
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import warnings
import re
warnings.filterwarnings('ignore')


# =============================================================================
# Configuration
# =============================================================================

MODEL_COLORS = {
    'lstm': '#1f77b4',
    'transformer': '#ff7f0e',
    'mamba': '#2ca02c',
    'hymba': '#d62728',
    'circmac': '#e377c2',
}

MODEL_NAMES = {
    'lstm': 'LSTM',
    'transformer': 'Transformer',
    'mamba': 'Mamba',
    'hymba': 'Hymba',
    'circmac': 'CircMAC',
}

TASK_NAMES = {
    'binding': 'Binding Prediction',
    'sites': 'Binding Site Prediction',
    'both': 'Joint Prediction',
}

AVAILABLE_FIGURES = {
    1: "Model Comparison Box Plots (3 tasks)",
    2: "All Tasks Bar Chart Comparison",
    3: "Performance Heatmap",
    4: "Training Curves",
    5: "Statistical Significance Table",
    6: "Binding Site Linear Plot (example)",
    7: "Binding Site Circular Plot (example)",
    8: "Cross-Attention Heatmap (example)",
    9: "Model-wise Site Prediction Comparison (example)",
}


# =============================================================================
# Data Loading
# =============================================================================

def load_exp1_results(log_dir: str = 'logs/exp1') -> pd.DataFrame:
    """Load all Exp1 results from log files."""
    results = []
    log_path = Path(log_dir)

    if not log_path.exists():
        print(f"Warning: Log directory '{log_dir}' not found")
        return pd.DataFrame()

    for log_file in log_path.glob('exp1_*.log'):
        match = re.match(r'exp1_(\w+)_(\w+)_s(\d+)', log_file.stem)
        if not match:
            continue

        model, task, seed = match.groups()

        with open(log_file, 'r') as f:
            content = f.read()

        if 'Training completed' not in content and 'Training Complete' not in content:
            continue

        best_match = re.search(r'Best epoch:\s*(\d+),\s*score:\s*([\d.]+)', content)
        if not best_match:
            continue

        best_epoch = int(best_match.group(1))
        best_score = float(best_match.group(2))

        results.append({
            'model': model,
            'task': task,
            'seed': int(seed),
            'best_epoch': best_epoch,
            'best_score': best_score,
        })

    return pd.DataFrame(results)


def load_training_curves(model: str, task: str, seed: int) -> Optional[Dict]:
    """Load training curves from JSON log."""
    json_path = Path(f'logs/{model}/exp1_{model}_{task}_s{seed}/{seed}/training.json')
    if not json_path.exists():
        return None
    with open(json_path, 'r') as f:
        return json.load(f)


# =============================================================================
# Figure 1: Box Plots
# =============================================================================

def fig1_boxplots(results: pd.DataFrame, output_dir: str, fmt: str, dpi: int):
    """Generate box plots for each task."""
    tasks = ['binding', 'sites', 'both']
    model_order = ['lstm', 'transformer', 'mamba', 'hymba', 'circmac']

    for task in tasks:
        df = results[results['task'] == task].copy()
        available_models = [m for m in model_order if m in df['model'].unique()]

        if not available_models:
            print(f"  Skipping {task}: no data")
            continue

        fig, ax = plt.subplots(figsize=(10, 6))

        positions = range(len(available_models))
        box_data = [df[df['model'] == m]['best_score'].values for m in available_models]

        bp = ax.boxplot(box_data, positions=positions, patch_artist=True, widths=0.6)

        for i, (patch, model) in enumerate(zip(bp['boxes'], available_models)):
            patch.set_facecolor(MODEL_COLORS.get(model, '#888888'))
            patch.set_alpha(0.7)

        # Add individual points
        for i, model in enumerate(available_models):
            y = df[df['model'] == model]['best_score'].values
            x = np.random.normal(i, 0.04, size=len(y))
            ax.scatter(x, y, alpha=0.8, color=MODEL_COLORS.get(model, '#888888'),
                       edgecolor='black', s=60, zorder=5)

        ax.set_xticks(positions)
        ax.set_xticklabels([MODEL_NAMES.get(m, m) for m in available_models], fontsize=12)
        ax.set_ylabel('AUROC' if task == 'binding' else 'F1 Score', fontsize=14)
        ax.set_title(f'{TASK_NAMES.get(task, task)}', fontsize=16, fontweight='bold')

        # Add mean annotations
        for i, model in enumerate(available_models):
            values = df[df['model'] == model]['best_score'].values
            if len(values) > 0:
                mean_val = np.mean(values)
                ax.annotate(f'{mean_val:.4f}', xy=(i, mean_val),
                           xytext=(0, 12), textcoords='offset points',
                           ha='center', fontsize=11, fontweight='bold')

        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        save_path = f'{output_dir}/fig1_{task}_boxplot.{fmt}'
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")


# =============================================================================
# Figure 2: All Tasks Comparison
# =============================================================================

def fig2_all_tasks(results: pd.DataFrame, output_dir: str, fmt: str, dpi: int):
    """Generate side-by-side comparison for all tasks."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    tasks = ['binding', 'sites', 'both']
    model_order = ['lstm', 'transformer', 'mamba', 'hymba', 'circmac']

    for ax, task in zip(axes, tasks):
        df = results[results['task'] == task].copy()
        available_models = [m for m in model_order if m in df['model'].unique()]

        if not available_models:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
            ax.set_title(TASK_NAMES.get(task, task), fontsize=14, fontweight='bold')
            continue

        means = [df[df['model'] == m]['best_score'].mean() for m in available_models]
        stds = [df[df['model'] == m]['best_score'].std() for m in available_models]
        colors = [MODEL_COLORS.get(m, '#888888') for m in available_models]

        bars = ax.bar(range(len(available_models)), means, yerr=stds,
                      color=colors, alpha=0.7, capsize=5, edgecolor='black', linewidth=1.5)

        ax.set_xticks(range(len(available_models)))
        ax.set_xticklabels([MODEL_NAMES.get(m, m) for m in available_models],
                          rotation=45, ha='right', fontsize=11)
        ax.set_ylabel('AUROC' if task == 'binding' else 'F1 Score', fontsize=12)
        ax.set_title(TASK_NAMES.get(task, task), fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Highlight best
        if means:
            best_idx = np.argmax(means)
            bars[best_idx].set_edgecolor('red')
            bars[best_idx].set_linewidth(3)

    plt.tight_layout()
    save_path = f'{output_dir}/fig2_all_tasks.{fmt}'
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# =============================================================================
# Figure 3: Performance Heatmap
# =============================================================================

def fig3_heatmap(results: pd.DataFrame, output_dir: str, fmt: str, dpi: int):
    """Generate performance heatmap."""
    models = ['lstm', 'transformer', 'mamba', 'hymba', 'circmac']
    tasks = ['binding', 'sites', 'both']

    perf_matrix = np.zeros((len(models), len(tasks)))
    std_matrix = np.zeros((len(models), len(tasks)))

    for i, model in enumerate(models):
        for j, task in enumerate(tasks):
            df = results[(results['model'] == model) & (results['task'] == task)]
            if not df.empty:
                perf_matrix[i, j] = df['best_score'].mean()
                std_matrix[i, j] = df['best_score'].std()
            else:
                perf_matrix[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(perf_matrix, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=1.0)

    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel('Score', rotation=-90, va="bottom", fontsize=12)

    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([TASK_NAMES.get(t, t) for t in tasks], fontsize=12)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([MODEL_NAMES.get(m, m) for m in models], fontsize=12)

    for i in range(len(models)):
        for j in range(len(tasks)):
            if not np.isnan(perf_matrix[i, j]):
                text = f'{perf_matrix[i, j]:.4f}\n±{std_matrix[i, j]:.4f}'
                color = 'white' if perf_matrix[i, j] < 0.75 else 'black'
                ax.text(j, i, text, ha='center', va='center', color=color, fontsize=11)
            else:
                ax.text(j, i, 'N/A', ha='center', va='center', color='gray', fontsize=11)

    ax.set_title('Model Performance Summary', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    save_path = f'{output_dir}/fig3_heatmap.{fmt}'
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# =============================================================================
# Figure 4: Training Curves
# =============================================================================

def fig4_training_curves(results: pd.DataFrame, output_dir: str, fmt: str, dpi: int):
    """Generate training curves for each model."""
    models = ['circmac', 'mamba', 'lstm', 'hymba']
    tasks = ['binding', 'sites']
    seeds = [1, 2, 3]

    for task in tasks:
        fig, ax = plt.subplots(figsize=(12, 6))

        for model in models:
            all_epochs = []
            all_scores = []

            for seed in seeds:
                data = load_training_curves(model, task, seed)
                if data is None:
                    continue

                valid_data = data.get('valid', {})
                for epoch_str, metrics in valid_data.items():
                    epoch = int(epoch_str)
                    if task == 'binding':
                        score = metrics.get('scores', {}).get('bind', {}).get('roc_auc', 0)
                    else:
                        score = metrics.get('scores', {}).get('sites', {}).get('f1_macro', 0)

                    if score > 0:
                        all_epochs.append(epoch)
                        all_scores.append(score)

            if all_epochs:
                # Group by epoch and compute mean
                df_temp = pd.DataFrame({'epoch': all_epochs, 'score': all_scores})
                df_mean = df_temp.groupby('epoch')['score'].agg(['mean', 'std']).reset_index()

                ax.plot(df_mean['epoch'], df_mean['mean'],
                       label=MODEL_NAMES.get(model, model),
                       color=MODEL_COLORS.get(model, '#888888'),
                       linewidth=2)
                ax.fill_between(df_mean['epoch'],
                               df_mean['mean'] - df_mean['std'],
                               df_mean['mean'] + df_mean['std'],
                               alpha=0.2, color=MODEL_COLORS.get(model, '#888888'))

        ax.set_xlabel('Epoch', fontsize=14)
        ax.set_ylabel('AUROC' if task == 'binding' else 'F1 Score', fontsize=14)
        ax.set_title(f'Training Progress - {TASK_NAMES.get(task, task)}',
                    fontsize=16, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = f'{output_dir}/fig4_training_{task}.{fmt}'
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")


# =============================================================================
# Figure 5: Statistical Tests Table
# =============================================================================

def fig5_stats_table(results: pd.DataFrame, output_dir: str, fmt: str, dpi: int):
    """Generate statistical significance table as figure."""
    from scipy import stats

    for task in ['binding', 'sites', 'both']:
        df = results[results['task'] == task]
        models = sorted(df['model'].unique())

        if len(models) < 2:
            continue

        # Compute p-value matrix
        n = len(models)
        p_matrix = np.ones((n, n))
        diff_matrix = np.zeros((n, n))

        for i, m1 in enumerate(models):
            for j, m2 in enumerate(models):
                if i >= j:
                    continue
                scores1 = df[df['model'] == m1]['best_score'].values
                scores2 = df[df['model'] == m2]['best_score'].values

                if len(scores1) >= 2 and len(scores2) >= 2:
                    _, p_value = stats.ttest_ind(scores1, scores2)
                    p_matrix[i, j] = p_value
                    p_matrix[j, i] = p_value
                    diff_matrix[i, j] = np.mean(scores1) - np.mean(scores2)
                    diff_matrix[j, i] = -diff_matrix[i, j]

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create heatmap of differences
        mask = np.triu(np.ones_like(diff_matrix, dtype=bool), k=0)

        sns.heatmap(diff_matrix, mask=mask, cmap='RdBu_r', center=0,
                   annot=True, fmt='.4f', ax=ax,
                   xticklabels=[MODEL_NAMES.get(m, m) for m in models],
                   yticklabels=[MODEL_NAMES.get(m, m) for m in models],
                   cbar_kws={'label': 'Score Difference'})

        # Add significance markers
        for i in range(n):
            for j in range(i+1, n):
                p = p_matrix[i, j]
                if p < 0.001:
                    marker = '***'
                elif p < 0.01:
                    marker = '**'
                elif p < 0.05:
                    marker = '*'
                else:
                    marker = ''

                if marker:
                    ax.text(j + 0.5, i + 0.7, marker, ha='center', va='center',
                           fontsize=12, fontweight='bold', color='black')

        ax.set_title(f'Pairwise Comparisons - {TASK_NAMES.get(task, task)}\n'
                    '(* p<0.05, ** p<0.01, *** p<0.001)',
                    fontsize=14, fontweight='bold')

        plt.tight_layout()
        save_path = f'{output_dir}/fig5_stats_{task}.{fmt}'
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")


# =============================================================================
# Figure 6: Binding Site Linear Plot (Example)
# =============================================================================

def fig6_binding_linear(output_dir: str, fmt: str, dpi: int):
    """Generate example binding site linear plot."""
    # Generate example data
    np.random.seed(42)
    seq_len = 500

    # Simulate binding probabilities with some peaks
    probs = np.random.uniform(0.1, 0.3, seq_len)
    # Add binding site regions
    probs[100:130] = np.random.uniform(0.7, 0.9, 30)
    probs[250:290] = np.random.uniform(0.6, 0.85, 40)
    probs[400:420] = np.random.uniform(0.75, 0.95, 20)

    # True labels
    labels = np.zeros(seq_len)
    labels[105:125] = 1
    labels[260:285] = 1
    labels[405:418] = 1

    fig, ax = plt.subplots(figsize=(14, 5))

    # Plot probability curve
    ax.plot(probs, color='dodgerblue', linewidth=1.5, label='Predicted probability', alpha=0.8)

    # Highlight predicted regions (threshold = 0.5)
    pred_binary = (probs >= 0.5).astype(int)

    # Find spans
    def get_spans(binary):
        spans = []
        start = None
        for i, v in enumerate(binary):
            if v == 1 and start is None:
                start = i
            elif v == 0 and start is not None:
                spans.append((start, i))
                start = None
        if start is not None:
            spans.append((start, len(binary)))
        return spans

    pred_spans = get_spans(pred_binary)
    true_spans = get_spans(labels.astype(int))

    for i, (s, e) in enumerate(pred_spans):
        ax.axvspan(s, e, color='skyblue', alpha=0.4,
                   label='Predicted region' if i == 0 else '')

    for i, (s, e) in enumerate(true_spans):
        ax.axvspan(s, e, color='orange', alpha=0.3,
                   label='True binding site' if i == 0 else '')

    ax.axhline(y=0.5, linestyle='--', color='gray', alpha=0.5, label='Threshold')

    ax.set_xlabel('Sequence Position', fontsize=14)
    ax.set_ylabel('Binding Probability', fontsize=14)
    ax.set_title('Binding Site Prediction Example', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, seq_len)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    save_path = f'{output_dir}/fig6_binding_linear.{fmt}'
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# =============================================================================
# Figure 7: Binding Site Circular Plot (Example)
# =============================================================================

def fig7_binding_circular(output_dir: str, fmt: str, dpi: int):
    """Generate example binding site circular plot."""
    np.random.seed(42)
    seq_len = 300

    # Simulate data
    probs = np.random.uniform(0.1, 0.4, seq_len)
    probs[30:50] = np.random.uniform(0.7, 0.9, 20)
    probs[120:145] = np.random.uniform(0.65, 0.85, 25)
    probs[220:240] = np.random.uniform(0.7, 0.95, 20)

    labels = np.zeros(seq_len)
    labels[35:48] = 1
    labels[125:140] = 1
    labels[225:238] = 1

    threshold = 0.5

    # Create circular plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})

    angles = np.linspace(0, 2 * np.pi, seq_len, endpoint=False)

    # Color by threshold
    colors = ['#ff6b6b' if p >= threshold else '#4ecdc4' for p in probs]

    bars = ax.bar(angles, probs, width=2*np.pi/seq_len, bottom=0.1,
                  color=colors, alpha=0.7, edgecolor='none')

    # Mark true binding sites
    true_sites = np.where(labels == 1)[0]
    ax.scatter(angles[true_sites], np.ones(len(true_sites)) * 1.15,
               c='orange', s=30, marker='o', zorder=5, edgecolor='black', linewidth=0.5)

    # Backbone circle
    theta_circle = np.linspace(0, 2 * np.pi, 100)
    ax.plot(theta_circle, np.ones(100) * 0.1, 'k-', linewidth=3, alpha=0.5)

    ax.set_ylim(0, 1.3)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    # Position labels
    ax.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False))
    positions = np.linspace(0, seq_len, 8, endpoint=False).astype(int)
    ax.set_xticklabels([f'{p}' for p in positions], fontsize=10)

    ax.set_title('Circular RNA Binding Sites\n', fontsize=16, fontweight='bold', y=1.08)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#ff6b6b', alpha=0.7, label=f'Predicted (≥{threshold})'),
        Patch(facecolor='#4ecdc4', alpha=0.7, label=f'Predicted (<{threshold})'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange',
                   markersize=10, markeredgecolor='black', label='True binding site')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=11)

    plt.tight_layout()
    save_path = f'{output_dir}/fig7_binding_circular.{fmt}'
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# =============================================================================
# Figure 8: Cross-Attention Heatmap (Example)
# =============================================================================

def fig8_attention_heatmap(output_dir: str, fmt: str, dpi: int):
    """Generate example cross-attention heatmap."""
    np.random.seed(42)
    L_circ, L_mir = 60, 22  # typical lengths

    # Simulate attention: miRNA seed region (2-8) gets most attention
    attn = np.random.uniform(0.01, 0.05, (L_circ, L_mir))
    # Binding site regions attend strongly to miRNA seed (positions 2-8)
    attn[15:25, 2:8] = np.random.uniform(0.15, 0.35, (10, 6))
    attn[40:50, 2:8] = np.random.uniform(0.10, 0.25, (10, 6))
    # Normalize per circRNA position
    attn = attn / attn.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(12, 8))

    sns.heatmap(
        attn, ax=ax, cmap='YlOrRd',
        xticklabels=[f'{i}' for i in range(L_mir)],
        yticklabels=[f'{i}' if i % 5 == 0 else '' for i in range(L_circ)],
        cbar_kws={'label': 'Attention Weight'},
        linewidths=0.1,
    )

    # Highlight miRNA seed region
    ax.axvspan(2, 8, color='blue', alpha=0.08)
    ax.text(5, -2, 'Seed Region', ha='center', fontsize=10, color='blue', fontweight='bold')

    # Highlight binding site regions on circRNA
    for (start, end) in [(15, 25), (40, 50)]:
        rect = plt.Rectangle((0, start), L_mir, end - start,
                              linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
        ax.add_patch(rect)

    ax.set_xlabel('miRNA Position', fontsize=14)
    ax.set_ylabel('circRNA Position', fontsize=14)
    ax.set_title('Cross-Attention: circRNA ← miRNA\n(Binding sites attend to miRNA seed region)',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    save_path = f'{output_dir}/fig8_attention_heatmap.{fmt}'
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# =============================================================================
# Figure 8-2: Attention + Sites Combined View (Example)
# =============================================================================

def fig8_attention_with_sites(output_dir: str, fmt: str, dpi: int):
    """Generate combined attention heatmap with site predictions."""
    np.random.seed(42)
    L_circ, L_mir = 80, 22

    # Simulate attention
    attn = np.random.uniform(0.02, 0.05, (L_circ, L_mir))
    attn[20:35, 2:8] = np.random.uniform(0.12, 0.30, (15, 6))
    attn[55:68, 2:8] = np.random.uniform(0.10, 0.25, (13, 6))
    attn = attn / attn.sum(axis=1, keepdims=True)

    # Simulate site predictions and labels
    site_probs = np.random.uniform(0.05, 0.25, L_circ)
    site_probs[22:33] = np.random.uniform(0.65, 0.90, 11)
    site_probs[57:66] = np.random.uniform(0.55, 0.80, 9)

    site_labels = np.zeros(L_circ)
    site_labels[23:32] = 1
    site_labels[58:65] = 1

    fig, axes = plt.subplots(1, 3, figsize=(14, 8),
                              gridspec_kw={'width_ratios': [1, 10, 1], 'wspace': 0.05})

    # Left: True labels
    ax_true = axes[0]
    ax_true.imshow(site_labels.reshape(-1, 1), aspect='auto', cmap='Oranges', vmin=0, vmax=1)
    ax_true.set_xticks([0])
    ax_true.set_xticklabels(['True'], fontsize=10)
    ax_true.set_yticks(range(0, L_circ, 10))
    ax_true.set_ylabel('circRNA Position', fontsize=12)

    # Center: Attention heatmap
    ax_attn = axes[1]
    sns.heatmap(attn, ax=ax_attn, cmap='YlOrRd',
                xticklabels=[f'{i}' for i in range(L_mir)],
                yticklabels=False,
                cbar_kws={'label': 'Attention Weight', 'shrink': 0.8})
    ax_attn.set_xlabel('miRNA Position', fontsize=12)
    ax_attn.set_title('Cross-Attention with Binding Sites', fontsize=14, fontweight='bold')

    # Right: Predicted probabilities
    ax_pred = axes[2]
    ax_pred.imshow(site_probs.reshape(-1, 1), aspect='auto', cmap='Blues', vmin=0, vmax=1)
    ax_pred.set_xticks([0])
    ax_pred.set_xticklabels(['Pred'], fontsize=10)
    ax_pred.set_yticks([])

    plt.tight_layout()
    save_path = f'{output_dir}/fig8_attention_with_sites.{fmt}'
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# =============================================================================
# Figure 9: Model-wise Site Prediction Comparison (Example)
# =============================================================================

def fig9_model_comparison(output_dir: str, fmt: str, dpi: int):
    """Generate example model-wise site prediction comparison."""
    np.random.seed(42)
    seq_len = 200

    # True labels
    labels = np.zeros(seq_len)
    labels[40:65] = 1
    labels[130:155] = 1

    # Simulate predictions from different models
    def make_preds(base_noise, site_boost, spread):
        probs = np.random.uniform(0.05, base_noise, seq_len)
        for s, e in [(40, 65), (130, 155)]:
            region = slice(max(0, s - spread), min(seq_len, e + spread))
            probs[region] = np.clip(
                np.random.uniform(site_boost - 0.15, site_boost, region.stop - region.start),
                0, 1)
        return probs

    model_preds = {
        'lstm':        make_preds(0.30, 0.55, 8),   # noisy, weak
        'transformer': make_preds(0.25, 0.65, 5),   # moderate
        'mamba':       make_preds(0.22, 0.70, 4),   # good
        'hymba':       make_preds(0.20, 0.72, 3),   # better
        'circmac':     make_preds(0.15, 0.85, 1),   # best, sharp peaks
    }

    n_models = len(model_preds)
    fig, axes = plt.subplots(n_models + 1, 1, figsize=(14, 2 + n_models * 2), sharex=True)

    # True labels on top
    ax = axes[0]
    ax.fill_between(range(seq_len), 0, labels, color='orange', alpha=0.5, step='mid')
    ax.set_ylabel('True', fontsize=11, fontweight='bold')
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 1])
    ax.grid(True, alpha=0.2)

    # Model predictions
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#e377c2']
    threshold = 0.5

    for i, (name, probs) in enumerate(model_preds.items()):
        ax = axes[i + 1]
        color = colors[i % len(colors)]

        ax.plot(probs, color=color, linewidth=1.5, alpha=0.8)
        ax.fill_between(range(seq_len), 0, probs,
                         where=probs >= threshold, color=color, alpha=0.3)
        ax.axhline(y=threshold, linestyle='--', color='gray', alpha=0.4, linewidth=0.8)

        # True regions as background
        for s, e in [(40, 65), (130, 155)]:
            ax.axvspan(s, e, color='orange', alpha=0.1)

        display_name = MODEL_NAMES.get(name, name)
        ax.set_ylabel(display_name, fontsize=11, fontweight='bold')
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 0.5, 1.0])
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel('Sequence Position', fontsize=12)
    fig.suptitle('Binding Site Predictions by Model', fontsize=14, fontweight='bold', y=1.01)

    plt.tight_layout()
    save_path = f'{output_dir}/fig9_model_comparison.{fmt}'
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate paper figures for CircMAC',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python generate_figures.py                      # Generate all figures (PNG, 300 DPI)
    python generate_figures.py --format jpg         # Generate as JPG
    python generate_figures.py --fig 1 2 3          # Generate specific figures
    python generate_figures.py --dpi 600            # High resolution
    python generate_figures.py --list               # Show available figures
        """
    )

    parser.add_argument('--output', '-o', default='figures',
                        help='Output directory (default: figures)')
    parser.add_argument('--format', '-f', default='png', choices=['png', 'jpg', 'pdf', 'svg'],
                        help='Output format (default: png)')
    parser.add_argument('--dpi', '-d', type=int, default=300,
                        help='DPI for raster formats (default: 300)')
    parser.add_argument('--fig', '-n', nargs='+', type=int,
                        help='Specific figure numbers to generate (e.g., --fig 1 2 3)')
    parser.add_argument('--list', '-l', action='store_true',
                        help='List available figures')

    args = parser.parse_args()

    # List figures
    if args.list:
        print("\nAvailable Figures:")
        print("=" * 50)
        for num, desc in AVAILABLE_FIGURES.items():
            print(f"  {num}: {desc}")
        print("\nUsage: python generate_figures.py --fig 1 2 3")
        return

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Load data
    print("\n" + "=" * 60)
    print("  CircMAC Paper Figure Generator")
    print("=" * 60)
    print(f"Output: {args.output}/")
    print(f"Format: {args.format.upper()}")
    print(f"DPI: {args.dpi}")
    print("=" * 60 + "\n")

    print("Loading experiment results...")
    results = load_exp1_results()
    print(f"  Found {len(results)} completed experiments\n")

    # Determine which figures to generate
    if args.fig:
        fig_nums = args.fig
    else:
        fig_nums = list(AVAILABLE_FIGURES.keys())

    # Generate figures
    for num in fig_nums:
        if num not in AVAILABLE_FIGURES:
            print(f"[!] Figure {num} not found, skipping...")
            continue

        print(f"[Figure {num}] {AVAILABLE_FIGURES[num]}")

        if num == 1:
            fig1_boxplots(results, args.output, args.format, args.dpi)
        elif num == 2:
            fig2_all_tasks(results, args.output, args.format, args.dpi)
        elif num == 3:
            fig3_heatmap(results, args.output, args.format, args.dpi)
        elif num == 4:
            fig4_training_curves(results, args.output, args.format, args.dpi)
        elif num == 5:
            fig5_stats_table(results, args.output, args.format, args.dpi)
        elif num == 6:
            fig6_binding_linear(args.output, args.format, args.dpi)
        elif num == 7:
            fig7_binding_circular(args.output, args.format, args.dpi)
        elif num == 8:
            fig8_attention_heatmap(args.output, args.format, args.dpi)
            fig8_attention_with_sites(args.output, args.format, args.dpi)
        elif num == 9:
            fig9_model_comparison(args.output, args.format, args.dpi)

        print()

    print("=" * 60)
    print(f"✓ Done! Figures saved to '{args.output}/'")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
