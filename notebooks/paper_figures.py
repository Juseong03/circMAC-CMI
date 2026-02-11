"""
Paper Figures Generator for CircMAC
=====================================
Visualization code for generating publication-ready figures.

Usage:
    from paper_figures import *

    # Load experiment results
    results = load_exp1_results()

    # Generate figures
    plot_model_comparison_boxplot(results)
    plot_roc_curves(results)
    plot_binding_site_circular(df_results, sample_idx=10)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Style settings for publication
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Color palette for models
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


# =============================================================================
# Data Loading
# =============================================================================

def load_exp1_results(log_dir: str = 'logs/exp1') -> pd.DataFrame:
    """Load all Exp1 results from log files."""
    import re

    results = []
    log_path = Path(log_dir)

    for log_file in log_path.glob('exp1_*.log'):
        # Parse experiment name
        match = re.match(r'exp1_(\w+)_(\w+)_s(\d+)', log_file.stem)
        if not match:
            continue

        model, task, seed = match.groups()

        # Read log file
        with open(log_file, 'r') as f:
            content = f.read()

        # Check if completed
        if 'Training completed' not in content and 'Training Complete' not in content:
            continue

        # Extract best score
        best_match = re.search(r'Best epoch:\s*(\d+),\s*score:\s*([\d.]+)', content)
        if not best_match:
            continue

        best_epoch = int(best_match.group(1))
        best_score = float(best_match.group(2))

        # Try to get test metrics from JSON logs
        json_path = Path(f'logs/{model}/exp1_{model}_{task}_s{seed}/{seed}/training.json')
        test_metrics = {}

        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                if 'test' in data and data['test']:
                    test_data = list(data['test'].values())[0]
                    if 'scores' in test_data:
                        test_metrics = test_data['scores']
            except:
                pass

        results.append({
            'model': model,
            'task': task,
            'seed': int(seed),
            'best_epoch': best_epoch,
            'best_score': best_score,
            'test_metrics': test_metrics,
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
# Figure 1: Model Comparison (Box Plot)
# =============================================================================

def plot_model_comparison_boxplot(
    results: pd.DataFrame,
    task: str = 'binding',
    metric: str = 'best_score',
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
):
    """
    Box plot comparing all models for a specific task.
    Shows variance across seeds.
    """
    df = results[results['task'] == task].copy()

    if df.empty:
        print(f"No data for task: {task}")
        return

    # Order models
    model_order = ['lstm', 'transformer', 'mamba', 'hymba', 'circmac']
    model_order = [m for m in model_order if m in df['model'].unique()]

    fig, ax = plt.subplots(figsize=figsize)

    # Create box plot
    positions = range(len(model_order))
    box_data = [df[df['model'] == m][metric].values for m in model_order]

    bp = ax.boxplot(box_data, positions=positions, patch_artist=True, widths=0.6)

    # Color boxes
    for i, (patch, model) in enumerate(zip(bp['boxes'], model_order)):
        patch.set_facecolor(MODEL_COLORS.get(model, '#888888'))
        patch.set_alpha(0.7)

    # Add individual points
    for i, model in enumerate(model_order):
        y = df[df['model'] == model][metric].values
        x = np.random.normal(i, 0.04, size=len(y))
        ax.scatter(x, y, alpha=0.8, color=MODEL_COLORS.get(model, '#888888'),
                   edgecolor='black', s=50, zorder=5)

    # Labels
    ax.set_xticks(positions)
    ax.set_xticklabels([MODEL_NAMES.get(m, m) for m in model_order])
    ax.set_ylabel('AUROC' if task == 'binding' else 'F1 Score')
    ax.set_title(f'{TASK_NAMES.get(task, task)} - Model Comparison')

    # Add mean values as text
    for i, model in enumerate(model_order):
        values = df[df['model'] == model][metric].values
        if len(values) > 0:
            mean_val = np.mean(values)
            ax.annotate(f'{mean_val:.4f}', xy=(i, mean_val),
                       xytext=(0, 10), textcoords='offset points',
                       ha='center', fontsize=10, fontweight='bold')

    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")

    plt.show()
    return fig


def plot_all_tasks_comparison(
    results: pd.DataFrame,
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None
):
    """
    Side-by-side comparison for all three tasks.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    tasks = ['binding', 'sites', 'both']
    model_order = ['lstm', 'transformer', 'mamba', 'hymba', 'circmac']

    for ax, task in zip(axes, tasks):
        df = results[results['task'] == task].copy()
        available_models = [m for m in model_order if m in df['model'].unique()]

        if df.empty:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(TASK_NAMES.get(task, task))
            continue

        # Bar chart with error bars
        means = [df[df['model'] == m]['best_score'].mean() for m in available_models]
        stds = [df[df['model'] == m]['best_score'].std() for m in available_models]
        colors = [MODEL_COLORS.get(m, '#888888') for m in available_models]

        bars = ax.bar(range(len(available_models)), means, yerr=stds,
                      color=colors, alpha=0.7, capsize=5, edgecolor='black')

        ax.set_xticks(range(len(available_models)))
        ax.set_xticklabels([MODEL_NAMES.get(m, m) for m in available_models], rotation=45, ha='right')
        ax.set_ylabel('AUROC' if task == 'binding' else 'F1 Score')
        ax.set_title(TASK_NAMES.get(task, task))
        ax.grid(True, alpha=0.3, axis='y')

        # Highlight best model
        if means:
            best_idx = np.argmax(means)
            bars[best_idx].set_edgecolor('red')
            bars[best_idx].set_linewidth(2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")

    plt.show()
    return fig


# =============================================================================
# Figure 2: Training Curves
# =============================================================================

def plot_training_curves(
    model: str,
    task: str,
    seeds: List[int] = [1, 2, 3],
    metric: str = 'roc_auc',
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
):
    """
    Plot training/validation curves across seeds.
    """
    fig, ax = plt.subplots(figsize=figsize)

    for seed in seeds:
        data = load_training_curves(model, task, seed)
        if data is None:
            continue

        # Extract validation scores
        valid_data = data.get('valid', {})
        epochs = []
        scores = []

        for epoch_str, metrics in valid_data.items():
            epoch = int(epoch_str)
            if task in ['binding', 'both']:
                score = metrics.get('scores', {}).get('bind', {}).get(metric, 0)
            else:
                score = metrics.get('scores', {}).get('sites', {}).get('f1_macro', 0)

            epochs.append(epoch)
            scores.append(score)

        if epochs:
            ax.plot(epochs, scores, label=f'Seed {seed}', alpha=0.8, linewidth=2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric.upper() if task == 'binding' else 'F1 Score')
    ax.set_title(f'{MODEL_NAMES.get(model, model)} - {TASK_NAMES.get(task, task)}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()
    return fig


# =============================================================================
# Figure 3: Binding Site Visualization (Linear)
# =============================================================================

def plot_binding_site_linear(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
    min_span_len: int = 10,
    title: str = "",
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None
):
    """
    Linear visualization of binding site predictions.
    """
    import torch.nn.functional as F
    import torch

    def extract_spans(binary, min_len=1):
        spans = []
        start = None
        for i, val in enumerate(binary):
            if val == 1 and start is None:
                start = i
            elif val == 0 and start is not None:
                if i - start >= min_len:
                    spans.append((start, i))
                start = None
        if start is not None and len(binary) - start >= min_len:
            spans.append((start, len(binary)))
        return spans

    min_len = min(len(probs), len(labels))
    probs = probs[:min_len]
    labels = labels[:min_len]

    preds_binary = (probs >= threshold).astype(int)
    pred_spans = extract_spans(preds_binary, min_span_len)
    true_spans = extract_spans(labels, min_span_len)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot probability curve
    ax.plot(probs, color='dodgerblue', linewidth=1.5, label='Predicted probability')

    # Highlight predicted spans
    for i, (start, end) in enumerate(pred_spans):
        ax.axvspan(start, end, color='skyblue', alpha=0.4,
                   label='Predicted span' if i == 0 else "")

    # Highlight true spans
    for i, (start, end) in enumerate(true_spans):
        ax.axvspan(start, end, color='orange', alpha=0.3,
                   label='True span' if i == 0 else "")

    ax.axhline(y=threshold, linestyle='--', color='gray', alpha=0.5, label=f'Threshold={threshold}')

    ax.set_xlabel('Sequence Position')
    ax.set_ylabel('Binding Probability')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, min_len)
    ax.set_ylim(0, 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()
    return fig


# =============================================================================
# Figure 4: Binding Site Visualization (Circular)
# =============================================================================

def plot_binding_site_circular(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
    title: str = "Circular RNA Binding Sites",
    figsize: Tuple[int, int] = (10, 10),
    save_path: Optional[str] = None
):
    """
    Circular visualization of binding sites - emphasizes circular nature of circRNA.
    """
    min_len = min(len(probs), len(labels))
    probs = probs[:min_len]
    labels = labels[:min_len]

    # Create angles for circular plot
    angles = np.linspace(0, 2 * np.pi, min_len, endpoint=False)

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': 'polar'})

    # Background circle (circRNA backbone)
    theta_circle = np.linspace(0, 2 * np.pi, 100)
    ax.plot(theta_circle, np.ones(100) * 0.5, 'k-', linewidth=2, alpha=0.3)

    # Plot predictions as bars from center
    colors = ['#ff6b6b' if p >= threshold else '#4ecdc4' for p in probs]
    bars = ax.bar(angles, probs, width=2*np.pi/min_len, bottom=0,
                  color=colors, alpha=0.7, edgecolor='none')

    # Mark true binding sites
    true_sites = np.where(labels == 1)[0]
    if len(true_sites) > 0:
        ax.scatter(angles[true_sites], np.ones(len(true_sites)) * 1.1,
                   c='orange', s=20, marker='o', label='True binding site', zorder=5)

    # Styling
    ax.set_ylim(0, 1.2)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    # Add position markers
    ax.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False))
    ax.set_xticklabels([f'{int(i/min_len*100)}%' for i in np.linspace(0, min_len, 8, endpoint=False)])

    ax.set_title(title, y=1.08, fontsize=14)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#ff6b6b', alpha=0.7, label=f'Predicted (≥{threshold})'),
        Patch(facecolor='#4ecdc4', alpha=0.7, label=f'Predicted (<{threshold})'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange',
                   markersize=10, label='True binding site')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()
    return fig


# =============================================================================
# Figure 5: Heatmap - Performance Summary Table
# =============================================================================

def plot_performance_heatmap(
    results: pd.DataFrame,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
):
    """
    Heatmap showing performance across all models and tasks.
    """
    models = ['lstm', 'transformer', 'mamba', 'hymba', 'circmac']
    tasks = ['binding', 'sites', 'both']

    # Create performance matrix
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

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(perf_matrix, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=1.0)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Score', rotation=-90, va="bottom")

    # Labels
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([TASK_NAMES.get(t, t) for t in tasks])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([MODEL_NAMES.get(m, m) for m in models])

    # Add text annotations
    for i in range(len(models)):
        for j in range(len(tasks)):
            if not np.isnan(perf_matrix[i, j]):
                text = f'{perf_matrix[i, j]:.4f}\n±{std_matrix[i, j]:.4f}'
                color = 'white' if perf_matrix[i, j] < 0.75 else 'black'
                ax.text(j, i, text, ha='center', va='center', color=color, fontsize=10)
            else:
                ax.text(j, i, 'N/A', ha='center', va='center', color='gray', fontsize=10)

    ax.set_title('Model Performance Summary')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()
    return fig


# =============================================================================
# Figure 6: Statistical Significance
# =============================================================================

def compute_statistical_tests(results: pd.DataFrame, task: str = 'sites') -> pd.DataFrame:
    """
    Compute pairwise t-tests between models for a specific task.
    """
    from scipy import stats

    df = results[results['task'] == task]
    models = df['model'].unique()

    # Create comparison matrix
    comparisons = []

    for m1 in models:
        for m2 in models:
            if m1 >= m2:
                continue

            scores1 = df[df['model'] == m1]['best_score'].values
            scores2 = df[df['model'] == m2]['best_score'].values

            if len(scores1) < 2 or len(scores2) < 2:
                continue

            t_stat, p_value = stats.ttest_ind(scores1, scores2)

            comparisons.append({
                'Model 1': MODEL_NAMES.get(m1, m1),
                'Model 2': MODEL_NAMES.get(m2, m2),
                'Mean 1': np.mean(scores1),
                'Mean 2': np.mean(scores2),
                'Diff': np.mean(scores1) - np.mean(scores2),
                't-statistic': t_stat,
                'p-value': p_value,
                'Significant': '***' if p_value < 0.001 else ('**' if p_value < 0.01 else ('*' if p_value < 0.05 else ''))
            })

    return pd.DataFrame(comparisons)


# =============================================================================
# Figure 7: Cross-Attention Heatmap
# =============================================================================

def plot_attention_heatmap(
    attn_map: np.ndarray,
    circ_tokens: Optional[List[str]] = None,
    mir_tokens: Optional[List[str]] = None,
    title: str = "Cross-Attention: circRNA ← miRNA",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
    top_k: int = 50,
):
    """
    Heatmap of cross-attention weights between circRNA and miRNA.

    Args:
        attn_map: [L_circ, L_mir] attention weight matrix
        circ_tokens: circRNA token labels (optional)
        mir_tokens: miRNA token labels (optional)
        top_k: Show only top_k circRNA positions with highest attention (for readability)
    """
    L_circ, L_mir = attn_map.shape

    # If circRNA is too long, select top_k most attended positions
    if L_circ > top_k:
        attn_sum = attn_map.sum(axis=1)  # [L_circ]
        top_indices = np.argsort(attn_sum)[-top_k:]
        top_indices = np.sort(top_indices)
        attn_map = attn_map[top_indices]
        if circ_tokens is not None:
            circ_tokens = [circ_tokens[i] for i in top_indices]
        circ_labels = [str(i) for i in top_indices]
    else:
        circ_labels = circ_tokens if circ_tokens else [str(i) for i in range(L_circ)]

    mir_labels = mir_tokens if mir_tokens else [str(i) for i in range(L_mir)]

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        attn_map,
        ax=ax,
        cmap='YlOrRd',
        xticklabels=mir_labels,
        yticklabels=circ_labels,
        cbar_kws={'label': 'Attention Weight'},
        linewidths=0.1,
    )

    ax.set_xlabel('miRNA Position', fontsize=14)
    ax.set_ylabel('circRNA Position', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")

    plt.show()
    return fig


def plot_attention_heatmap_with_sites(
    attn_map: np.ndarray,
    site_probs: np.ndarray,
    site_labels: np.ndarray,
    mir_tokens: Optional[List[str]] = None,
    title: str = "Attention Map with Binding Sites",
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None,
):
    """
    Combined view: attention heatmap + binding site predictions on the side.

    Args:
        attn_map: [L_circ, L_mir] attention weights
        site_probs: [L_circ] predicted binding probabilities
        site_labels: [L_circ] true binding labels (0/1)
    """
    L_circ, L_mir = attn_map.shape
    mir_labels = mir_tokens if mir_tokens else [str(i) for i in range(L_mir)]

    fig, axes = plt.subplots(1, 3, figsize=figsize,
                              gridspec_kw={'width_ratios': [1, 8, 1], 'wspace': 0.05})

    # Left: True labels
    ax_true = axes[0]
    ax_true.imshow(site_labels.reshape(-1, 1), aspect='auto', cmap='Oranges', vmin=0, vmax=1)
    ax_true.set_xticks([0])
    ax_true.set_xticklabels(['True'], fontsize=10)
    ax_true.set_yticks([])
    ax_true.set_ylabel('circRNA Position', fontsize=12)

    # Center: Attention heatmap
    ax_attn = axes[1]
    sns.heatmap(attn_map, ax=ax_attn, cmap='YlOrRd',
                xticklabels=mir_labels, yticklabels=False,
                cbar_kws={'label': 'Attention Weight', 'shrink': 0.8})
    ax_attn.set_xlabel('miRNA Position', fontsize=12)
    ax_attn.set_title(title, fontsize=14, fontweight='bold')

    # Right: Predicted probabilities
    ax_pred = axes[2]
    ax_pred.imshow(site_probs.reshape(-1, 1), aspect='auto', cmap='Blues', vmin=0, vmax=1)
    ax_pred.set_xticks([0])
    ax_pred.set_xticklabels(['Pred'], fontsize=10)
    ax_pred.set_yticks([])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")

    plt.show()
    return fig


# =============================================================================
# Figure 8: Model-wise Site Prediction Comparison
# =============================================================================

def plot_model_site_comparison(
    model_preds: Dict[str, np.ndarray],
    labels: np.ndarray,
    threshold: float = 0.5,
    title: str = "Binding Site Predictions by Model",
    figsize: Tuple[int, int] = (14, None),
    save_path: Optional[str] = None,
):
    """
    Compare binding site predictions from multiple models on the same sample.

    Args:
        model_preds: {model_name: probs_array} dict of predicted probabilities
        labels: true binding site labels (0/1)
        threshold: decision threshold
    """
    n_models = len(model_preds)
    seq_len = len(labels)

    if figsize[1] is None:
        figsize = (figsize[0], 2 + n_models * 2)

    fig, axes = plt.subplots(n_models + 1, 1, figsize=figsize, sharex=True)

    # True labels on top
    ax = axes[0]
    true_color = np.where(labels == 1, 1.0, 0.0)
    ax.fill_between(range(seq_len), 0, labels, color='orange', alpha=0.5, step='mid')
    ax.set_ylabel('True', fontsize=11, fontweight='bold')
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 1])
    ax.grid(True, alpha=0.2)

    # Model predictions
    model_names = list(model_preds.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#e377c2']

    for i, (name, probs) in enumerate(model_preds.items()):
        ax = axes[i + 1]
        probs = probs[:seq_len]

        # Probability curve
        color = colors[i % len(colors)]
        ax.plot(probs, color=color, linewidth=1.5, alpha=0.8)
        ax.fill_between(range(len(probs)), 0, probs,
                         where=probs >= threshold, color=color, alpha=0.3)
        ax.axhline(y=threshold, linestyle='--', color='gray', alpha=0.4, linewidth=0.8)

        # Highlight true regions for reference
        true_spans = _get_spans(labels)
        for s, e in true_spans:
            ax.axvspan(s, e, color='orange', alpha=0.1)

        display_name = MODEL_NAMES.get(name, name)
        ax.set_ylabel(display_name, fontsize=11, fontweight='bold')
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 0.5, 1.0])
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel('Sequence Position', fontsize=12)
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()
    return fig


def _get_spans(binary):
    """Extract contiguous spans from binary array."""
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


# =============================================================================
# Utility: Generate All Figures
# =============================================================================

def generate_all_figures(output_dir: str = 'figures'):
    """
    Generate all paper figures and save to output directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load results
    print("Loading experiment results...")
    results = load_exp1_results()
    print(f"Loaded {len(results)} experiment results")

    if results.empty:
        print("No results found!")
        return

    # Figure 1: Model Comparison Box Plots
    print("\n[1/5] Generating model comparison plots...")
    for task in ['binding', 'sites', 'both']:
        plot_model_comparison_boxplot(
            results, task=task,
            save_path=f'{output_dir}/fig1_{task}_boxplot.png'
        )

    # Figure 2: All Tasks Comparison
    print("\n[2/5] Generating all tasks comparison...")
    plot_all_tasks_comparison(
        results,
        save_path=f'{output_dir}/fig2_all_tasks.png'
    )

    # Figure 3: Performance Heatmap
    print("\n[3/5] Generating performance heatmap...")
    plot_performance_heatmap(
        results,
        save_path=f'{output_dir}/fig3_heatmap.png'
    )

    # Figure 4: Training Curves (for CircMAC)
    print("\n[4/5] Generating training curves...")
    for task in ['binding', 'sites']:
        try:
            plot_training_curves(
                'circmac', task,
                save_path=f'{output_dir}/fig4_circmac_{task}_curves.png'
            )
        except Exception as e:
            print(f"  Could not generate training curves for {task}: {e}")

    # Statistical Tests
    print("\n[5/5] Computing statistical tests...")
    for task in ['binding', 'sites', 'both']:
        stats_df = compute_statistical_tests(results, task)
        if not stats_df.empty:
            stats_df.to_csv(f'{output_dir}/stats_{task}.csv', index=False)
            print(f"  Saved: {output_dir}/stats_{task}.csv")

    print(f"\n✓ All figures saved to '{output_dir}/'")


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate paper figures')
    parser.add_argument('--output', '-o', default='figures', help='Output directory')
    parser.add_argument('--task', '-t', default=None, help='Specific task to plot')
    args = parser.parse_args()

    if args.task:
        results = load_exp1_results()
        plot_model_comparison_boxplot(results, task=args.task)
    else:
        generate_all_figures(args.output)
