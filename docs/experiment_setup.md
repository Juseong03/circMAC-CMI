# Experiment Setup (Final)

> Last updated: 2026-02-12
> For LaTeX paper writing reference

---

## Overview

| Exp | Research Question | Description | Runs | Script (2GPU) |
|-----|------------------|-------------|------|---------------|
| **1** | RQ1: Pretrained RNA models vs CircMAC-PT | Pretrained Model Comparison (fair + max) | 48 | `exp1_fair.sh` + `exp1_max.sh` |
| **2** | RQ2: Best pretraining strategy | Pretraining Task Analysis (ss1 vs ss5) | 75 | `exp2_ss1.sh` + `exp2_ss5.sh` |
| **3** | RQ3: Best encoder architecture | Encoder Architecture Comparison | 15 | `exp3_exp5.sh` |
| **4** | RQ4: CircMAC component contributions | CircMAC Ablation Study | 27 | `exp4_exp6.sh` |
| **5** | RQ5: Best interaction mechanism | Interaction Mechanism Comparison | 9 | `exp3_exp5.sh` |
| **6** | RQ6: Best site head structure | Site Head Structure Comparison | 6 | `exp4_exp6.sh` |
| | | **Total** | **180** | |

---

## Dataset

### Source
- circRNA-miRNA interaction dataset
- Train: `df_train_final.pkl` | Test: `df_test_final.pkl`

### Statistics

| | Train | Test |
|--|-------|------|
| Total samples | 45,272 | 17,708 |
| Binding = 1 (positive) | 19,674 | 7,838 |
| Binding = 0 (negative) | 25,598 | 9,870 |
| **Sites task** (binding=1 only) | **19,674** | **7,838** |

### Sequence Length

| | circRNA | miRNA |
|--|---------|-------|
| Mean | 601 nt | 22 nt |
| Min | 150 nt | 16 nt |
| Max | 1,000 nt | 28 nt |

### Data Availability by max_len (Sites task)

| max_len | Train samples | % of total | Test samples |
|---------|--------------|------------|--------------|
| 438 (rnabert) | 7,343 | 37.3% | 2,817 |
| 510 (rnaernie) | 9,479 | 48.2% | 3,680 |
| 1,022 (rnafm/rnamsm/circmac) | 19,674 | 100.0% | 7,838 |
| **440 (fair comparison)** | **~7,400** | **~37.6%** | **~2,830** |

### Data Split
- Train / Validation / Test = 50% / 22% / 28% (of train set)
- Test set (`df_test_final.pkl`) is held out entirely
- 3 random seeds (1, 2, 3) for reproducibility

### Pretraining Data
| File | Description | Samples |
|------|-------------|---------|
| `df_circ_ss.pkl` | 1 secondary structure per circRNA | ~29,973 |
| `df_circ_ss_5.pkl` | 5 stochastic SS samples per circRNA | ~149,249 |

---

## Task Definition

All experiments use **task = sites** (token-level binding site prediction).

```
Input:  circRNA sequence c = (c_1, ..., c_L), miRNA sequence m = (m_1, ..., m_M)
Output: Site labels y = (y_1, ..., y_L) where y_i in {0, 1}
```

- **Sites**: Per-position binary classification (which nucleotides are binding sites?)
- **Binding**: Derived from sites (mean pooling of site probabilities)

---

## Common Hyperparameters

| Parameter | Value | Note |
|-----------|-------|------|
| d_model | 128 | Hidden dimension |
| n_layer | 6 | Number of encoder blocks |
| optimizer | AdamW | - |
| lr | 1e-4 | Learning rate |
| epochs | 150 | Max training epochs |
| earlystop | 20 | Early stopping patience |
| seeds | 1, 2, 3 | Report mean +/- std |
| interaction | cross_attention | Default for all experiments |
| target_model | rnabert | miRNA encoder (frozen) |
| site_head | conv1d | Default (except exp6) |
| max_len | 1024 | Default (except exp1 fair) |
| kmer | 1 | Single nucleotide tokenization |

---

## Evaluation Metrics

### Sites Task (Primary)
| Metric | Description | LaTeX |
|--------|-------------|-------|
| **F1 (macro)** | Macro-averaged F1 score | $F_1^{macro}$ |
| **F1 (positive)** | F1 for binding class | $F_1^{pos}$ |
| **AUROC** | Area under ROC curve | AUROC |
| **AUPRC** | Area under Precision-Recall curve | AUPRC |
| **MCC** | Matthews Correlation Coefficient | MCC |
| Accuracy | Overall accuracy | Acc |
| Precision (macro/pos) | Macro/positive precision | $P^{macro}$, $P^{pos}$ |
| Recall (macro/pos) | Macro/positive recall | $R^{macro}$, $R^{pos}$ |
| **Span-F1** | F1 on contiguous binding spans | $F_1^{span}$ |
| Span-Precision | Precision on spans | $P^{span}$ |
| Span-Recall | Recall on spans | $R^{span}$ |

### Binding Task (Derived)
| Metric | Description |
|--------|-------------|
| F1 (macro) | Macro-averaged F1 |
| AUROC | Area under ROC curve |
| AUPRC | Area under PR curve |
| MCC | Matthews Correlation Coefficient |

### Threshold
- Site threshold: Sweep over [0.1, 0.9] on validation set, select best by F1 macro

---

## Experiment 1: Pretrained RNA Model Comparison (RQ1)

> **Question**: Does CircMAC with circRNA-specific pretraining outperform general-purpose pretrained RNA models?

### Models

| Model | Source | Pretrain Data | Hidden Dim | Max Pos | Trainable Params |
|-------|--------|---------------|------------|---------|------------------|
| RNABERT | multimolecule | RNAcentral | 120 | 440 | ~1M |
| RNA-FM | multimolecule | RNAcentral | 640 | 1024 | ~95M |
| RNAErnie | multimolecule | RNAcentral | 768 | 512 | ~86M |
| RNA-MSM | multimolecule | RNAcentral | 768 | 1024 | ~96M |
| **CircMAC-PT** | Ours | circRNA (ss1/ss5) | 128 | 1024 | ~3M |

### Setup

| Parameter | Value |
|-----------|-------|
| batch_size | 32 (smaller for large pretrained models) |
| epochs | 150 |
| earlystop | 20 |

### Part 1: Fair Comparison (max_len = 440)
All models train/evaluate on **identical data** (max_len=440, rnabert's limit).

| Model | Mode | Runs |
|-------|------|------|
| RNABERT, RNA-FM, RNAErnie, RNA-MSM | Frozen (encoder fixed, projection trainable) | 4 x 3 = 12 |
| RNABERT, RNA-FM, RNAErnie, RNA-MSM | Trainable (encoder fine-tuned) | 4 x 3 = 12 |
| CircMAC-PT | Fine-tuned | 1 x 3 = 3 |
| **Subtotal** | | **27** |

### Part 2: Max Performance (model-specific max_len)
Each model uses its own maximum sequence length. Skip RNABERT (same as fair).

| Model | Mode | max_len | Runs |
|-------|------|---------|------|
| RNA-FM, RNAErnie, RNA-MSM | Frozen | model-specific | 3 x 3 = 9 |
| RNA-FM, RNAErnie, RNA-MSM | Trainable | model-specific | 3 x 3 = 9 |
| CircMAC-PT | Fine-tuned | 1024 | 1 x 3 = 3 |
| **Subtotal** | | | **21** |

**Total Exp1: 48 runs**

### Table Template (LaTeX)

```latex
% Table: Fair Comparison (max_len=440)
\begin{table}[t]
\caption{Comparison with pretrained RNA models (fair, max\_len=440)}
\begin{tabular}{llcccccc}
\toprule
Model & Mode & $F_1^{macro}$ & $F_1^{pos}$ & AUROC & AUPRC & MCC & $F_1^{span}$ \\
\midrule
RNABERT & Frozen & & & & & & \\
RNABERT & Trainable & & & & & & \\
RNA-FM & Frozen & & & & & & \\
RNA-FM & Trainable & & & & & & \\
RNAErnie & Frozen & & & & & & \\
RNAErnie & Trainable & & & & & & \\
RNA-MSM & Frozen & & & & & & \\
RNA-MSM & Trainable & & & & & & \\
\midrule
\textbf{CircMAC-PT} & Fine-tuned & & & & & & \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Experiment 2: Pretraining Strategy Analysis (RQ2)

> **Question**: Which self-supervised pretraining tasks are most effective for circRNA binding site prediction?

### Pretraining Configurations

| Config ID | Tasks | Description |
|-----------|-------|-------------|
| No-PT | None | No pretraining (baseline) |
| MLM | MLM | Masked Language Modeling (15% masking) |
| MLM+NTP | MLM, NTP | + Next Token Prediction |
| MLM+SSP | MLM, SSP | + Secondary Structure Prediction |
| MLM+Pair | MLM, Pairing | + Base Pairing Matrix (CircularPairingHead) |
| MLM+CPCL | MLM, CPCL | + **Circular Permutation Contrastive Learning** |
| MLM+BSJ | MLM, BSJ_MLM | + **BSJ-focused MLM** (50% masking near BSJ) |
| MLM+CPCL+BSJ | MLM, CPCL, BSJ_MLM | Circular-aware combination |
| MLM+CPCL+BSJ+Pair | MLM, CPCL, BSJ_MLM, Pairing | **Best candidate** |
| Full | All 7 tasks | MLM+NTP+SSP+SSL_multi+Pairing+CPCL+BSJ_MLM |

### Data Variants

| Tag | File | Description | Samples |
|-----|------|-------------|---------|
| ss1 | `df_circ_ss` | 1 SS per RNA | ~30K |
| ss5 | `df_circ_ss_5` | 5 stochastic SS per RNA | ~150K |

### Setup

**Phase 1: Pretraining**
| Parameter | Value |
|-----------|-------|
| batch_size | 64 |
| epochs | 300 |
| earlystop | 30 |
| max_len | 1022 |

**Phase 2: Fine-tuning**
| Parameter | Value |
|-----------|-------|
| batch_size | 128 |
| epochs | 150 |
| earlystop | 20 |

### Run Count
- Phase 1: 9 configs x 2 data = **18 pretrain runs**
- Phase 2: (1 baseline + 9 x 2 configs) x 3 seeds = **57 finetune runs**
- **Total Exp2: 75 runs**

### Table Template (LaTeX)

```latex
\begin{table}[t]
\caption{Effect of pretraining strategies on binding site prediction}
\begin{tabular}{lcccccc}
\toprule
Pretrain Config & Data & $F_1^{macro}$ & $F_1^{pos}$ & AUROC & AUPRC & $\Delta F_1$ \\
\midrule
No pretrain & - & & & & & baseline \\
\midrule
MLM & ss1 & & & & & \\
MLM+NTP & ss1 & & & & & \\
MLM+SSP & ss1 & & & & & \\
MLM+Pair & ss1 & & & & & \\
MLM+CPCL & ss1 & & & & & \\
MLM+BSJ & ss1 & & & & & \\
MLM+CPCL+BSJ & ss1 & & & & & \\
MLM+CPCL+BSJ+Pair & ss1 & & & & & \\
Full & ss1 & & & & & \\
\midrule
MLM & ss5 & & & & & \\
... & ... & & & & & \\
\textbf{MLM+CPCL+BSJ+Pair} & \textbf{ss5} & & & & & \\
Full & ss5 & & & & & \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Experiment 3: Encoder Architecture Comparison (RQ3)

> **Question**: Which encoder architecture is best for circRNA representation learning (from scratch)?

### Models

| Model | Type | Key Feature |
|-------|------|-------------|
| LSTM | RNN | Bidirectional sequential processing |
| Transformer | Attention | Global self-attention |
| Mamba | SSM | State Space Model (selective scan) |
| Hymba | Hybrid | Attention + Mamba |
| **CircMAC** | Hybrid | **Attention + Mamba + CNN + Circular-aware** |

### Setup
| Parameter | Value |
|-----------|-------|
| batch_size | 128 |
| Pretraining | None (from scratch) |

**Total Exp3: 5 models x 3 seeds = 15 runs**

### Table Template (LaTeX)

```latex
\begin{table}[t]
\caption{Encoder architecture comparison for binding site prediction}
\begin{tabular}{llccccccc}
\toprule
Model & Type & $F_1^{macro}$ & $F_1^{pos}$ & AUROC & AUPRC & MCC & $F_1^{span}$ \\
\midrule
LSTM & RNN & & & & & & \\
Transformer & Attention & & & & & & \\
Mamba & SSM & & & & & & \\
Hymba & Hybrid & & & & & & \\
\textbf{CircMAC} & \textbf{Hybrid} & & & & & & \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Experiment 4: CircMAC Ablation Study (RQ4)

> **Question**: How much does each CircMAC component contribute to performance?

### Ablation Configurations

| Config | Attention | Mamba | CNN | Circ. Bias | Circ. Padding | Flag |
|--------|:---------:|:-----:|:---:|:----------:|:-------------:|------|
| **Full CircMAC** | O | O | O | O | O | (none) |
| w/o Attention | X | O | O | - | O | `--no_attn` |
| w/o Mamba | O | X | O | O | O | `--no_mamba` |
| w/o CNN | O | O | X | O | - | `--no_conv` |
| w/o Circ. Bias | O | O | O | X | O | `--no_circular_rel_bias` |
| w/o Circ. Padding | O | O | O | O | X | `--no_circular_window` |
| Attention only | O | X | X | O | - | `--no_mamba --no_conv` |
| Mamba only | X | X | O | - | O | `--no_attn --no_conv` |
| CNN only | X | O | X | - | O | `--no_attn --no_mamba` |

### Setup
| Parameter | Value |
|-----------|-------|
| batch_size | 128 |
| model | circmac (from scratch) |

**Total Exp4: 9 configs x 3 seeds = 27 runs**

### Table Template (LaTeX)

```latex
\begin{table}[t]
\caption{Ablation study of CircMAC components}
\begin{tabular}{lcccccc}
\toprule
Configuration & $F_1^{macro}$ & $F_1^{pos}$ & AUROC & AUPRC & $F_1^{span}$ & $\Delta F_1$ \\
\midrule
\textbf{Full CircMAC} & & & & & & - \\
\midrule
\multicolumn{7}{l}{\textit{Branch ablation}} \\
\quad w/o Attention & & & & & & \\
\quad w/o Mamba & & & & & & \\
\quad w/o CNN & & & & & & \\
\midrule
\multicolumn{7}{l}{\textit{Circular feature ablation}} \\
\quad w/o Circular Bias & & & & & & \\
\quad w/o Circular Padding & & & & & & \\
\midrule
\multicolumn{7}{l}{\textit{Single branch}} \\
\quad Attention only & & & & & & \\
\quad Mamba only & & & & & & \\
\quad CNN only & & & & & & \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Experiment 5: Interaction Mechanism (RQ5)

> **Question**: Which interaction mechanism between circRNA and miRNA is most effective?

### Configurations

| Mechanism | Description | Flag |
|-----------|-------------|------|
| Concat | Concatenate circRNA and miRNA embeddings | `--interaction concat` |
| Elementwise | Element-wise multiplication | `--interaction elementwise` |
| **Cross-attention** | miRNA attends to circRNA | `--interaction cross_attention` |

### Setup
| Parameter | Value |
|-----------|-------|
| model | circmac (from scratch) |
| batch_size | 128 |

**Total Exp5: 3 mechanisms x 3 seeds = 9 runs**

### Table Template (LaTeX)

```latex
\begin{table}[t]
\caption{Effect of interaction mechanism}
\begin{tabular}{lccccc}
\toprule
Interaction & $F_1^{macro}$ & $F_1^{pos}$ & AUROC & AUPRC & $F_1^{span}$ \\
\midrule
Concatenation & & & & & \\
Element-wise & & & & & \\
\textbf{Cross-attention} & & & & & \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Experiment 6: Site Head Structure (RQ6)

> **Question**: Which classifier head is better for token-level site prediction?

### Configurations

| Head Type | Description | Flag |
|-----------|-------------|------|
| **Conv1D** | Multi-scale 1D convolution classifier | `--site_head_type conv1d` |
| Linear | Simple linear projection | `--site_head_type linear` |

### Setup
| Parameter | Value |
|-----------|-------|
| model | circmac (from scratch) |
| batch_size | 128 |

**Total Exp6: 2 types x 3 seeds = 6 runs**

### Table Template (LaTeX)

```latex
\begin{table}[t]
\caption{Effect of site head architecture}
\begin{tabular}{lccccc}
\toprule
Site Head & $F_1^{macro}$ & $F_1^{pos}$ & AUROC & AUPRC & $F_1^{span}$ \\
\midrule
Linear & & & & & \\
\textbf{Conv1D} & & & & & \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Server Distribution

| Server | GPU 0 | GPU 1 |
|--------|-------|-------|
| **Server 1** | exp2 ss1 (36 runs) | exp2 ss5 (39 runs) |
| **Server 2** | exp1 fair (27 runs) | exp1 max (21 runs) |
| **Server 3** | exp3 + exp5 (24 runs) | exp4 + exp6 (33 runs) |

### Execution Order
1. **Server 1 & 3**: Start immediately (independent)
2. **Server 2**: exp1 CircMAC-PT requires exp2 best model → run after Server 1 finishes (or auto-skip)

---

## Visualization Experiments (Post-training)

> Run AFTER main experiments complete. Requires trained model checkpoints.

### V1. Binding Site Prediction Heatmap
- **Input**: Best model checkpoint + test samples
- **Output**: Side-by-side comparison of predicted vs true binding sites
- **Purpose**: Qualitative demonstration of site localization quality
- **Timing**: After exp3 (best encoder) or exp2 (best pretrained)

### V2. Attention Map Visualization
- **Input**: Cross-attention weights from forward pass
- **Output**: circRNA-miRNA attention heatmap
- **Purpose**: Show which circRNA positions attend to miRNA
- **Timing**: After exp5 (cross-attention is best)

### V3. BSJ Region Analysis
- **Input**: Per-position prediction accuracy near BSJ (positions 0~10% and 90%~100%)
- **Output**: Accuracy curve by position (BSJ vs middle)
- **Purpose**: Demonstrate circular-aware features help BSJ region
- **Timing**: After exp4 (full vs w/o circular features)

### V4. Training Curves
- **Input**: Training logs from all experiments
- **Output**: Loss/F1 over epochs
- **Purpose**: Show convergence behavior
- **Timing**: After all experiments

### V5. Pretraining Task Contribution
- **Input**: Exp2 results (all pretraining configs)
- **Output**: Bar chart of F1 improvement over no-pretrain baseline
- **Purpose**: Show which pretraining tasks contribute most
- **Timing**: After exp2

### V6. Ablation Component Contribution
- **Input**: Exp4 results
- **Output**: Bar chart showing F1 drop when each component is removed
- **Purpose**: Show each component is essential
- **Timing**: After exp4

### V7. Router Weight Distribution (if applicable)
- **Input**: CircMAC router weights per position
- **Output**: Distribution of branch weights (attention/mamba/cnn)
- **Purpose**: Analyze which branch is preferred for different sequence regions
- **Timing**: After exp4

### V8. Sequence Length Stratified Analysis
- **Input**: Test predictions grouped by circRNA length
- **Output**: Performance by length bin
- **Purpose**: Show model robustness across sequence lengths
- **Timing**: After exp3

---

## Paper Figure Plan

| Figure | Content | Source Exp |
|--------|---------|------------|
| **Fig 1** | CircMAC architecture overview | - |
| **Fig 2** | Circular-aware features (bias, padding, CPCL) | - |
| **Fig 3** | Encoder comparison bar chart + training curves | Exp 3 |
| **Fig 4** | Pretrained model comparison (fair + max) | Exp 1 |
| **Fig 5** | Pretraining strategy analysis | Exp 2 |
| **Fig 6** | Ablation study bar chart | Exp 4 |
| **Fig 7** | Binding site prediction visualization | V1, V2 |
| **Fig 8** | BSJ region analysis | V3 |

### Supplementary Figures
- Interaction mechanism comparison (Exp 5)
- Site head comparison (Exp 6)
- Router weight distribution (V7)
- Full training curves (V4)
- Additional case studies

---

## Reproducibility Checklist

- [x] Seeds: 1, 2, 3 (report mean +/- std)
- [x] Data splits: Fixed pkl files
- [x] Hyperparameters: Documented above
- [x] All scripts: `scripts/2gpu/`
- [ ] Environment: `requirements.txt`
- [ ] Hardware: GPU model, VRAM
- [ ] Training time per experiment
- [ ] Model parameter counts
