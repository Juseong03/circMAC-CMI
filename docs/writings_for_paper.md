# Paper Writing Materials

## Paper Title Options

1. **CircMAC: Circular-aware Multi-branch Architecture for circRNA-miRNA Interaction Prediction**
2. **Learning Circular RNA Representations with Back-Splice Junction Aware Pretraining**
3. **Beyond Linear Sequences: Circular-aware Deep Learning for circRNA-miRNA Binding Site Prediction**

---

## Abstract (Draft)

> Circular RNAs (circRNAs) are covalently closed RNA molecules that play crucial roles in gene regulation by acting as microRNA (miRNA) sponges. Accurate prediction of circRNA-miRNA interactions and binding sites is essential for understanding disease mechanisms. However, existing methods treat circRNAs as linear sequences, ignoring their inherent circular topology and the unique characteristics of Back-Splice Junctions (BSJ). We propose **CircMAC**, a **Circ**ular-aware **M**ulti-branch **A**ttention and **C**onvolutional architecture that explicitly models the circular nature of circRNAs. CircMAC integrates three complementary branches—Attention with circular relative bias, Mamba for sequential modeling, and Circular CNN for local pattern extraction—with an adaptive routing mechanism. We further introduce novel self-supervised pretraining tasks: **Circular Permutation Contrastive Learning (CPCL)** and **BSJ-focused Masked Language Modeling (BSJ_MLM)**, designed to capture circRNA-specific properties. Extensive experiments demonstrate that CircMAC outperforms state-of-the-art methods on circRNA-miRNA binding prediction and achieves superior binding site localization, particularly in BSJ regions.

---

## 1. Introduction

### 1.1 Opening (Hook)
```
Circular RNAs (circRNAs) have emerged as important regulators of gene expression,
particularly through their role as competitive endogenous RNAs that sequester
microRNAs (miRNAs). Unlike linear RNAs, circRNAs form covalently closed loops
through back-splice junctions (BSJ), granting them remarkable stability and
unique functional properties. Understanding circRNA-miRNA interactions is crucial
for [disease application].
```

### 1.2 Problem Statement
```
Despite significant progress in RNA interaction prediction, existing methods
suffer from a critical limitation: they model circRNAs as linear sequences,
completely ignoring the circular topology that defines these molecules.
This linearization introduces several problems:

1. **Artificial boundaries**: The BSJ, where the 3' and 5' ends connect,
   is treated as two distant sequence ends.

2. **Position-dependent representations**: Models learn position-specific
   features that don't account for rotational equivalence.

3. **Incomplete local context**: Patterns spanning the BSJ are fragmented.
```

### 1.3 Our Contributions
```
We address these limitations with the following contributions:

1. **CircMAC Architecture**: A novel multi-branch architecture combining
   Attention (with circular relative bias), Mamba (for sequential patterns),
   and Circular CNN (with circular padding) through adaptive routing.

2. **Circular-aware Pretraining**: Two novel self-supervised tasks:
   - CPCL: Learning rotational invariance of circular sequences
   - BSJ_MLM: Focused masking around back-splice junctions

3. **Comprehensive Evaluation**: Extensive experiments showing [X]% improvement
   on binding prediction and [Y]% on binding site localization.
```

---

## 2. Related Work

### 2.1 circRNA-miRNA Interaction Prediction
- Traditional: sequence alignment, thermodynamics (RNAhybrid, miRanda)
- Deep learning: CNN-based, RNN-based
- Transformer-based: attention mechanisms
- **Gap**: None explicitly model circular topology

### 2.2 RNA Language Models
- RNABERT, RNAErnie, RNA-FM
- Pretraining objectives: MLM, structure prediction
- **Gap**: Designed for linear RNAs, not circular

### 2.3 State Space Models for Sequences
- Mamba, S4, Hyena
- Advantages for long sequences
- **Gap**: Not adapted for circular sequences

---

## 3. Methods

### 3.1 Problem Formulation
```
Given:
- circRNA sequence: c = (c_1, c_2, ..., c_L) where c_L is adjacent to c_1
- miRNA sequence: m = (m_1, m_2, ..., m_M)

Tasks:
1. Binding prediction: f(c, m) → {0, 1}
2. Binding site prediction: g(c, m) → {0, 1}^L
```

### 3.2 CircMAC Architecture

#### 3.2.1 Circular Relative Position Bias
```
Standard relative position: d(i,j) = |i - j|
Circular relative position: d_circ(i,j) = min(|i-j|, L-|i-j|)

This ensures positions near the BSJ (e.g., position 0 and L-1)
have small relative distances.
```

#### 3.2.2 Circular CNN
```
Standard padding: pad(x) = [0, ..., 0, x_1, ..., x_L, 0, ..., 0]
Circular padding: pad_circ(x) = [x_{L-k}, ..., x_L, x_1, ..., x_L, x_1, ..., x_k]

This allows convolution kernels to see context across the BSJ.
```

#### 3.2.3 Multi-branch Fusion
```
Given input x:
  branch_attn = Attention(x)  # global with circular bias
  branch_mamba = Mamba(x)     # sequential
  branch_cnn = CircularCNN(x) # local patterns

  # Adaptive routing
  weights = Router(concat(branch_attn, branch_mamba, branch_cnn))
  output = Σ weights_i * branch_i
```

### 3.3 Circular-aware Pretraining

#### 3.3.1 CPCL (Circular Permutation Contrastive Learning)
```
Key insight: A circRNA is equivalent to any of its circular permutations.

For sequence c = (c_1, ..., c_L):
  c' = CircularShift(c, k) = (c_{k+1}, ..., c_L, c_1, ..., c_k)

Objective: Representations of c and c' should be similar.

Loss: InfoNCE with (c, c') as positive pair
```

#### 3.3.2 BSJ_MLM (Back-Splice Junction focused MLM)
```
Standard MLM: Random 15% masking
BSJ_MLM: Biased masking toward BSJ region

BSJ region: positions [0, 0.1L] ∪ [0.9L, L]
Mask distribution: 50% from BSJ region, 50% random

This forces the model to learn BSJ-specific context.
```

---

## 4. Experiments

### 4.1 Experimental Setup

#### Dataset
- **Source**: circRNA-miRNA interaction database
- **Training set**: 45,272 samples (df_train_final.pkl)
- **Test set**: 17,708 samples (df_test_final.pkl)
- **Max sequence length**: 1,024 nucleotides
- **Split**: 50% train, 22% validation, 28% test

#### Tasks
1. **Binding**: Binary classification (does circRNA bind to miRNA?)
2. **Sites**: Token-level classification (which positions are binding sites?)
3. **Both**: Multi-task learning (binding + sites jointly)

#### Baseline Models
| Category | Models |
|----------|--------|
| RNN-based | LSTM |
| Attention-based | Transformer |
| SSM-based | Mamba |
| Hybrid | Hymba |
| Pretrained RNA | RNABERT, RNAFM, RNAErnie, RNAMSM |
| **Ours** | CircMAC, CircMAC-PT |

#### Training Configuration
| Parameter | Exp1/Exp4 | Exp2 (PT) | Exp2 (FT) | Exp3 |
|-----------|-----------|-----------|-----------|------|
| d_model | 128 | 128 | 128 | 128 |
| n_layer | 6 | 6 | 6 | - |
| batch_size | 128 | 64 | 128 | 32 |
| epochs | 150 | 300 | 150 | 150 |
| early_stop | 20 | 30 | 20 | 20 |
| lr | 1e-4 | 1e-4 | 1e-4 | 1e-4 |
| optimizer | AdamW | AdamW | AdamW | AdamW |
| num_workers | 8 | 8 | 8 | 8 |

#### Evaluation Metrics
- **Binding task**: Accuracy, F1, Precision, Recall, MCC, ROC-AUC, PR-AUC
- **Sites task**: Token-level Accuracy, F1, Precision, Recall
- **All experiments**: 3 random seeds (1, 2, 3) with mean ± std reported

#### Implementation
- Framework: PyTorch
- Target model: RNABERT (for miRNA encoding)
- Cross-attention: Enabled for all models
- Hardware: NVIDIA GPU

### 4.2 Main Results (Table Template)

**Table 1: Encoder Architecture Comparison (Binding Task)**

| Model | Type | Acc | F1 | AUROC | AUPRC | MCC |
|-------|------|-----|----|----|----|----|
| LSTM | RNN | - | - | - | - | - |
| Transformer | Attention | - | - | - | - | - |
| Mamba | SSM | - | - | - | - | - |
| Hymba | Hybrid | - | - | - | - | - |

| **CircMAC** | Ours | **-** | **-** | **-** | **-** | **-** |

**Table 2: Encoder Architecture Comparison (Sites Task)**

| Model | Acc | F1 | IoU | Precision | Recall |
|-------|-----|----|-----|-----------|--------|
| LSTM | - | - | - | - | - |
| Transformer | - | - | - | - | - |
| Mamba | - | - | - | - | - |
| Hymba | - | - | - | - | - |

| **CircMAC** | **-** | **-** | **-** | **-** | **-** |

**Table 3: Comparison with Pretrained RNA Models**

| Model | Pretrain Data | Binding F1 | Sites F1 | Sites IoU |
|-------|--------------|------------|----------|-----------|
| RNABERT | RNAcentral | - | - | - |
| RNAFM | RNAcentral | - | - | - |
| RNAErnie | RNAcentral | - | - | - |
| RNAMSM | RNAcentral | - | - | - |
| **CircMAC-PT** | circRNA | **-** | **-** | **-** |

### 4.3 Ablation Study (Table Template)

**Table 4: Ablation Study - Component Analysis**

| Configuration | Flag | Binding F1 | Sites F1 | Both F1 |
|--------------|------|----|----|---|
| Full CircMAC | - | - | - | - |
| w/o Attention | `--no_attn` | - | - | - |
| w/o Mamba | `--no_mamba` | - | - | - |
| w/o Circular CNN | `--no_conv` | - | - | - |
| w/o Circular Rel. Bias | `--no_circular_rel_bias` | - | - | - |
| w/o Circular Padding | `--no_circular_window` | - | - | - |
| Attention Only | `--no_mamba --no_conv` | - | - | - |
| Mamba Only | `--no_attn --no_conv` | - | - | - |
| CNN Only | `--no_attn --no_mamba` | - | - | - |

### 4.4 Pretraining Analysis (Table Template)

**Table 5: Effect of Pretraining Strategies**

| Config Name | Pretrain Tasks | Binding F1 | Sites F1 | Both F1 |
|-------------|----------------|------------|----------|---------|
| No pretrain | - | - | - | - |
| mlm | MLM | - | - | - |
| mlm_ntp | MLM + NTP | - | - | - |
| mlm_ssp | MLM + SSP | - | - | - |
| mlm_pairing | MLM + Pairing | - | - | - |
| mlm_cpcl | MLM + CPCL | - | - | - |
| mlm_bsj | MLM + BSJ_MLM | - | - | - |
| mlm_cpcl_bsj | MLM + CPCL + BSJ_MLM | - | - | - |
| mlm_cpcl_bsj_pair | MLM + CPCL + BSJ_MLM + Pairing | - | - | - |
| full | All tasks | - | - | - |

**Pretraining Task Descriptions:**
- **MLM**: Masked Language Modeling (15% masking)
- **NTP**: Next Token Prediction
- **SSP**: Secondary Structure Prediction
- **Pairing**: Base Pairing Matrix Prediction (circular-aware)
- **CPCL**: Circular Permutation Contrastive Learning (novel)
- **BSJ_MLM**: Back-Splice Junction focused MLM (novel)

---

## 5. Analysis & Discussion

### 5.1 Why Circular-awareness Matters
- Case study: BSJ spanning binding sites
- Visualization of attention patterns
- Comparison of linear vs circular distance effects

### 5.2 Branch Contribution Analysis
- When is Attention most useful? (long-range)
- When is CNN most useful? (local motifs)
- When is Mamba most useful? (sequential patterns)
- Router weight distribution analysis

### 5.3 Binding Site Localization
- Position-wise accuracy analysis
- BSJ region vs middle region comparison
- Example visualizations

---

## 6. Conclusion

```
We presented CircMAC, a novel architecture for circRNA-miRNA interaction
prediction that explicitly models the circular nature of circRNAs. Through
circular relative position bias, circular padding in CNN, and novel
pretraining tasks (CPCL and BSJ_MLM), CircMAC achieves state-of-the-art
performance on both binding prediction and binding site localization.

Our work demonstrates the importance of incorporating domain-specific
structural knowledge into deep learning models for RNA analysis.
Future directions include extending this approach to other circular
biomolecules and exploring circRNA-protein interactions.
```

---

## Figures to Create

### Figure 1: Overview
- (a) circRNA structure with BSJ highlighted
- (b) CircMAC architecture diagram
- (c) Three branches with circular-aware features

### Figure 2: Circular-aware Features
- (a) Linear vs Circular relative distance matrix
- (b) Standard vs Circular padding illustration
- (c) CPCL concept: rotational invariance

### Figure 3: Main Results
- (a) Bar chart: model comparison
- (b) ROC curves
- (c) PR curves

### Figure 4: Ablation Study
- (a) Component ablation bar chart
- (b) Pretraining ablation bar chart

### Figure 5: Binding Site Analysis
- (a) Example binding site predictions (color-coded)
- (b) Position-wise accuracy heatmap
- (c) BSJ region vs middle region accuracy

### Figure 6: Attention/Router Analysis
- (a) Attention weight visualization
- (b) Router weight distribution by position
- (c) Case study: BSJ-spanning site

---

## Supplementary Materials

### A. Implementation Details

#### A.1 CircMAC Architecture Hyperparameters
```
d_model: 128          # Hidden dimension
n_layer: 6            # Number of CircMAC blocks
n_heads: 8            # Attention heads (d_model / 16)
d_state: 16           # Mamba state dimension
d_conv: 4             # Mamba conv dimension
expand: 2             # Mamba expansion factor
conv_kernel: 7        # Circular CNN kernel size
```

#### A.2 Training Configuration
```
Optimizer: AdamW
Learning rate: 1e-4
Weight decay: 0.01
Batch size: 128 (64 for pretraining, 32 for pretrained models)
Max epochs: 150 (300 for pretraining)
Early stopping: 20 epochs (30 for pretraining)
Gradient clipping: 1.0
```

#### A.3 Data Processing
```
Max sequence length: 1024
K-mer: 1 (single nucleotide)
Vocab: A, C, G, U, [CLS], [EOS], [PAD], [MASK]
Target encoding: RNABERT (mean pooling)
```

### B. Dataset Statistics
- Sequence length distributions
- Binding site statistics
- Train/Val/Test split details

### C. Additional Results
- Per-miRNA family performance
- Sequence length stratified results
- Computational efficiency comparison

### D. Visualization Gallery
- More binding site prediction examples
- Attention pattern examples

---

## Key Phrases for Paper

### Novelty Claims
- "first to explicitly model circular topology in circRNA representation learning"
- "novel circular-aware pretraining objectives"
- "adaptive multi-branch architecture"

### Technical Contributions
- "circular relative position bias"
- "circular permutation contrastive learning"
- "back-splice junction focused masking"

### Results Claims (to be filled)
- "achieves X% improvement in binding prediction accuracy"
- "improves binding site IoU by Y%"
- "particularly effective in BSJ region with Z% accuracy gain"
