# CMI-MAC: Circular RNA-MicroRNA Interaction with Multi-branch Attention and Circular-aware Modeling

## 1. Introduction

### 1.1 Background

**Circular RNAs (circRNAs)** are covalently closed RNA molecules that function as microRNA (miRNA) sponges, regulating gene expression. Accurate prediction of circRNA-miRNA binding sites is crucial for understanding disease mechanisms and therapeutic target discovery.

### 1.2 Problem Statement

Limitations of existing approaches:
1. **Linear sequence modeling**: Most models treat circRNA as linear sequences, ignoring circular topology
2. **BSJ (Back-Splice Junction) neglect**: Critical junction region characteristics are not captured
3. **Single-branch architecture**: Limited pattern diversity with single Attention or RNN structures

### 1.3 Our Approach: CMI-MAC

**C**ircRNA-**M**iRNA **I**nteraction with **M**ulti-branch **A**ttention and **C**ircular-aware modeling

**Key Contributions:**
1. **CircMAC Architecture**: 3-branch hybrid (Attention + Mamba + Circular CNN)
2. **Circular-aware Pretraining**: CPCL, BSJ_MLM specialized for circRNA
3. **Site-First Unified Head**: Sites prediction as main task, binding derived via pooling

---

## 2. Architecture

### 2.1 CircMAC Block

```
Input [B, L, D]
    |
    +--- in_proj ---+--- Q, K, V --- Attention (circular bias) ---+
    |               |                                              |
    |               +--- base ------- Mamba (sequential) ---------+
    |                         |                                    |
    |                         +------ CircularCNN (local) --------+
    |                                                              |
    +--- 3-branch Router --- Weighted Fusion --- out_proj --- Output [B, L, D]
```

### 2.2 Key Components

| Component | Role | Circular-aware |
|-----------|------|----------------|
| **AttentionBranch** | Global dependency | Circular relative bias |
| **Mamba** | Sequential patterns | Inherent in SSM |
| **CircularCNN** | Local motifs | Circular padding |
| **3-branch Router** | Adaptive fusion | Dynamic weighting |

### 2.3 Model Comparison

| Feature | LSTM | Transformer | Mamba | Hymba | TTHymba | **CircMAC** |
|---------|------|-------------|-------|-------|---------|-------------|
| Global Attention | X | O | X | O | O | **O** |
| Sequential SSM | X | X | O | O | O | **O** |
| Local CNN | X | X | X | X | X | **O** |
| Circular Bias | X | X | X | X | X | **O** |
| Circular Padding | X | X | X | X | X | **O** |

---

## 3. Site-First Unified Head (New Approach)

### 3.1 Philosophy

```
OLD: Separate heads for binding (CLS token) and sites (per-position)
NEW: Sites prediction is MAIN task, binding is DERIVED from sites
```

**Key Changes:**
- **No CLS token needed** - aligns with circular nature of circRNA
- **Sites prediction**: Per-position binary classification [B, L, 1]
- **Binding prediction**: Derived via mean/max pooling of site probabilities

### 3.2 UnifiedSiteHead Architecture

```python
class UnifiedSiteHead:
    def forward(x, mask):
        # 1. Feature enhancement with MultiKernelCNN
        x_enhanced = feature_enhancer(x)  # [B, L, D]

        # 2. Per-position site prediction (MAIN TASK)
        sites_logits = site_classifier(x_enhanced)  # [B, L, 1]
        sites_probs = sigmoid(sites_logits)

        # 3. Derive binding from sites (SECONDARY)
        binding_score = mean(sites_probs, dim=1)  # [B, 1]
        binding_logits = logit(binding_score)

        return {
            'sites_logits': sites_logits,
            'binding_logits': binding_logits,
            'sites_probs': sites_probs
        }
```

### 3.3 Usage

```bash
python training.py \
    --model_name circmac \
    --task sites \
    --use_unified_head \
    --binding_pooling mean \
    --is_cross_attention
```

---

## 4. Pretraining Strategy

### 4.1 Self-supervised Tasks

| Task | Description | circRNA Relevance |
|------|-------------|-------------------|
| **MLM** | Masked Language Modeling | General sequence |
| **NTP** | Next Token Prediction | Sequential pattern |
| **SSP** | Secondary Structure Prediction | RNA folding |
| **CPCL** | Circular Permutation CL | **Circular identity** |
| **BSJ_MLM** | BSJ-focused MLM | **Junction learning** |
| **Pairing** | Base pair prediction | Structure |

### 4.2 Novel Pretraining Tasks

#### CPCL (Circular Permutation Contrastive Learning)
- circRNA is the same molecule regardless of starting position
- Learn that circular shifts produce same embeddings
- InfoNCE loss for positive pairs (original, shifted)

#### BSJ_MLM (Back-Splice Junction MLM)
- Higher masking probability at BSJ region (sequence start/end)
- 50% of masked tokens from BSJ region
- Enhanced junction context learning

---

## 5. Project Structure

```
cmi_mac/
├── models/
│   ├── circmac.py          # CircMAC architecture (main)
│   ├── transformer.py      # Transformer (baseline)
│   ├── mamba.py            # Mamba (baseline)
│   ├── hymba.py            # Hymba (baseline)
│   ├── lstm.py             # LSTM (baseline)
│   ├── model.py            # ModelWrapper
│   ├── heads.py            # Task heads (UnifiedSiteHead, etc.)
│   └── modules.py          # Common modules
│
├── trainer.py              # Training/pretraining loops
├── training.py             # Supervised training CLI
├── pretraining.py          # Self-supervised pretraining CLI
├── data.py                 # Dataset classes
├── utils.py                # Utilities and metrics
├── utils_config.py         # Model configurations
│
├── scripts/
│   ├── exp1_pretrained_comparison.sh  # Exp 1: Pretrained models
│   ├── exp2_encoder_comparison.sh     # Exp 2: Encoder comparison
│   ├── exp3_ablation.sh               # Exp 3: Ablation study
│   ├── exp4_pretraining_tasks.sh      # Exp 4: Pretraining analysis
│   └── run_all.sh                     # Run all experiments
│
├── docs/
│   ├── overview.md         # This file
│   ├── experiments.md      # Experiment design
│   └── draft.md            # Paper draft notes
│
├── check_results.py        # Monitor experiment progress
└── generate_figures.py     # Generate paper figures
```

---

## 6. Experiments

### 6.1 Overview

| Exp | Focus | Key Finding |
|-----|-------|-------------|
| **1** | Pretrained Model Comparison | CircMAC-PT > RNABert, RNAFM, etc. |
| **2** | Encoder Architecture | CircMAC best for circRNA |
| **3** | Ablation Study | Each circular component essential |
| **4** | Pretraining Tasks | CPCL + BSJ_MLM are key |

### 6.2 Metrics

**Sites Prediction (Main):**
- F1 Score, IoU, AUROC, AUPRC
- Span F1, Span Precision, Span Recall

**Derived Binding:**
- F1 Score, Accuracy, MCC, AUROC

---

## 7. Quick Start

### 7.1 Training with Unified Head

```bash
# Single experiment
python training.py \
    --model_name circmac \
    --task sites \
    --use_unified_head \
    --binding_pooling mean \
    --d_model 128 \
    --n_layer 6 \
    --batch_size 128 \
    --epochs 150 \
    --device 0

# Run all experiments
bash scripts/run_all.sh 0
```

### 7.2 Check Progress

```bash
python check_results.py --summary
```

### 7.3 Generate Figures

```bash
python generate_figures.py --format png --dpi 300
```

---

## 8. Key Files Modified for Unified Head

| File | Changes |
|------|---------|
| `models/heads.py` | Added `UnifiedSiteHead` class |
| `models/model.py` | Added `_set_unified_site_head()` method |
| `trainer.py` | Full unified head support in training loop |
| `training.py` | Added `--use_unified_head`, `--binding_pooling` args |

---

## 9. Citation

```bibtex
@article{circmac2025,
  title={CMI-MAC: Circular RNA-MicroRNA Interaction Prediction with
         Multi-branch Attention and Circular-aware Modeling},
  author={...},
  year={2025}
}
```
