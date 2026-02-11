# Experiment Design (Revised v2)

## Key Change: Site-First Unified Head Approach

**Core Philosophy**:
- **Sites prediction is the MAIN task** (per-position binding site classification)
- **Binding prediction is DERIVED from sites** via mean/max pooling
- **No CLS token needed** - aligns with circular nature of circRNA

```python
# New unified head approach
--use_unified_head          # Enable site-first approach
--binding_pooling mean      # Derive binding from mean of site probabilities
```

---

## Research Questions

| RQ | Question |
|----|----------|
| **RQ1** | Pretrained RNA 모델 대비 CircMAC (w/ pretraining)의 sites 예측 성능은? |
| **RQ2** | Encoder 아키텍처 비교: CircMAC이 다른 encoder보다 우수한가? |
| **RQ3** | CircMAC의 각 circular component가 성능에 미치는 영향은? |
| **RQ4** | 어떤 pretraining task가 sites 예측에 효과적인가? |

---

## Experiment Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    EXPERIMENT PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [Exp 1] Pretrained Model Comparison                           │
│     CircMAC-PT vs RNABERT vs RNAFM vs RNAErnie vs RNAMSM       │
│     Task: Sites prediction (unified head)                       │
│     → CircMAC이 pretrained 모델보다 강력!                        │
│                           ↓                                     │
│  [Exp 2] Encoder Architecture Comparison                        │
│     LSTM → Transformer → Mamba → Hymba → TTHymba → CircMAC     │
│     Task: Sites prediction (unified head, from scratch)         │
│     → CircMAC encoder가 circRNA에 최적화됨                       │
│                           ↓                                     │
│  [Exp 3] Ablation Study                                        │
│     CircMAC component 기여도 분석                                │
│     → 각 circular component가 필수적임                           │
│                           ↓                                     │
│  [Exp 4] Pretraining Task Analysis                              │
│     MLM, CPCL, BSJ_MLM 등 효과 분석                             │
│     → Circular-aware pretraining이 핵심                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Experiment 1: Pretrained Model Comparison (RQ1)

### 1.1 Objective
CircMAC (with pretraining) vs 기존 pretrained RNA 모델 비교
**핵심 포인트**: 우리 모델이 circRNA에 특화된 pretraining으로 더 강력!

### 1.2 Models

| Model | Source | Pretraining Data | Note |
|-------|--------|------------------|------|
| RNABERT | multimolecule | RNAcentral | General RNA |
| RNAFM | multimolecule | RNAcentral | General RNA |
| RNAErnie | multimolecule | RNAcentral | General RNA |
| RNAMSM | multimolecule | RNAcentral | General RNA |
| **CircMAC-PT** | Ours | circRNA data | **Circular-aware** |

### 1.3 Experimental Setup

**Hyperparameters:**
| Parameter | Value | Note |
|-----------|-------|------|
| d_model | 128 | - |
| n_layer | 6 | - |
| batch_size | 32 | Smaller for pretrained models |
| epochs | 150 | - |
| early_stop | 20 | - |
| lr | 1e-4 | - |
| use_unified_head | True | **Site-first approach** |
| binding_pooling | mean | Derive binding from sites |

**Total runs**: 5 models x 3 seeds = **15 runs**

```bash
bash scripts/exp1_pretrained_comparison.sh [GPU_ID]
```

### 1.4 Results Table Template

**Table 1: Sites Prediction - Pretrained Models**

| Model | Sites F1 | Sites IoU | AUROC | AUPRC | Derived Binding F1 |
|-------|----------|-----------|-------|-------|-------------------|
| RNABERT | | | | | |
| RNAFM | | | | | |
| RNAErnie | | | | | |
| RNAMSM | | | | | |
| **CircMAC-PT** | | | | | |

### 1.5 Visualization
- [ ] Sites prediction heatmap comparison
- [ ] Attention map visualization
- [ ] BSJ region performance comparison

---

## Experiment 2: Encoder Architecture Comparison (RQ2)

### 2.1 Objective
동일 조건에서 encoder 아키텍처별 성능 비교 (from scratch, no pretraining)
**핵심 포인트**: CircMAC encoder가 circRNA에 최적화됨

### 2.2 Models

| Model | Type | Key Feature |
|-------|------|-------------|
| LSTM | RNN | Sequential processing |
| Transformer | Attention | Global attention |
| Mamba | SSM | State space model |
| Hymba | Hybrid | Attention + Mamba |
| TTHymba | Hybrid | Attention + Mamba + Router |
| **CircMAC** | Hybrid | **Attention + Mamba + CNN + Circular-aware** |

### 2.3 Experimental Setup

**Hyperparameters:**
| Parameter | Value |
|-----------|-------|
| d_model | 128 |
| n_layer | 6 |
| batch_size | 128 |
| epochs | 150 |
| early_stop | 20 |
| lr | 1e-4 |
| use_unified_head | True |
| binding_pooling | mean |

**Total runs**: 6 models x 3 seeds = **18 runs**

```bash
bash scripts/exp2_encoder_comparison.sh [GPU_ID]
```

### 2.4 Results Table Template

**Table 2: Sites Prediction - Encoder Comparison**

| Model | Sites F1 | Sites IoU | AUROC | AUPRC | Derived Binding F1 |
|-------|----------|-----------|-------|-------|-------------------|
| LSTM | | | | | |
| Transformer | | | | | |
| Mamba | | | | | |
| Hymba | | | | | |
| TTHymba | | | | | |
| **CircMAC** | | | | | |

### 2.5 Visualization
- [ ] Performance bar chart
- [ ] Training curves comparison
- [ ] BSJ region analysis

---

## Experiment 3: Ablation Study (RQ3)

### 3.1 Objective
CircMAC의 각 component 기여도 분석
**핵심 포인트**: 각 circular component가 필수적임을 증명

### 3.2 Ablation Configurations

| Config | Attention | Mamba | CNN | Circular Bias | Circular Padding | Flag |
|--------|-----------|-------|-----|---------------|------------------|------|
| **Full** | O | O | O | O | O | (none) |
| w/o Attn | X | O | O | - | O | `--no_attn` |
| w/o Mamba | O | X | O | O | O | `--no_mamba` |
| w/o CNN | O | O | X | O | - | `--no_conv` |
| w/o Circ Bias | O | O | O | X | O | `--no_circular_rel_bias` |
| w/o Circ Pad | O | O | O | O | X | `--no_circular_window` |

### 3.3 Experimental Setup

**Total runs**: 6 configs x 3 seeds = **18 runs**

```bash
bash scripts/exp3_ablation.sh [GPU_ID]
```

### 3.4 Results Table Template

**Table 3: Ablation Study Results**

| Configuration | Sites F1 | Sites IoU | Delta F1 | Delta IoU |
|--------------|----------|-----------|------|-------|
| **Full CircMAC** | | | - | - |
| w/o Attention | | | | |
| w/o Mamba | | | | |
| w/o CNN | | | | |
| w/o Circular Bias | | | | |
| w/o Circular Padding | | | | |

### 3.5 Visualization
- [ ] Component contribution bar chart
- [ ] Circular vs Non-circular comparison

---

## Experiment 4: Pretraining Task Analysis (RQ4)

### 4.1 Objective
어떤 pretraining task가 sites 예측에 효과적인지 분석
**핵심 포인트**: Circular-aware pretraining (CPCL, BSJ_MLM)이 핵심

### 4.2 Pretraining Configurations

| Config ID | Tasks | Description |
|-----------|-------|-------------|
| PT-0 | None | No pretraining (baseline) |
| PT-1 | MLM | Masked Language Modeling only |
| PT-2 | MLM + NTP | + Next Token Prediction |
| PT-3 | MLM + SSP | + Secondary Structure Prediction |
| PT-4 | MLM + CPCL | + **Circular Permutation CL** |
| PT-5 | MLM + BSJ_MLM | + **BSJ-focused MLM** |
| PT-6 | MLM + CPCL + BSJ_MLM | **Circular-aware combination** |
| PT-7 | Full | All pretraining tasks |

### 4.3 Experimental Setup

**Phase 1: Pretraining**
| Parameter | Value |
|-----------|-------|
| d_model | 128 |
| n_layer | 6 |
| batch_size | 64 |
| epochs | 300 |
| early_stop | 30 |

**Phase 2: Fine-tuning**
| Parameter | Value |
|-----------|-------|
| batch_size | 128 |
| epochs | 150 |
| early_stop | 20 |
| use_unified_head | True |

**Total runs**: 8 configs x 3 seeds = **24 runs** (fine-tuning only)

```bash
bash scripts/exp4_pretraining_tasks.sh [GPU_ID]
```

### 4.4 Results Table Template

**Table 4: Pretraining Task Effect**

| Pretrain Config | Sites F1 | Sites IoU | Delta vs No-PT |
|-----------------|----------|-----------|------------|
| No Pretrain | | | baseline |
| MLM | | | |
| MLM + NTP | | | |
| MLM + SSP | | | |
| MLM + CPCL | | | |
| MLM + BSJ_MLM | | | |
| **MLM + CPCL + BSJ** | | | |
| Full | | | |

### 4.5 Visualization
- [ ] Pretraining effect bar chart
- [ ] CPCL vs non-CPCL comparison

---

## Summary: All Experiments

| Exp | Focus | Models/Configs | Seeds | Total Runs | Key Finding |
|-----|-------|----------------|-------|------------|-------------|
| 1 | Pretrained models | 5 | 3 | 15 | CircMAC > general RNA models |
| 2 | Encoder comparison | 6 | 3 | 18 | CircMAC best for circRNA |
| 3 | Ablation | 6 | 3 | 18 | Each component essential |
| 4 | Pretraining tasks | 8 | 3 | 24 | CPCL + BSJ_MLM key |
| **Total** | | | | **75** | |

---

## Execution Order and Timeline

```
Week 1: Exp 2 (Encoder Comparison)
    -> Show CircMAC encoder is best for circRNA
    -> Can start immediately (from scratch)

Week 2: Exp 3 (Ablation Study)
    -> Show each circular component is essential
    -> Can run in parallel with Exp 2

Week 3-4: Exp 4 (Pretraining Tasks)
    -> Find best pretraining config
    -> Requires pretrained models

Week 5: Exp 1 (Pretrained Model Comparison)
    -> Compare with existing RNA models
    -> Depends on: Best pretrain config from Exp 4

Week 6-7: Paper Writing
    -> Compile results, create figures
```

---

## Figure Plan

### Main Figures (4-5 figures)

1. **Figure 1: Model Architecture**
   - CircMAC block diagram
   - Circular-aware components highlighted

2. **Figure 2: Pretrained Model Comparison (Exp 1)**
   - Bar chart: Sites F1, IoU comparison
   - Heatmap: Sites prediction visualization

3. **Figure 3: Encoder Comparison (Exp 2)**
   - Bar chart: 6 encoders x Sites metrics
   - Training curves

4. **Figure 4: Ablation Study (Exp 3)**
   - Component contribution bar chart
   - Circular component importance

5. **Figure 5: Pretraining Analysis (Exp 4)**
   - Pretraining task effect
   - CPCL/BSJ_MLM contribution

### Supplementary Figures

- BSJ region analysis
- Attention visualization
- Error analysis
- Case studies

---

## Scripts Summary

| Script | Purpose | Total Runs |
|--------|---------|------------|
| `scripts/exp1_pretrained_comparison.sh` | Exp 1: Pretrained models | 15 |
| `scripts/exp2_encoder_comparison.sh` | Exp 2: Encoder comparison | 18 |
| `scripts/exp3_ablation.sh` | Exp 3: Ablation study | 18 |
| `scripts/exp4_pretraining_tasks.sh` | Exp 4: Pretraining tasks | 24 |
| `scripts/run_all.sh` | Run all experiments | 75 |

---

## Reproducibility Checklist

- [ ] Seeds: 1, 2, 3 (report mean +/- std)
- [ ] Data splits: Fixed and saved
- [ ] Hyperparameters: Documented in each experiment
- [ ] Environment: requirements.txt frozen
- [ ] Checkpoints: Best model saved for each config
- [ ] Logs: Training logs saved to `logs/`
- [ ] Code: Use `--use_unified_head` for site-first approach
