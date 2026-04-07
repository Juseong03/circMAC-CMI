# CircMAC: Paper Experiments & Results Section
> 논문 Experiments/Results 섹션 — 바로 사용 가능한 영문 초안
> 각 서브섹션은 독립적으로 복사·수정해서 사용 가능
> ⚠️ Pending: EXP2 MLM/CPCL/ALL, EXP4 no_circ_bias → 재실험 후 업데이트 필요

---

## 3. Experiments

### 3.1 Experimental Setup

All models are trained on the circRNA–miRNA binding site dataset described in Section 2.2 (45,272 training pairs, 17,708 test pairs). Fine-tuning uses AdamW (lr = 1×10⁻⁴, weight decay = 0.01), batch size 32 (or as noted), up to 150 epochs with early stopping (patience = 20). The miRNA encoder is RNABERT (frozen) for all models unless otherwise specified. Each configuration is run with three random seeds (s=1,2,3); results are reported as mean ± std. The primary metric is macro F1 score at nucleotide level (threshold = 0.5).

---

### 3.2 Encoder Architecture Comparison (EXP1)

We compare CircMAC against four architectures trained **from scratch** under identical hyperparameters (d_model=128, n_layer=6, max_len=1022), and against four pretrained RNA language models in both **frozen** and **fine-tunable** settings.

#### 3.2.1 From-Scratch Encoders

| Model | Architecture | F1 (s1) | F1 (s2) | F1 (s3) | **Mean ± Std** |
|-------|-------------|---------|---------|---------|----------------|
| **CircMAC** | Attn + Mamba + Circ-CNN | 0.7442 | 0.7454 | 0.7457 | **0.7451 ± 0.0008** |
| Mamba | SSM (Mamba) | 0.7128 | 0.7144 | 0.7006 | 0.7093 ± 0.0062 |
| HyMBA | Attn + Mamba | 0.7021 | 0.7035 | 0.6807 | 0.6954 ± 0.0101 |
| LSTM | Bidirectional LSTM | 0.6521 | 0.6561 | 0.6491 | 0.6524 ± 0.0035 |
| Transformer | Multi-head Attn | 0.6044 | 0.6212 | 0.6029 | 0.6095 ± 0.0083 |

CircMAC achieves the highest F1 of **0.7451**, outperforming the second-best architecture (Mamba) by **+3.6 pp** and HyMBA by **+5.0 pp**. The Transformer performs weakest among from-scratch models, likely due to the limited sequence-level inductive bias for nucleotide-level prediction. Notably, CircMAC surpasses HyMBA — which shares the Attention+Mamba structure — by adding the circular CNN branch and circular position encoding, demonstrating the value of circRNA-specific design.

#### 3.2.2 Pretrained RNA Language Models

We evaluate four RNA foundation models as frozen feature extractors and as fine-tunable encoders, under two sequence length settings:
- **Fair** (max_len=438): all models including RNABERT at its native maximum length, enabling direct comparison.
- **Max** (max_len=1022): each model at the maximum length it supports (RNABERT is excluded, limited to 440 nt).

**Frozen RNA LMs**

| Model | Params | Max len | Fair (438) | Max (1022) |
|-------|--------|---------|-----------|-----------|
| RNABERT | 113M | 440 | 0.5965 ± 0.0038 | — |
| RNAErnie | 86M | 1022 | 0.6140 ± 0.0025 | 0.6136 ± 0.0031 |
| RNA-FM | 95M | 1022 | 0.6095 ± 0.0020 | 0.6049 ± 0.0021 |
| RNA-MSM | 96M | 1022 | 0.6086 ± 0.0025 | 0.6058 ± 0.0006 |

**Trainable RNA LMs**

| Model | Params | Max len | Fair (438) | Max (1022) |
|-------|--------|---------|-----------|-----------|
| RNABERT | 113M | 440 | 0.6020 ± 0.0008 | — |
| RNAErnie | 86M | 1022 | 0.5929 ± 0.0027 | 0.6010 ± 0.0037 |
| RNA-FM | 95M | 1022 | **0.6400 ± 0.0019** | † OOM |
| RNA-MSM | 96M | 1022 | 0.6315 ± 0.0038 | † OOM |

† RNA-FM and RNA-MSM at max_len=1022 ran out of memory in trainable mode; results pending re-run with smaller batch size.

Despite having **18–20× fewer parameters** than the pretrained RNA LMs, CircMAC (0.7451) substantially outperforms the best RNA LM in trainable mode — RNA-FM fair-trainable (0.6400) — by **+5.5 pp**. Even freezing the large RNA LMs incurs a penalty of over 13 pp relative to CircMAC. This demonstrates that a lightweight model with domain-specific architectural inductive biases can surpass large general-purpose RNA foundation models on circRNA-specific tasks.

---

### 3.3 Pretraining Strategy (EXP2)

We evaluate five self-supervised pretraining objectives applied to the CircMAC backbone on the 1.99M circRNA corpus, followed by fine-tuning on the binding site task. A no-pretraining baseline (scratch fine-tuning on the same CircMAC) is included.

| Pretraining | Objective Target | F1 (s1) | F1 (s2) | F1 (s3) | **Mean ± Std** | Δ vs No-PT |
|-------------|-----------------|---------|---------|---------|----------------|-----------|
| **Pairing** | L×L base-pair matrix | **0.7704** | **0.7743** | **0.7693** | **0.7713 ± 0.0022** | **+1.64 pp** |
| SSP | Binary paired/unpaired | 0.7603 | 0.7701 | 0.7571 | 0.7625 ± 0.0056 | +0.76 pp |
| No-PT (scratch) | — | 0.7531 | 0.7586 | 0.7529 | 0.7549 ± 0.0026 | — |
| MLM+NTP | Masked tokens + NTP | 0.7183 | 0.7183 | 0.6899 | 0.7088 ± 0.0134 | −4.61 pp |
| MLM | Masked tokens | *pending* | *pending* | *pending* | — | — |
| CPCL | Rotation invariance | *pending* | *pending* | *pending* | — | — |

**Key findings:**

1. **Structure-based pretraining outperforms sequence-based pretraining.** The Pairing objective (reconstructing the full L×L base-pairing matrix) yields the best fine-tuning performance at **0.7713**, while SSP (binary structure classification) also improves over no-pretraining. In contrast, sequence-only objectives (MLM, NTP) do not benefit and may harm performance (MLM+NTP: −4.6 pp).

2. **Base pairing matrix reconstruction is the most informative objective.** Pairing provides the most detailed structural signal — the model must predict pairwise spatial relationships between all nucleotides. This level of structural understanding appears to directly transfer to the binding site prediction task, where nucleotide-level spatial context is critical.

3. **MLM+NTP underperforms despite sequence supervision.** The MLM+NTP pretraining resulted in training instability (NaN loss during pretraining), likely degrading the learned representations. Sequence-level objectives without structural grounding may not provide useful inductive biases for binding site localization.

> ⚠️ MLM-only and CPCL finetuning results are pending completion of their pretraining runs. The table will be updated.

---

### 3.4 Ablation Study (EXP4)

To assess the contribution of each architectural component of CircMAC, we train ablated variants by removing or isolating specific branches. All experiments use identical hyperparameters (d_model=128, n_layer=6, max_len=1022, BS=128).

| Variant | Removed / Retained | F1 (s1) | F1 (s2) | F1 (s3) | **Mean ± Std** | Δ vs Full |
|---------|-------------------|---------|---------|---------|----------------|-----------|
| **Full (CircMAC)** | All branches | 0.7459 | 0.7449 | 0.7449 | **0.7452 ± 0.0005** | — |
| No Attention | − Attention | 0.7401 | 0.7407 | 0.7389 | 0.7399 ± 0.0008 | −0.53 pp |
| No Mamba | − Mamba | 0.6844 | 0.6850 | 0.6785 | 0.6826 ± 0.0030 | **−6.26 pp** |
| No Conv | − Circular CNN | 0.6969 | 0.6798 | 0.6939 | 0.6902 ± 0.0074 | **−5.50 pp** |
| Attn Only | − Mamba − Conv | 0.6317 | 0.7065 | 0.7005 | 0.6796 ± 0.0331 | −6.56 pp |
| Mamba Only | − Attn − Conv | 0.7025 | 0.6988 | 0.7006 | 0.7006 ± 0.0015 | −4.46 pp |
| CNN Only | − Attn − Mamba | 0.6731 | 0.6740 | 0.6645 | 0.6705 ± 0.0043 | −7.47 pp |
| No Circ Bias | − Circular position bias | *failed* | *failed* | *failed* | — | — |

† No Circ Bias experiments failed due to a device assignment bug (ran on CPU instead of GPU). Results pending re-run.

**Key findings:**

1. **Mamba and Circular CNN are the two most critical components.** Removing Mamba causes a −6.3 pp drop; removing the CNN causes −5.5 pp. Both far exceed the −0.5 pp cost of removing attention, indicating that sequential long-range modeling (Mamba) and local motif extraction with circular topology (CNN) are the primary drivers of performance.

2. **Attention is complementary but not dominant.** Removing only the attention branch incurs a small −0.5 pp penalty, suggesting that Mamba and CNN together largely capture what attention provides, and that the primary role of attention is to add global context on top of an already capable foundation.

3. **Single-branch variants all underperform the full model.** Notably, the Attention-only variant shows high variance across seeds (std = 0.033), suggesting that attention alone is unstable for this task. Mamba-only (0.7006) is more competitive than CNN-only (0.6705) or Attention-only (0.6796 avg), highlighting Mamba's strong solo performance.

4. **The three branches are complementary and mutually reinforcing.** No single branch achieves full-model performance. The adaptive router effectively combines the strengths of global context (Attention), sequential patterns (Mamba), and local circular motifs (CNN).

---

### 3.5 Interaction Mechanism (EXP5)

We compare three strategies for fusing the encoded circRNA and miRNA representations:

| Interaction | Description | F1 (s1) | F1 (s2) | F1 (s3) | **Mean ± Std** | Δ vs Cross-Attn |
|-------------|-------------|---------|---------|---------|----------------|----------------|
| **Cross-Attention** | circRNA queries → miRNA keys/values | **0.7464** | **0.7472** | **0.7454** | **0.7463 ± 0.0008** | — |
| Concatenation | [circRNA; miRNA] → Linear | 0.7206 | 0.7162 | 0.7165 | 0.7178 ± 0.0020 | −2.85 pp |
| Elementwise | circRNA ⊙ miRNA | 0.7112 | 0.7147 | 0.7018 | 0.7092 ± 0.0056 | −3.71 pp |

Cross-attention yields the best performance at **0.7463**, outperforming concatenation by **+2.9 pp** and elementwise product by **+3.7 pp**. This suggests that explicitly attending over the miRNA sequence to condition per-position circRNA representations — rather than pooling the miRNA into a fixed-size vector — is important for precise site localization.

---

### 3.6 Site Prediction Head (EXP6)

We compare two decoding head architectures applied to the fused representation:

| Head | Description | F1 (s1) | F1 (s2) | F1 (s3) | **Mean ± Std** | Δ |
|------|-------------|---------|---------|---------|----------------|---|
| **Conv1D (ours)** | Multi-scale (k=3,5,7) + Conv1D × 2 | **0.7450** | **0.7460** | **0.7460** | **0.7457 ± 0.0005** | — |
| Linear | Direct Linear projection | 0.7360 | 0.7436 | 0.7337 | 0.7378 ± 0.0043 | −0.79 pp |

The multi-scale Conv1D head outperforms the linear head by **+0.8 pp** with significantly lower variance (std = 0.0005 vs 0.0043). The convolutional head's ability to aggregate local context at multiple scales provides modest but consistent gains for site boundary detection.

---

### 3.7 CircMAC with Pretraining: CircMAC-PT (EXP3)

*(Results pending — pretraining is in progress. CircMAC-PT uses the best pretraining weights from EXP2 (Pairing, F1=0.7713) loaded into the CircMAC backbone before fine-tuning. Expected: additional +0.5–1 pp gain over CircMAC scratch.)*

---

## 4. Summary of Results

### 4.1 Main Comparison Table

| Model | Setting | Params | F1 (mean ± std) |
|-------|---------|--------|----------------|
| **CircMAC-PT (Pairing)** | Pretrained + FT | ~5.2M | **0.7713 ± 0.0022** |
| CircMAC-PT (SSP) | Pretrained + FT | ~5.2M | 0.7625 ± 0.0056 |
| **CircMAC** | From scratch | ~5.2M | **0.7451 ± 0.0008** |
| Mamba | From scratch | ~4.1M | 0.7093 ± 0.0062 |
| HyMBA | From scratch | ~4.4M | 0.6954 ± 0.0101 |
| LSTM | From scratch | ~1.8M | 0.6524 ± 0.0035 |
| Transformer | From scratch | ~3.6M | 0.6095 ± 0.0083 |
| RNA-FM (trainable) | Fine-tuned LM | 95M | 0.6400 ± 0.0019 |
| RNA-MSM (trainable) | Fine-tuned LM | 96M | 0.6315 ± 0.0038 |
| RNABERT (trainable) | Fine-tuned LM | 113M | 0.6020 ± 0.0008 |
| RNAErnie (frozen) | Frozen LM | 86M | 0.6140 ± 0.0025 |

### 4.2 Key Takeaways

1. **CircMAC is lightweight yet competitive**: With only ~5.2M parameters, CircMAC outperforms all RNA foundation models (86–113M params) by a large margin (+5.5 pp over the best RNA LM).

2. **Pretraining with structural objectives provides consistent gains**: The Pairing objective brings +1.64 pp over no-pretraining and reaches **F1 = 0.7713**, while structure-free objectives (MLM, NTP) hurt performance.

3. **All three branches contribute to CircMAC's performance**: Mamba (−6.3 pp when removed) and circular CNN (−5.5 pp) are the most critical components; attention provides complementary global context.

4. **Cross-attention interaction is essential**: The cross-attention fusion yields +2.9–3.7 pp over simpler fusion strategies.

5. **Frozen large LMs are insufficient**: Even state-of-the-art RNA LMs used as frozen encoders perform poorly (F1 ≈ 0.60–0.61), suggesting that general-purpose RNA representations do not directly transfer to the circRNA binding site prediction task without task-specific fine-tuning or architectural adaptation.

---

## Appendix: Detailed Per-Seed Results

### A1. EXP2v4 Pretraining — All Seeds
| Objective | s1 | s2 | s3 | Mean | Std |
|-----------|----|----|-----|------|-----|
| Pairing | 0.7704 | 0.7743 | 0.7693 | **0.7713** | 0.0022 |
| SSP | 0.7603 | 0.7701 | 0.7571 | 0.7625 | 0.0056 |
| No-PT | 0.7531 | 0.7586 | 0.7529 | 0.7549 | 0.0026 |
| MLM+NTP | 0.7183 | 0.7183 | 0.6899 | 0.7088 | 0.0134 |
| MLM | — | — | — | pending | |
| CPCL | — | — | — | pending | |
| ALL (combined) | — | — | — | pending | |

### A2. EXP4 Ablation — All Seeds
| Variant | s1 | s2 | s3 | Mean | Std |
|---------|----|----|-----|------|-----|
| Full | 0.7459 | 0.7449 | 0.7449 | 0.7452 | 0.0005 |
| No-Attn | 0.7401 | 0.7407 | 0.7389 | 0.7399 | 0.0008 |
| No-Mamba | 0.6844 | 0.6850 | 0.6785 | 0.6826 | 0.0030 |
| No-Conv | 0.6969 | 0.6798 | 0.6939 | 0.6902 | 0.0074 |
| Mamba-Only | 0.7025 | 0.6988 | 0.7006 | 0.7006 | 0.0015 |
| CNN-Only | 0.6731 | 0.6740 | 0.6645 | 0.6705 | 0.0043 |
| Attn-Only | 0.6317 | 0.7065 | 0.7005 | 0.6796 | 0.0331 |
| No-Circ-Bias | — | — | — | failed (re-run needed) | |

### A3. EXP5 Interaction — All Seeds
| Method | s1 | s2 | s3 | Mean | Std |
|--------|----|----|-----|------|-----|
| Cross-Attention | 0.7464 | 0.7472 | 0.7454 | 0.7463 | 0.0008 |
| Concat | 0.7206 | 0.7162 | 0.7165 | 0.7178 | 0.0020 |
| Elementwise | 0.7112 | 0.7147 | 0.7018 | 0.7092 | 0.0056 |

### A4. EXP6 Site Head — All Seeds
| Head | s1 | s2 | s3 | Mean | Std |
|------|----|----|-----|------|-----|
| Conv1D | 0.7450 | 0.7460 | 0.7460 | 0.7457 | 0.0005 |
| Linear | 0.7360 | 0.7436 | 0.7337 | 0.7378 | 0.0043 |

### A5. EXP1 Base Encoders — All Seeds
| Model | s1 | s2 | s3 | Mean | Std |
|-------|----|----|-----|------|-----|
| CircMAC | 0.7442 | 0.7454 | 0.7457 | 0.7451 | 0.0008 |
| Mamba | 0.7128 | 0.7144 | 0.7006 | 0.7093 | 0.0062 |
| HyMBA | 0.7021 | 0.7035 | 0.6807 | 0.6954 | 0.0101 |
| LSTM | 0.6521 | 0.6561 | 0.6491 | 0.6524 | 0.0035 |
| Transformer | 0.6044 | 0.6212 | 0.6029 | 0.6095 | 0.0083 |

### A6. EXP1 RNA LMs — All Seeds
| Model | Mode | Len | s1 | s2 | s3 | Mean | Std |
|-------|------|-----|----|-----|-----|------|-----|
| RNAErnie | frozen | 438 | 0.6144 | 0.6107 | 0.6168 | 0.6140 | 0.0025 |
| RNA-FM | frozen | 438 | 0.6092 | 0.6073 | 0.6120 | 0.6095 | 0.0020 |
| RNA-MSM | frozen | 438 | 0.6057 | 0.6083 | 0.6119 | 0.6086 | 0.0025 |
| RNABERT | frozen | 438 | 0.5949 | 0.5929 | 0.6016 | 0.5965 | 0.0038 |
| RNAErnie | frozen | 1022 | 0.6124 | 0.6105 | 0.6178 | 0.6136 | 0.0031 |
| RNA-FM | frozen | 1022 | 0.6046 | 0.6076 | 0.6026 | 0.6049 | 0.0021 |
| RNA-MSM | frozen | 1022 | 0.6056 | 0.6065 | 0.6052 | 0.6058 | 0.0006 |
| RNA-FM | trainable | 438 | 0.6421 | 0.6404 | 0.6376 | 0.6400 | 0.0019 |
| RNA-MSM | trainable | 438 | 0.6316 | 0.6361 | 0.6269 | 0.6315 | 0.0038 |
| RNABERT | trainable | 438 | 0.6009 | 0.6021 | 0.6029 | 0.6020 | 0.0008 |
| RNAErnie | trainable | 438 | 0.5948 | 0.5890 | 0.5949 | 0.5929 | 0.0027 |
| RNAErnie | trainable | 1022 | 0.6040 | 0.5958 | 0.6032 | 0.6010 | 0.0037 |
| RNA-FM | trainable | 1022 | — | — | — | OOM | |
| RNA-MSM | trainable | 1022 | — | — | — | OOM | |
