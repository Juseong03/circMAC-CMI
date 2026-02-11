# Paper Draft: Key Ideas & Sentences

> 이 문서는 full paper 작성을 위한 핵심 아이디어, key sentences, 논리 흐름을 정리한 것입니다.
> 각 섹션별로 **왜 → 문제 → 해결** 구조로 작성되어 있습니다.

---

## Title Options

1. **CircMAC: Circular-aware Multi-branch Architecture for circRNA-miRNA Interaction Prediction**
2. **Beyond Linear Sequences: Learning Circular RNA Representations with Topology-aware Deep Learning**
3. **Modeling Circular Topology for circRNA-miRNA Binding Site Prediction**

**Keywords**: circRNA, miRNA, circular topology, back-splice junction, deep learning, sequence modeling, binding site prediction

---

## Abstract (Key Points)

```
[Background] circRNA는 miRNA sponge로서 유전자 조절에 중요한 역할
[Gap] 기존 방법들은 circRNA를 linear sequence로 취급 → circular 특성 무시
[Method] CircMAC: Attention + Mamba + CNN with circular-aware features
[Novelty] CPCL, BSJ_MLM 등 circular-specific pretraining
[Results] binding prediction과 site localization에서 SOTA 달성
[Impact] circular biomolecule 연구의 새로운 패러다임 제시
```

---

## 1. Introduction

### 1.1 Why Important? (연구의 중요성)

**Key Idea**: circRNA-miRNA 상호작용은 질병 메커니즘 이해와 치료 타겟 발굴에 핵심

**Key Sentences**:
- "Circular RNAs (circRNAs) have emerged as critical regulators of gene expression, functioning primarily as microRNA (miRNA) sponges that modulate post-transcriptional regulation."
- "Unlike linear RNAs, circRNAs possess exceptional stability due to their covalently closed loop structure, making them promising biomarkers and therapeutic targets."
- "Accurate prediction of circRNA-miRNA interactions is essential for understanding disease mechanisms, including cancer, cardiovascular diseases, and neurological disorders."

**Supporting Evidence** (논문에서 인용할 것):
- circRNA가 miRNA sponge로 작용한다는 연구들
- circRNA와 질병의 연관성 연구들
- circRNA의 안정성과 생물학적 특성

---

### 1.2 Current Limitations (현재 한계)

**Key Idea**: 기존 방법들은 circRNA의 핵심 특성인 "circular topology"를 무시

**Problem 1: Artificial Boundary at BSJ**
```
Linear view:   5'---[sequence]---3'
                     ↑ BSJ treated as two distant ends

Circular view:      ┌──────────┐
                    │ sequence │ ← BSJ connects 3' to 5'
                    └──────────┘
```
- "Existing methods treat circRNAs as linear sequences, introducing artificial boundaries at the back-splice junction (BSJ)."
- "This linearization disrupts the continuity of sequence patterns that span across the BSJ."

**Problem 2: Position-dependent Representations**
- "Linear position encodings assign distant indices to nucleotides adjacent in the circular structure."
- "Models learn spurious position-specific features that fail to capture rotational equivalence."

**Problem 3: Incomplete Local Context**
- "Convolutional filters cannot capture patterns that cross the artificial boundary."
- "Local motifs spanning the BSJ are fragmented into separate, unrelated patterns."

**Problem 4: Pretraining Objectives Ignore Circularity**
- "Standard masked language modeling (MLM) does not account for the circular nature of circRNAs."
- "Existing RNA language models (RNABERT, RNAFM) are trained on linear RNAs."

**Key Sentence (Summary)**:
- "Despite the circular nature being the defining characteristic of circRNAs, no existing deep learning method explicitly models this topology."

---

### 1.3 Our Solution (제안하는 해결책)

**Key Idea**: Circular topology를 명시적으로 모델링하는 아키텍처와 pretraining 제안

**Solution Overview**:
```
Problem                          → Solution
─────────────────────────────────────────────────────
Artificial BSJ boundary          → Circular padding in CNN
Position-dependent features      → Circular relative position bias
Missing circular pretraining     → CPCL (rotational invariance)
BSJ region underrepresented      → BSJ_MLM (focused masking)
Single modality limitation       → Multi-branch fusion (Attn + Mamba + CNN)
```

**Key Sentences**:
- "We propose CircMAC, a **Circ**ular-aware **M**ulti-branch **A**ttention and **C**onvolutional architecture that explicitly models the circular topology of circRNAs."
- "Our key insight is that a circRNA sequence is equivalent to any of its circular permutations, and the model should learn representations that respect this invariance."

---

### 1.4 Contributions (기여)

**Contribution 1: Architecture**
- "We introduce CircMAC, a novel multi-branch architecture combining Attention (with circular relative bias), Mamba (for sequential patterns), and Circular CNN (with circular padding) through adaptive routing."

**Contribution 2: Pretraining**
- "We propose two circular-aware pretraining objectives: CPCL (Circular Permutation Contrastive Learning) for learning rotational invariance, and BSJ_MLM for focused learning around back-splice junctions."

**Contribution 3: Empirical Validation**
- "Through extensive experiments, we demonstrate that CircMAC achieves state-of-the-art performance on circRNA-miRNA binding prediction and binding site localization."

---

## 2. Related Work

### 2.1 circRNA-miRNA Interaction Prediction

**Traditional Methods**:
- Sequence alignment: miRanda, RNAhybrid
- Thermodynamic stability calculation
- **Limitation**: Rule-based, cannot capture complex patterns

**Deep Learning Methods**:
- CNN-based: local pattern extraction
- RNN/LSTM-based: sequential modeling
- Transformer-based: global attention
- **Limitation**: All treat circRNA as linear sequence

**Key Sentence**:
- "While deep learning has advanced RNA interaction prediction, existing methods universally ignore the circular topology that fundamentally distinguishes circRNAs from linear RNAs."

---

### 2.2 RNA Language Models

**Existing Models**:
| Model | Pretraining | Data | Limitation |
|-------|-------------|------|------------|
| RNABERT | MLM | RNAcentral | Linear RNA focused |
| RNAFM | MLM + Structure | RNAcentral | No circular awareness |
| RNAErnie | MLM + Motif | RNAcentral | Linear sequences |
| RNAMSM | MSA-based | Rfam | Alignment-based |

**Key Sentence**:
- "Existing RNA language models are pretrained on predominantly linear RNA sequences with objectives designed for linear structures, making them suboptimal for circRNA representation."

---

### 2.3 State Space Models

**Mamba & SSMs**:
- Efficient long sequence modeling
- Linear complexity O(n)
- Strong sequential pattern capture

**Key Sentence**:
- "While state space models like Mamba offer efficient sequence modeling, they process sequences unidirectionally without explicit circular topology awareness."

---

## 3. Methods

### 3.1 Problem Formulation

**Input**:
- circRNA sequence: $c = (c_1, c_2, ..., c_L)$ where $c_L$ is adjacent to $c_1$
- miRNA sequence: $m = (m_1, m_2, ..., m_M)$

**Tasks**:
1. **Binding**: $f(c, m) \rightarrow \{0, 1\}$ — does circRNA bind to miRNA?
2. **Sites**: $g(c, m) \rightarrow \{0, 1\}^L$ — which positions are binding sites?
3. **Both**: Multi-task learning of binding + sites

**Key Insight**:
- "A circRNA sequence $(c_1, ..., c_L)$ is biologically equivalent to any circular permutation $(c_k, c_{k+1}, ..., c_L, c_1, ..., c_{k-1})$."

---

### 3.2 CircMAC Architecture

#### 3.2.1 Circular Relative Position Bias

**Idea**: 거리 계산 시 circular topology 고려

```
Standard:   d(i,j) = |i - j|
Circular:   d_circ(i,j) = min(|i-j|, L-|i-j|)
```

**Example** (L=100):
```
Position 0 and 99:
  Linear distance:   99 (far apart)
  Circular distance: 1  (adjacent!)
```

**Key Sentence**:
- "We modify the attention mechanism with circular relative position bias, ensuring that positions adjacent in the circular structure have small relative distances regardless of their linear indices."

---

#### 3.2.2 Circular CNN

**Idea**: Circular padding으로 BSJ 경계 제거

```
Standard padding:
  [0, 0, x₁, x₂, ..., x_L, 0, 0]
  → kernel cannot see across boundary

Circular padding:
  [x_{L-1}, x_L, x₁, x₂, ..., x_L, x₁, x₂]
  → kernel sees continuous circular context
```

**Key Sentence**:
- "Circular padding enables convolutional kernels to capture local patterns that span the back-splice junction, eliminating the artificial boundary present in standard padding."

---

#### 3.2.3 Multi-branch Fusion

**Architecture**:
```
Input x
    │
    ├──→ [Attention + Circular Bias] ──→ h_attn  (global patterns)
    │
    ├──→ [Mamba SSM] ──────────────────→ h_mamba (sequential patterns)
    │
    └──→ [Circular CNN] ───────────────→ h_cnn   (local motifs)
    │
    └──→ [Router] → weights
    │
    Output = Σ weights_i × h_i
```

**Why Multi-branch?**:
| Branch | Strength | Captures |
|--------|----------|----------|
| Attention | Global context | Long-range dependencies |
| Mamba | Sequential | Order-sensitive patterns |
| CNN | Local | Binding motifs, k-mers |

**Key Sentence**:
- "The three branches complement each other: Attention captures global dependencies with circular awareness, Mamba models sequential patterns efficiently, and Circular CNN extracts local binding motifs across the BSJ."

---

### 3.3 Circular-aware Pretraining

#### 3.3.1 CPCL (Circular Permutation Contrastive Learning)

**Key Insight**: circRNA의 모든 circular permutation은 동일한 분자

**Method**:
```
Original:    c = (A, C, G, U, A, G)
Permuted:    c' = (G, U, A, G, A, C)  ← circular shift by 2

Objective: repr(c) ≈ repr(c')
Loss: InfoNCE with (c, c') as positive pair
```

**Key Sentences**:
- "CPCL leverages the key insight that a circRNA is biologically identical to any of its circular permutations."
- "By learning representations invariant to circular shifts, the model captures the true circular nature of these molecules."

---

#### 3.3.2 BSJ_MLM (Back-Splice Junction focused MLM)

**Key Insight**: BSJ 영역은 circRNA에서 가장 중요한 구조적 특징

**Method**:
```
Standard MLM:  Random 15% masking across entire sequence
BSJ_MLM:       50% from BSJ region + 50% random

BSJ region: positions [0, 0.1L] ∪ [0.9L, L]
(First and last 10% of the sequence)
```

**Key Sentences**:
- "Standard MLM treats all positions equally, but the BSJ region is uniquely important for circRNA structure and function."
- "BSJ_MLM biases masking toward the junction region, forcing the model to learn the contextual relationships that define circular closure."

---

## 4. Experiments

### 4.1 Research Questions

| RQ | Question | Experiment |
|----|----------|------------|
| RQ1 | CircMAC가 다른 encoder보다 우수한가? | Exp 1 |
| RQ2 | Circular-aware pretraining이 효과적인가? | Exp 2 |
| RQ3 | 기존 pretrained RNA 모델 대비 성능은? | Exp 3 |
| RQ4 | 각 component의 기여도는? | Exp 4 |

### 4.2 Expected Results & Stories

**Story 1: CircMAC > Baselines**
- "CircMAC outperforms all baseline encoders by explicitly modeling circular topology."
- "The improvement is most pronounced in BSJ-spanning binding sites."

**Story 2: CPCL + BSJ_MLM Synergy**
- "CPCL alone provides moderate improvement by learning rotational invariance."
- "BSJ_MLM further enhances BSJ-region accuracy."
- "Combined, they achieve synergistic gains."

**Story 3: Domain-specific > General Pretrained**
- "CircMAC-PT trained on circRNA data outperforms RNA-FM/BERT trained on general RNAs."
- "This demonstrates the importance of domain-specific pretraining."

**Story 4: All Branches Contribute**
- "Removing any branch degrades performance, confirming complementary contributions."
- "Circular features (bias, padding) provide consistent improvements."

---

## 5. Discussion

### 5.1 Why Circular-awareness Matters

**Key Points**:
- BSJ 영역의 binding site 예측 정확도 향상
- Linear 모델은 BSJ 경계에서 성능 저하
- Circular 인식으로 이 문제 해결

**Key Sentence**:
- "Our analysis reveals that baseline models systematically underperform on binding sites spanning the BSJ, precisely where circular-aware modeling provides the greatest advantage."

---

### 5.2 Broader Impact

**For circRNA Research**:
- More accurate interaction prediction
- Better understanding of circRNA function

**For Methodology**:
- Template for circular biomolecule modeling
- Can extend to circular DNA, bacterial genomes

**Key Sentence**:
- "Our work establishes a paradigm for incorporating structural priors into deep learning models for biomolecules with non-linear topologies."

---

## 6. Conclusion

**Summary Key Sentence**:
- "We presented CircMAC, the first deep learning architecture that explicitly models the circular topology of circRNAs for miRNA interaction prediction."

**Technical Novelty**:
- Circular relative position bias
- Circular padding in CNN
- CPCL and BSJ_MLM pretraining

**Results**:
- State-of-the-art on binding prediction
- Superior binding site localization
- Particular improvements in BSJ regions

**Future Work**: 
- Extend to circRNA-protein interactions
- Apply to other circular biomolecules
- Investigate attention patterns for interpretability

---

## Key Phrases for Writing

### Novelty Claims (처음이라고 주장할 수 있는 것들)
- "first to explicitly model circular topology in circRNA representation"
- "novel circular-aware self-supervised objectives"
- "first multi-branch architecture with circular-specific adaptations"

### Comparison Phrases
- "In contrast to existing methods that treat circRNAs as linear..."
- "Unlike prior work, our approach explicitly accounts for..."
- "While previous methods ignore the circular nature..."

### Result Phrases (수치 채워넣기)
- "CircMAC achieves [X]% improvement in AUROC over the best baseline"
- "The improvement is most pronounced ([Y]%) in BSJ-spanning binding sites"
- "CPCL + BSJ_MLM pretraining provides [Z]% gain over MLM-only"

---

## Figure Ideas

### Figure 1: Motivation & Overview
- (a) circRNA structure with BSJ highlighted
- (b) Linear vs Circular distance visualization
- (c) CircMAC architecture overview

### Figure 2: Circular-aware Features
- (a) Circular relative position matrix
- (b) Circular vs standard padding comparison
- (c) CPCL concept illustration

### Figure 3: Main Results
- Bar charts comparing models
- ROC/PR curves

### Figure 4: Ablation & Analysis
- Component contribution
- BSJ region vs middle region accuracy

---

## Writing Checklist

- [ ] Introduction: 문제의 중요성 → 현재 한계 → 우리 해결책
- [ ] Related Work: 공정한 비교, 기존 연구의 gap 명시
- [ ] Methods: 수식 + 직관적 설명 + 그림
- [ ] Experiments: RQ 기반, 명확한 setup, 재현 가능
- [ ] Results: 수치 + 분석 + insight
- [ ] Discussion: Why it works, limitations, future work
- [ ] Conclusion: 간결한 요약

---

*Last updated: 2026-01-30*
