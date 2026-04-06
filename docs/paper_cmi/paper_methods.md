# CircMAC: Paper Methods Section
> 논문 Methods 섹션 — 바로 사용 가능한 영문 초안
> 각 서브섹션은 독립적으로 복사·수정해서 사용 가능

---

## 2. Methods

### 2.1 Task Formulation

We address the problem of predicting miRNA binding sites on circular RNA (circRNA) sequences at nucleotide resolution.
Given a circRNA sequence $\mathbf{c} = (c_1, c_2, \ldots, c_L)$ of length $L$ and a target miRNA sequence $\mathbf{m} = (m_1, m_2, \ldots, m_M)$, the model outputs a binary label for each nucleotide position:

$$\hat{\mathbf{y}} = (\hat{y}_1, \hat{y}_2, \ldots, \hat{y}_L), \quad \hat{y}_i \in \{0, 1\}$$

where $\hat{y}_i = 1$ indicates that position $i$ participates in miRNA binding.

A key distinction from linear RNA is that circRNA is formed by **back-splicing**, creating a covalent bond between the 3′ and 5′ ends at the **Back-Splice Junction (BSJ)**. Consequently, positions 1 and $L$ are spatially adjacent in the molecule, and binding motifs may span this junction — a topological property that linear sequence models cannot capture.

---

### 2.2 Dataset

#### 2.2.1 Binding Site Annotation

We constructed a dataset of experimentally supported circRNA–miRNA binding pairs from the **FL-circAS** database [CITE], which provides AGO-CLIP-validated binding site annotations at nucleotide resolution.

**Positive samples** were defined as pairs satisfying all of the following criteria:
- Miranda alignment score ≥ 155
- Binding free energy ≤ −20 kcal/mol
- Supported by ≥ 1 AGO-CLIP experiment

**Negative samples** were derived from RNAhybrid predictions for pairs not appearing in the positive set. Training negatives required MFE < −25 kcal/mol, *p*-value < 0.05, and alignment length ≥ 18 nt; test negatives used relaxed thresholds (MFE < −20, *p* < 0.1, length ≥ 18).

circRNA sequences were filtered to lengths between 150 and 1,000 nucleotides. The final dataset comprises **45,272 training pairs** (19,674 positive / 25,598 negative) and **17,708 test pairs** (7,838 / 9,870), split at the pair level to prevent data leakage.

**Binding site mask.** For each positive pair with multiple reported sites, individual binary masks (±2 nt bandwidth around each reported site) were merged via element-wise OR to produce a single binding site vector of length $L$.

#### 2.2.2 circRNA Sequences

circRNA isoform sequences were extracted from **circAtlas v3.0** [CITE] and **FL-circAS** [CITE]. Exon coordinates were mapped to the human reference genome (GRCh38/hg38), concatenated, and strand-corrected (reverse complement for minus-strand; T→U substitution). Only isoforms detected by both isoCirc and CIRI-long algorithms were retained, yielding 40,255 unique circRNA isoforms.

#### 2.2.3 Pretraining Dataset

For self-supervised pretraining, we predicted RNA secondary structures for 39,781 circRNA isoforms using **RNAsubopt** with the `--circ` flag (enabling circular RNA topology). For each isoform, 50 suboptimal structures were sampled, producing a pretraining corpus of **1,988,672 (circRNA, structure) pairs**. Each entry includes:
- Nucleotide sequence
- Dot-bracket secondary structure
- Base-pairing index for each position (`pairing[i] ∈ {-1, 0, ..., L-1}`, where -1 denotes unpaired)

---

### 2.3 CircMAC Architecture

We propose **CircMAC** (**Circ**ular-aware **M**ulti-branch **A**ttention and **C**onvolutional), a lightweight encoder (≈5.2M parameters) designed for circRNA-specific sequence modeling. The full model consists of four stages: (1) embedding and downsampling, (2) a stack of CircMACBlocks, (3) upsampling with skip connection, and (4) cross-attention interaction with miRNA followed by a site prediction head.

#### 2.3.1 Input Embedding

Both circRNA and miRNA sequences are tokenized using a **k-mer tokenizer** (k=1 by default, vocabulary size 500) and mapped to dense embeddings of dimension $d$ (default $d = 128$). The circRNA embedding is then downsampled by a factor of 2 via a stride-2 Conv1D layer to reduce computation for long sequences.

#### 2.3.2 CircMACBlock

Each CircMACBlock processes a sequence representation through three parallel branches followed by an adaptive router:

$$\mathbf{h} = \text{out\_proj}\!\left(\sum_{k \in \{\text{attn, mamba, cnn}\}} g_k \cdot \text{Branch}_k(\mathbf{x})\right) + \mathbf{x}$$

where $\mathbf{x} \in \mathbb{R}^{B \times L \times d}$ and the gates $\mathbf{g} \in \mathbb{R}^{B \times L \times 3}$ are computed dynamically per position by a two-layer MLP followed by softmax.

**Branch 1 — Attention with Circular Relative Position Bias.**
Multi-head self-attention captures long-range dependencies across the sequence. To reflect the circular topology of circRNA, we replace the standard linear relative position bias with a **circular distance bias**:

$$\text{score}(i, j) = \frac{Q_i K_j^\top}{\sqrt{d_h}} + b(i,j)$$

$$b(i,j) = -\alpha \cdot d_\text{circ}(i, j), \quad d_\text{circ}(i, j) = \min(|i - j|,\ L - |i - j|)$$

where $\alpha$ is a learnable slope and $d_\text{circ}$ is the circular minimum distance. This causes positions flanking the BSJ (positions 1 and $L$) to be treated as spatially close, enabling attention to span the junction.

**Branch 2 — Mamba (Selective State Space Model).**
We include a Mamba layer [CITE] for efficient long-range sequential modeling with linear complexity:

$$h_t = \mathbf{A} h_{t-1} + \mathbf{B} x_t, \quad y_t = \mathbf{C} h_t$$

where $\mathbf{A}, \mathbf{B}, \mathbf{C}$ are input-selective parameters. The causal, sequential structure of Mamba is complementary to Next Token Prediction (NTP) pretraining.

**Branch 3 — Depthwise CNN with Circular Padding.**
Local nucleotide patterns (e.g., seed match motifs) are captured by a depthwise Conv1D. To preserve continuity at the BSJ, we apply **circular padding** instead of zero-padding:

$$x_\text{padded} = [\ldots, x_{L-1}, x_L, x_1, x_2, \ldots, x_L, x_1, x_2, \ldots]$$

This ensures that motifs spanning the BSJ are extracted without boundary artifacts.

**Adaptive Router.**
The three branch outputs are combined via input-conditioned gating:

$$\mathbf{g} = \text{Softmax}(\text{MLP}([\bar{\mathbf{h}}_\text{attn};\ \bar{\mathbf{h}}_\text{mamba};\ \bar{\mathbf{h}}_\text{cnn}])) \in \mathbb{R}^{B \times L \times 3}$$

where $\bar{\mathbf{h}}$ denotes the mean-pooled branch output. The router learns to weight each branch adaptively for each position.

#### 2.3.3 Multi-Scale Processing

Following the HyMBA architecture [CITE], CircMAC processes sequences at half resolution:

$$\mathbf{h}_\downarrow = \text{Conv1D}_{\text{stride=2}}(\mathbf{e}_\text{circ}) \in \mathbb{R}^{B \times L/2 \times d}$$
$$\mathbf{H}_\downarrow = \text{CircMACBlock}^{(N)}(\mathbf{h}_\downarrow)$$
$$\mathbf{E}_\text{circ} = \text{Upsample}_{\times 2}(\mathbf{H}_\downarrow) + \mathbf{e}_\text{circ} \in \mathbb{R}^{B \times L \times d}$$

The skip connection preserves fine-grained sequence information while the downsampled path captures global context.

#### 2.3.4 Cross-Attention Interaction

After encoding both sequences, the model computes miRNA-conditioned circRNA representations via cross-attention, where the circRNA encoding serves as queries and the miRNA encoding as keys/values:

$$\mathbf{E}_\text{fused} = \text{Linear}\!\left([\mathbf{C};\, \mathbf{E}_\text{circ}]\right), \quad \mathbf{C} = \text{Softmax}\!\left(\frac{Q K^\top}{\sqrt{d}}\right) V$$

$$Q = \mathbf{E}_\text{circ} W^Q,\quad K = \mathbf{E}_\text{miRNA} W^K,\quad V = \mathbf{E}_\text{miRNA} W^V$$

This produces a position-wise representation of the circRNA conditioned on the specific miRNA partner.

#### 2.3.5 Site Prediction Head

The interaction features are refined by a multi-scale convolutional enhancer (parallel Conv1D with kernel sizes 3, 5, and 7, average-pooled and combined with a residual connection), followed by a two-layer Conv1D head:

$$\text{Conv1D}(d \to d/2,\ k=3) \to \text{BN} \to \text{GELU} \to \text{Dropout} \to \text{Conv1D}(d/2 \to 2,\ k=1)$$

The output logits $\hat{\mathbf{y}} \in \mathbb{R}^{B \times L \times 2}$ are trained with weighted cross-entropy loss.

#### 2.3.6 Model Configuration

| Hyperparameter | Value | Note |
|---|---|---|
| $d$ (d_model) | 128 | Hidden dimension |
| $N$ (n_layer) | 6 | Number of CircMACBlocks |
| $n_\text{heads}$ | 8 | Attention heads |
| $L_\text{max}$ | 1022 | Maximum circRNA length (nt) |
| CNN kernel size | 7 | Depthwise Conv1D |
| Mamba $d_\text{state}$ | 16 | SSM state dimension |
| Mamba expand | 2 | Channel expansion ratio |
| Total parameters | ~5.2M | |

---

### 2.4 Self-Supervised Pretraining

We pretrain CircMAC on unlabeled circRNA sequences with secondary structure annotations using five self-supervised objectives. These can be used individually or in combination.

#### 2.4.1 Masked Language Modeling (MLM)

Following BERT [CITE], we randomly mask 15% of nucleotide tokens and train the model to recover them from bidirectional context:

$$\mathcal{L}_\text{MLM} = -\sum_{i \in \mathcal{M}} \log P(c_i \mid \mathbf{c}_{\setminus \mathcal{M}})$$

where $\mathcal{M}$ is the set of masked positions.

#### 2.4.2 Next Token Prediction (NTP)

In the autoregressive formulation, the model predicts each token from its left context:

$$\mathcal{L}_\text{NTP} = -\sum_{t=1}^{L} \log P(c_t \mid c_1, \ldots, c_{t-1})$$

NTP is structurally aligned with the causal Mamba branch, encouraging the sequential SSM to model nucleotide transition patterns.

#### 2.4.3 Secondary Structure Prediction (SSP)

Using dot-bracket secondary structures predicted by **RNAsubopt --circ**, we train a binary classification head to predict whether each nucleotide is base-paired:

$$\mathcal{L}_\text{SSP} = -\sum_{i=1}^{L} \left[ s_i \log \hat{p}_i + (1 - s_i) \log (1 - \hat{p}_i) \right]$$

where $s_i \in \{0, 1\}$ is the binary structure label (1 = paired, 0 = unpaired).

#### 2.4.4 Base Pairing Matrix Reconstruction (Pairing)

A more detailed structural objective trains the model to reconstruct the full $L \times L$ pairing matrix, explicitly predicting which positions interact:

$$\mathcal{L}_\text{Pair} = \frac{1}{\|\mathbf{M}\|} \sum_{i,j} M_{ij} \cdot \text{BCE}(\hat{P}_{ij},\ P_{ij})$$

where $P_{ij} \in \{0,1\}$ indicates whether positions $i$ and $j$ are base-paired, $\hat{P}_{ij}$ is the predicted logit, and $M_{ij} = \mathbb{1}[\text{both } i,j \text{ are non-padding}]$ is a validity mask. This is implemented via a dedicated `CircularPairingHead` that operates on the upper triangular matrix to exploit the circular symmetry of RNA stem-loops.

#### 2.4.5 Circular Permutation Contrastive Learning (CPCL)

We introduce **CPCL**, a pretraining objective specific to circular RNA. Since circRNA is a covalently closed ring, the choice of BSJ as the "start" of the sequence is arbitrary — any rotation of the sequence represents the same molecule. CPCL treats two randomly rotated versions of the same circRNA as a positive pair:

$$\mathbf{z} = \text{MeanPool}(f_\theta(\mathbf{c})), \quad \mathbf{z}' = \text{MeanPool}(f_\theta(\text{Rotate}(\mathbf{c}, k)))$$

$$\mathcal{L}_\text{CPCL} = -\log \frac{\exp(\text{sim}(\mathbf{z}, \mathbf{z}') / \tau)}{\sum_{j=1}^{2B} \mathbb{1}[j \neq \text{pos}] \exp(\text{sim}(\mathbf{z}, \mathbf{z}_j) / \tau)}$$

where $\text{Rotate}(\mathbf{c}, k) = (c_{k+1}, \ldots, c_L, c_1, \ldots, c_k)$, $\text{sim}(\cdot)$ is cosine similarity, $\tau$ is a temperature parameter, and in-batch samples serve as negatives (NT-Xent loss [CITE]).

CPCL encourages the model to learn **rotation-invariant representations** of circRNA — representations that are independent of where the BSJ was placed during sequencing.

#### 2.4.6 Pretraining Setup

All objectives are trained on the 1.99M-sample pretraining corpus. Combined objectives jointly optimize the sum of applicable losses. We train with AdamW (lr = 1×10⁻³, weight decay = 0.01) for up to 1,000 epochs with early stopping (patience = 100) based on validation loss. Batch size is 192 for single-objective training.

| Objective | Target | circRNA-specific | Branch affinity |
|---|---|---|---|
| MLM | Masked tokens | ✗ | Attention, CNN |
| NTP | Next token | ✗ | Mamba |
| SSP | Binary structure | △ | Attention, CNN |
| Pairing | $L \times L$ matrix | △ | Attention |
| CPCL | Rotation invariance | ✅ | All branches |

△: Uses circRNA-specific secondary structure (RNAsubopt --circ), applicable to any RNA

---

### 2.5 Fine-Tuning

After pretraining, the CircMAC backbone weights are loaded and the full model (CircMAC + cross-attention + site head) is fine-tuned end-to-end on the labeled circRNA–miRNA dataset. The miRNA sequence is encoded by a frozen RNABERT [CITE] encoder.

Fine-tuning uses AdamW (lr = 1×10⁻⁴), batch size 32, for up to 150 epochs with early stopping (patience = 20). The site prediction loss is weighted cross-entropy accounting for class imbalance.

---

### 2.6 Baselines

**Encoder baselines (EXP1/EXP3).**
We compare CircMAC against four architectures trained from scratch with identical hyperparameters: LSTM, Transformer, Mamba, and HyMBA (Attention+Mamba). We also evaluate four pretrained RNA language models — RNABERT (113M params), RNAErnie (86M), RNA-FM (95M), RNA-MSM (96M) — in both frozen and fine-tuned modes, at their maximum supported sequence length.

**Ablation study (EXP4).**
To validate each component of CircMAC, we ablate: removing individual branches (`--no_attn`, `--no_mamba`, `--no_conv`), single-branch variants (`attn_only`, `mamba_only`, `cnn_only`), and removing the circular relative bias (`--no_circular_rel_bias`).

**Interaction mechanism (EXP5).**
Three circRNA–miRNA interaction strategies are compared: concatenation, element-wise product, and cross-attention (ours).

**Site head (EXP6).**
Two decoding heads are compared: a multi-scale Conv1D head (ours) versus a simple linear projection.

---

### 2.7 Evaluation Metric

All models are evaluated on the held-out test set. The primary metric is **site-level macro F1 score**, computed as the harmonic mean of precision and recall over nucleotide-level binary predictions after thresholding the softmax output at 0.5.

---

## Key Numbers for Abstract / Introduction

| Finding | Value |
|---|---|
| CircMAC parameters | ~5.2M |
| RNA-FM/MSM parameters | ~95–96M |
| CircMAC (scratch) F1 | **0.7400** |
| Best RNA-FM trainable F1 | 0.6283 (fair) / 0.6377 (trainable) |
| Best pretraining gain (Pairing) | 0.7494 → **0.7682** (+1.9 pp) |
| Ablation: removing Mamba | 0.7397 → 0.6766 (−6.3 pp) |
| Ablation: removing Conv | 0.7397 → 0.6861 (−5.4 pp) |
| Cross-attention vs concat | 0.7413 vs 0.7142 (+2.7 pp) |
| Conv1D head vs linear | 0.7408 vs 0.7328 (+0.8 pp) |
