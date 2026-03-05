# CircMAC: Method & Architecture

> 논문 방법론 섹션 초안 (Korean/English mixed)

---

## 1. Problem Formulation

**Input**:
- circRNA sequence: $\mathbf{c} = (c_1, c_2, \ldots, c_L)$, where $L \leq 1022$
- miRNA sequence: $\mathbf{m} = (m_1, m_2, \ldots, m_M)$, where $M \leq 25$

**Output**:
- Site labels: $\hat{\mathbf{y}} = (\hat{y}_1, \ldots, \hat{y}_L)$, $\hat{y}_i \in \{0, 1\}$
  (각 nucleotide가 binding site인지 여부)

**Key difference from linear RNA**:
circRNA는 Back-Splice Junction(BSJ)으로 인해 양 끝이 연결된 **원형 구조**를 가짐.
→ 위치 $1$과 위치 $L$은 실제로 인접 (linear 모델은 이를 무시).

---

## 2. Overall Framework

```
┌────────────────────────────────────────────────────────────────────┐
│                      CMI-MAC Framework                             │
│                                                                    │
│  circRNA sequence (c_1,...,c_L)       miRNA sequence (m_1,...,m_M) │
│         │                                      │                   │
│         ▼                                      ▼                   │
│  ┌─────────────┐                      ┌─────────────────┐          │
│  │   CircMAC   │                      │  RNABERT        │          │
│  │  (Encoder)  │                      │  (Target Enc.)  │          │
│  │  Circular-  │                      │  [frozen]       │          │
│  │   aware     │                      └────────┬────────┘          │
│  └──────┬──────┘                               │                   │
│         │ E_circ [B,L,D]                       │ E_mirna [B,M,D]   │
│         └──────────────┬────────────────────────┘                   │
│                        ▼                                            │
│              ┌──────────────────┐                                   │
│              │ Cross-Attention  │  (circRNA queries miRNA)          │
│              │  Interaction     │                                   │
│              └────────┬─────────┘                                   │
│                       │ E_fused [B,L,D]                             │
│                       ▼                                             │
│              ┌──────────────────┐                                   │
│              │    Site Head     │  (Conv1D multi-scale)             │
│              │ [B,L,D] → [B,L,2]│                                  │
│              └────────┬─────────┘                                   │
│                       │                                             │
│                  ŷ = (ŷ_1,...,ŷ_L)                                 │
│                  per-position binary labels                         │
└────────────────────────────────────────────────────────────────────┘
```

---

## 3. CircMAC Encoder

### 3.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    CircMAC Encoder                              │
│                                                                 │
│  Token IDs [B, L]                                               │
│       │                                                         │
│       ▼                                                         │
│  Embedding [B, L, D]                                            │
│       │                                                         │
│       ├──────────────────────── skip ──────────────────────┐    │
│       │                                                     │    │
│       ▼                                                     │    │
│  Downsample (stride-2 depthwise conv)                       │    │
│  [B, L, D] → [B, L/2, D]          ← Multi-scale            │    │
│       │                                                     │    │
│       ▼                                                     │    │
│  ┌─────────────────────────────────────┐                    │    │
│  │      CircMACBlock × N layers        │                    │    │
│  │  ┌─────────────────────────────┐    │                    │    │
│  │  │        RMSNorm              │    │                    │    │
│  │  │           │                 │    │                    │    │
│  │  │    ┌──────┴──────┐          │    │                    │    │
│  │  │    │  in_proj    │          │    │                    │    │
│  │  │    │(Q,K,V,base) │          │    │                    │    │
│  │  │    └──┬───┬───┬──┘          │    │                    │    │
│  │  │  Q,K,V│   │base│base        │    │                    │    │
│  │  │       │   │    │            │    │                    │    │
│  │  │       ▼   ▼    ▼            │    │                    │    │
│  │  │  ┌────┐ ┌────┐ ┌─────┐     │    │                    │    │
│  │  │  │Attn│ │Mamb│ │ CNN │     │    │                    │    │
│  │  │  │+⊕  │ │    │ │circ.│     │    │                    │    │
│  │  │  └──┬─┘ └──┬─┘ └──┬──┘     │    │                    │    │
│  │  │     └──────┴───────┘        │    │                    │    │
│  │  │           │                 │    │                    │    │
│  │  │        Router               │    │                    │    │
│  │  │     (gate_1, gate_2, gate_3)│    │                    │    │
│  │  │           │                 │    │                    │    │
│  │  │      Weighted Sum           │    │                    │    │
│  │  │           │                 │    │                    │    │
│  │  │       out_proj              │    │                    │    │
│  │  │           │                 │    │                    │    │
│  │  │    + residual               │    │                    │    │
│  │  └─────────────────────────────┘    │                    │    │
│  └─────────────────────────────────────┘                    │    │
│       │                                                     │    │
│       ▼                                                     │    │
│  Upsample (linear interp.)                                  │    │
│  [B, L/2, D] → [B, L, D]                                   │    │
│       │                                                     │    │
│       └──────────────── + ──────────────────────────────────┘    │
│                         │                                         │
│                   E_circ [B, L, D]                                │
└─────────────────────────────────────────────────────────────────┘
```

**Hyperparameters** (실험 설정):
| Parameter | Value |
|-----------|-------|
| `d_model` D | 128 |
| `n_layer` N | 6 |
| `n_heads` H | 8 |
| `d_state` | 16 (Mamba state) |
| `d_conv` | 4 (Mamba conv) |
| `expand` | 2 (Mamba d_inner = 256) |
| `conv_kernel_size` | 7 (CNN) |

---

### 3.2 Attention Branch (Circular-aware)

```
Q [B, L/2, D] ──┐
K [B, L/2, D] ──┤──→  Attention Scores [B, H, L/2, L/2]
                 │           │
                 │    + Circular Relative Bias
                 │           │
                 │    ────────────────────────────────────
                 │    d_circular(i,j) = min(|i-j|, L-|i-j|)
                 │    bias(i,j) = -slope × d_circular(i,j)
                 │    ────────────────────────────────────
                 │           │
                 │        Softmax
                 │           │
V [B, L/2, D] ──┘──→  Context [B, L/2, D]
```

**Circular Relative Bias**:
- 기존 bias: $d(i,j) = |i-j|$ → 양 끝 멀다고 판단 (linear 가정)
- CircMAC bias: $d_{\text{circ}}(i,j) = \min(|i-j|,\; L - |i-j|)$ → **원형 거리** 반영
- BSJ 근방 (position 1 ↔ position L) 간 attention 강화

---

### 3.3 Mamba Branch (Sequential)

```
base [B, L/2, D]
      │
      ▼
  Mamba SSM
  - d_state = 16 (selective state)
  - d_conv  = 4  (local conv inside Mamba)
  - d_inner = D × expand = 256 (expanded dimension)
      │
      ▼
 output [B, L/2, D]
```

- Selective State Space Model로 long-range sequential dependency 포착
- 선형 시간 복잡도 O(L)

---

### 3.4 Circular CNN Branch

```
base [B, L/2, D]
      │
      ▼  Transpose [B, D, L/2]
      │
 Circular Padding
 ┌──── pad_size = (kernel_size-1)//2 = 3 ────┐
 │  [last 3 tokens] + [sequence] + [first 3] │
 │  → wraps BSJ: end tokens attend to start! │
 └────────────────────────────────────────────┘
      │  [B, D, L/2 + 6]
      ▼
 Depthwise Conv1d (kernel=7, groups=D)
      │  [B, D, L/2]
      ▼  Transpose [B, L/2, D]
 output [B, L/2, D]
```

- **Circular padding**: 시퀀스 끝과 시작을 이어붙여 BSJ의 연속성 표현
- Depthwise conv: 채널별 독립 처리, 파라미터 효율적

---

### 3.5 Router (Adaptive Branch Fusion)

```
attn_out  [B, L/2, D] ─┐
mamba_out [B, L/2, D] ─┼→  concat → [B, L/2, 3D]
cnn_out   [B, L/2, D] ─┘                │
                                    MeanPool (over L)
                                         │  [B, 3D]
                                    Linear(3D → D)
                                    GELU
                                    Linear(D → 3)
                                    Softmax
                                         │
                               (gate_1, gate_2, gate_3) [B, 3]
                                         │  unsqueeze → [B, 1, 3]
                                         ▼
              output = gate_1 * attn_out + gate_2 * mamba_out + gate_3 * cnn_out
                     → [B, L/2, D]
```

- Position별 고정 weight가 아닌 **sequence-level adaptive gating**
- 각 입력마다 세 branch의 기여도를 자동으로 결정

---

### 3.6 Multi-Scale Architecture (Global)

```
Input [B, L, D]
      │
      ├────────────────── skip connection ──────────────────────┐
      │                                                          │
  Downsample                                                     │
  Conv1d(stride=2) → [B, L/2, D]                                │
      │                                                          │
  N × CircMACBlock (at L/2 resolution)                          │
      │                                                          │
  Upsample (linear interp.) → [B, L, D]                         │
      │                                                          │
      └───────────────────── + ──────────────────────────────────┘
                             │
                       [B, L, D]
```

- 절반 해상도에서 학습: 계산량 감소, 더 넓은 receptive field
- Skip connection: 원본 해상도 정보 보존

---

## 4. circRNA-miRNA Interaction: Cross-Attention

```
E_circ [B, L, D] ──→  Q_proj → Q [B, H, L, d]
                                       │
E_mirna [B, M, D] ─→  K_proj → K [B, H, M, d]  ─→  Q @ K^T → [B, H, L, M]
                    └→ V_proj → V [B, H, M, d]          │
                                                     Softmax
                                                         │
                                                   attn @ V → [B, H, L, d]
                                                         │
                                                    Reshape [B, L, D]
                                                         │
                                            Concat([E_circ, context]) [B, L, 2D]
                                                         │
                                            fusion_proj [2D → D] → [B, L, D]
```

- circRNA의 각 position이 miRNA 전체에 attend
- **miRNA 정보가 circRNA 표현에 통합**됨
- 비대칭 길이 허용 (L ≫ M)

---

## 5. Site Prediction Head

```
E_fused [B, L, D]
      │
      ▼
MultiKernelCNN (Feature Enhancer)
  ├── Conv1d(D→D, kernel=3, padding=1) ─┐
  ├── Conv1d(D→D, kernel=5, padding=2) ─┼→ Mean → [B, L, D]
  └── Conv1d(D→D, kernel=7, padding=3) ─┘
  + Residual + LayerNorm
      │  [B, L, D]
      ▼  Permute [B, D, L]
Conv1d(D → D/2, kernel=3, padding=1)
BatchNorm1d → GELU → Dropout(0.1)
      │  [B, D/2, L]
Conv1d(D/2 → 2, kernel=1)
      │  [B, 2, L]
      ▼  Permute [B, L, 2]
  logits [B, L, 2]
      │
  Softmax → ŷ_i = P(site | c_i, m)
```

- **Multi-scale local context** (kernel 3, 5, 7 평균): 다양한 범위의 binding pattern 포착
- 1D conv로 token-level (per-position) 예측

---

## 6. Pretraining (Self-Supervised, Exp2)

```
┌──────────────────────────────────────────────────────────────────┐
│              CircMAC Pretraining on circRNA data                 │
│                                                                  │
│  circRNA [B, L]  (df_circ_ss_5, ~149K sequences)                │
│       │                                                          │
│       ▼                                                          │
│  CircMAC Backbone                                                │
│  (same architecture, no interaction head)                        │
│       │                                                          │
│  E [B, L, D]                                                     │
│       │                                                          │
│  ┌────┴────┬──────────┬───────────┬──────────┐                  │
│  │   MLM   │   NTP    │   CPCL    │  Pairing │                  │
│  │(15% mask│(next tok │(circular  │(base pair│                  │
│  │ predict)│ predict) │ perm. CL) │ matrix)  │                  │
│  └────┬────┴──────────┴───────────┴──────────┘                  │
│       │                                                          │
│   UncertaintyWeightingLoss (multi-task)                          │
│   L = Σ exp(-log σ_k²) L_k + log σ_k²                          │
└──────────────────────────────────────────────────────────────────┘
```

### Pretraining Tasks

| Task | Description | circRNA-specific? |
|------|-------------|:-----------------:|
| **MLM** | 15% token 무작위 마스킹 후 복원 | No |
| **NTP** | 다음 토큰 예측 (auto-regressive) | No |
| **SSP** | 이차구조 dot-bracket 토큰 예측 | No |
| **Pairing** | 염기쌍 행렬 예측 (CircularPairingHead) | Partial |
| **CPCL** | Circular Permutation Contrastive Learning | **Yes** |

### CPCL (Circular Permutation Contrastive Learning)

```
원본 circRNA: c = [c_1, c_2, ..., c_k, c_{k+1}, ..., c_L]
                                   ↑ random cut point

Positive pair:
  view_1 = [c_1,   ..., c_k,   c_{k+1}, ..., c_L]  (원본)
  view_2 = [c_{k+1},..., c_L,  c_1,     ..., c_k]  (circular permutation)

→ 같은 circRNA의 다른 rotation → positive pair (동일 정체성)
→ 다른 circRNA → negative pair

CircMAC은 circular permutation에 invariant해야 함
→ BSJ-aware representation 학습
```

---

## 7. 전체 파라미터 수 비교 (d_model=128, n_layer=6)

| Model | Params | Training Mode | Data Coverage |
|-------|:------:|:-------------:|:-------------:|
| RNABERT | ~0.5M | frozen / trainable | 37.6% |
| RNAErnie | ~86M | frozen / trainable | 48.2% |
| RNA-FM | ~95M | frozen / trainable | 100% |
| RNA-MSM | ~96M | frozen / trainable | 100% |
| **CircMAC-PT** | **~3M** | **fine-tuned** | **100%** |

---

## 8. Key Design Choices (논문 어필 포인트)

| Component | 이유 |
|-----------|------|
| **Circular Relative Bias** | linear bias는 BSJ 무시 → circular distance로 BSJ 인접 표현 |
| **Circular CNN Padding** | CNN 경계 효과 제거 → BSJ 연속성 모델링 |
| **CPCL Pretraining** | circRNA는 rotation-invariant → circular permutation으로 자기지도학습 |
| **Multi-scale** | 계산 효율 + 넓은 receptive field |
| **3-branch Router** | 각 입력별 adaptive fusion → 일반화 성능 향상 |
| **Cross-attention** | circRNA 각 위치가 miRNA에 직접 attend → 정밀한 binding 예측 |
