# CircMAC — Methods Technical Details
> 코드에서 직접 추출한 정확한 수치들. Methods 작성용.

---

## 1. Data & Tokenization

### circRNA
| Item | Value |
|------|-------|
| Tokenization | KmerTokenizer, **k = 1** (single nucleotide) |
| Vocabulary | `<pad>=0, <cls>=1, <eos>=2, <unk>=3, <mask>=4, <null>=5, A=6, C=7, G=8, U=9, N=10` |
| vocab_size | **11** |
| max_len (CircMAC) | **1022** nt (covers 100% of dataset) |
| max_len (RNABERT) | 440 (37.6% coverage) |
| max_len (RNAErnie) | 510 (48.2% coverage) |
| max_len (RNA-FM/MSM) | 1024 (100% coverage) |

### miRNA
| Item | Value |
|------|-------|
| max_len | **25** nt |
| Encoder | RNABERT (frozen, pretrained) |

### Dataset split
| Split | File |
|-------|------|
| Train | `data/df_train_final.pkl` |
| Test  | `data/df_test_final.pkl` |
| Pretrain | `df_circ_ss_5` (~149K unlabeled circRNA sequences) |

---

## 2. CircMAC Architecture (Experiment Default)

### Global hyperparameters
| Parameter | Value |
|-----------|-------|
| `d_model` (D) | **128** |
| `n_layer` (N) | **6** |
| `n_heads` (H) | **8** |
| `head_dim` (d) | 128 / 8 = **16** |

### Multi-scale (Downsample / Upsample)
| Component | Detail |
|-----------|--------|
| Downsample | `Conv1d(D, D, kernel=3, stride=2, groups=D)` + `LayerNorm` |
| Upsample | `F.interpolate(scale_factor=2, mode='linear')` + `Conv1d(D, D, kernel=1)` + `LayerNorm` |
| 연산 해상도 | **L/2** (blocks 내부) |
| Skip connection | 원본 `[B, L, D]`를 upsample 결과에 **element-wise 덧셈** |

### Per-block Input Projection
```
in_proj: Linear(D → 4D)
  split → Q [B, L/2, D]    (attention query)
          K [B, L/2, D]    (attention key)
          V [B, L/2, D]    (attention value)
          base [B, L/2, D] (Mamba + CNN 입력)
```

---

## 3. Three Branches

### Branch 1 — Attention (Circular-aware)
| Parameter | Value |
|-----------|-------|
| Type | Multi-head self-attention |
| Heads (H) | **8** |
| Head dim (d) | **16** |
| Normalization | **RMSNorm** (after attention) |
| Position bias | Circular relative bias (아래 수식) |

**Circular relative bias:**
```
d_circ(i, j) = min(|i - j|, L/2 - |i - j|)   # L/2: 연산 해상도
bias(i, j)   = -1.0 × d_circ(i, j)            # slope = 1.0 (fixed)

attn_score(i,j) = (Q_i · K_j) / sqrt(d) + bias(i, j)
```
- slope는 **고정값 1.0** (learnable 아님)
- BSJ 근방 position끼리 더 높은 attention score → BSJ 연속성 표현

---

### Branch 2 — Mamba (Sequential SSM)
| Parameter | Value |
|-----------|-------|
| Type | Mamba selective SSM |
| `d_model` | **128** |
| `d_state` | **16** (hidden state dimension N) |
| `d_conv` | **4** (Mamba 내부 local conv kernel) |
| `expand` | **2** → `d_inner = 128 × 2 = 256` |
| Normalization | **RMSNorm** (after Mamba) |
| Input | `base [B, L/2, D]` |
| Complexity | **O(L)** (linear) |

---

### Branch 3 — Circular CNN (Local patterns)
| Parameter | Value |
|-----------|-------|
| Type | Depthwise Conv1d |
| `kernel_size` | **7** |
| `pad_size` | (7-1)/2 = **3** |
| `groups` | D (depthwise, channel-independent) |
| Padding mode | **`F.pad(mode='circular')`** |
| Normalization | **RMSNorm** (after CNN) |
| Input | `base [B, L/2, D]` |

**Circular padding 동작:**
```
원본:  [..., x_{L-3}, x_{L-2}, x_{L-1}, x_L, x_1, x_2, x_3, ...]
패딩후: [x_{L-2}, x_{L-1}, x_L, | x_1, ..., x_L, | x_1, x_2, x_3]
                                    ↑ 실제 시퀀스 ↑
→ kernel이 BSJ 경계를 이어서 볼 수 있음
```

---

## 4. Router (Adaptive Branch Fusion)
| Parameter | Value |
|-----------|-------|
| Type | Token-level MLP gating |
| Input | 각 branch output의 mean → concat: `[B, L, n_branches × D]` |
| Hidden | `Linear(n_branches × D → D)` + `GELU` |
| Output | `Linear(D → n_branches)` + `Softmax` → gates `[B, L, n_branches]` |
| Fusion | `output = Σ gate_k × branch_k` (token-wise weighted sum) |

> **주의:** Router는 **token-level** (position-wise)로 동작.
> 즉, 같은 시퀀스 내에서 position마다 다른 branch 가중치가 부여됨.

```
실제 코드:
  router_input = cat([b.mean(dim=1, keepdim=True).expand(-1, L, -1)
                      for b in branches], dim=-1)  # [B, L, n*D]
  gates = self.router(router_input)                # [B, L, n]
  fused = Σ gates[..., k:k+1] * branch_k          # [B, L, D]
```

**Output projection:**
```
out_proj: Linear(D → D)
+ residual connection from block input
```

---

## 5. circRNA–miRNA Cross-Attention Interaction
| Parameter | Value |
|-----------|-------|
| Query | `E_circ [B, L, D]` (circRNA) |
| Key, Value | `E_mirna [B, M, D]` (miRNA, M≤25) |
| Heads | **1** (`cross_attention_multihead=1`, config 기본값) |
| Post-fusion | `concat([E_circ, context]) [B, L, 2D]` → `Linear(2D → D)` |

---

## 6. Site Prediction Head
```
E_fused [B, L, D]
   │
MultiKernelCNN:
   ├── Conv1d(D→D, kernel=3, padding=1)
   ├── Conv1d(D→D, kernel=5, padding=2)   → mean → [B, L, D]
   └── Conv1d(D→D, kernel=7, padding=3)
   + residual + LayerNorm
   │
   Permute [B, D, L]
   │
Conv1d(D → D/2, kernel=3, padding=1)    # D/2 = 64
BatchNorm1d → GELU → Dropout(0.1)
   │
Conv1d(D/2 → 2, kernel=1)
   │
   Permute [B, L, 2]
   │
logits [B, L, 2]  →  softmax → per-position P(site)
```

---

## 7. Pretraining Objectives

### MLM (Masked Language Modeling)
| Parameter | Value |
|-----------|-------|
| Masking ratio | **15%** (random uniform) |
| Mask token id | **4** (`<mask>`) |
| Loss | Cross-entropy (masked positions only) |

### CPCL (Circular Permutation Contrastive Learning)
| Parameter | Value |
|-----------|-------|
| Augmentation | Random circular shift: `shift ~ Uniform(1, L-1)` (per sample) |
| Positive pair | `(original, shifted)` → 같은 circRNA의 다른 rotation |
| Negative pairs | 같은 batch 내 다른 샘플 |
| Pooling | Mean pooling over L → `[B, D]` |
| Projection head | `Linear(D → D)` (no hidden layer) |
| Normalization | L2 normalize (F.normalize) |
| Loss | **InfoNCE** (= NT-Xent via cross_entropy) |
| Temperature (τ) | **0.1** |

```
z1 = normalize(Linear(mean(backbone(x))))         # [B, D]
z2 = normalize(Linear(mean(backbone(x_shifted)))) # [B, D]
logits = z1 @ z2.T / 0.1                          # [B, B]
loss   = cross_entropy(logits, arange(B))
```

### BSJ-MLM (Back-Splice Junction focused MLM)
| Parameter | Value |
|-----------|-------|
| Total masking ratio | **15%** of L |
| BSJ focus ratio | **50%** of masks → BSJ region |
| Random ratio | **50%** of masks → rest of sequence |
| BSJ region | First and last **10%** of sequence (min 5 tokens) |
| Loss | Cross-entropy (masked positions only) |

```
n_mask     = int(L × 0.15)
n_bsj_mask = int(n_mask × 0.5)    # ~7.5% from BSJ ends
n_rand     = n_mask - n_bsj_mask   # ~7.5% random

BSJ region = positions [1, ⌊0.1L⌋] ∪ [⌊0.9L⌋, L]
```

### Multi-task Loss (UncertaintyWeightingLoss)
| Parameter | Value |
|-----------|-------|
| Formula | `L = Σ_k exp(-log σ_k²) × L_k + log σ_k²` |
| `log σ_k²` | Learnable parameter per task (init = 0) |
| Note | Total loss can be **negative** (정상 동작) |

---

## 8. Training Configuration

### Fine-tuning (Exp 1, 3, 4, 5, 6)
| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | **1e-4** |
| Weight decay | 0.01 |
| Batch size | **128** (CircMAC from scratch) |
|             | **32** (pretrained model comparison) |
| Epochs | **150** |
| Early stopping | **20** epochs |
| Grad clipping | 1.0 |

### Pretraining (Exp 2)
| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | **5e-4** |
| Weight decay | **0.01** |
| Batch size | **64** |
| Epochs | **200** |
| Early stopping | **30** epochs |

### Seeds & Reporting
- Seeds: **1, 2, 3**
- Reported: **mean ± std**

---

## 9. Model Size Summary

| Model | Params | Notes |
|-------|:------:|-------|
| RNABERT | ~0.5M | Positional enc. max 440 |
| RNAErnie | ~86M | Positional enc. max 510 |
| RNA-FM | ~95M | max 1024 |
| RNA-MSM | ~96M | max 1024 |
| **CircMAC (ours)** | **~3M** | max 1022, circular-aware |

---

## 10. Key Design Choices (논문 어필용)

| Component | 기존 방법 | CircMAC | 효과 |
|-----------|----------|---------|------|
| Position bias | `d(i,j) = \|i-j\|` (linear) | `d_circ(i,j) = min(\|i-j\|, L-\|i-j\|)` | BSJ 양 끝 위치가 가깝게 인식됨 |
| CNN padding | zero padding | circular padding (F.pad mode='circular') | BSJ 경계 연속성 표현 |
| Branch fusion | fixed equal weight | adaptive token-level router | 입력별 최적 branch 조합 |
| Pretraining | MLM only (일반 RNA LM) | MLM + CPCL (circular 전용) | circRNA rotation invariance 학습 |
| Sequence coverage | 37-48% (RNABERT/Ernie) | **100%** | 전체 데이터 활용 가능 |
