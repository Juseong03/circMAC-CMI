# CircMAC Architecture

## Overview

**CircMAC** (**Circ**RNA **M**ulti-branch **A**ttention and **C**onvolutional architecture)는 circRNA–miRNA 결합 위치 예측을 위해 설계된 딥러닝 모델이다. circRNA의 원형 위상(circular topology)을 명시적으로 모델링하기 위해 세 가지 핵심 설계 원칙을 따른다.

1. **3-branch hybrid encoder**: Attention + Mamba + Circular CNN의 상호보완적 조합
2. **Circular-aware components**: 원형 위치 편향(bias)과 원형 패딩(padding)
3. **Cross-attention interaction**: circRNA와 miRNA 표현의 효과적인 결합

---

## 전체 파이프라인

```
circRNA 서열                    miRNA 서열
    │                               │
    ▼                               ▼
K-mer Tokenizer                 K-mer Tokenizer
    + Embedding [B, L, D]           + Embedding [B, M, D]
    │                               │
    ▼                               │
Downsample ×½ (Conv1D stride=2)    │
    │                               │
    ▼                               │
┌─────────────────────┐            │
│   CircMACBlock × N  │            │
│  (default N = 6)    │            │
└─────────────────────┘            │
    │                               │
    ▼                               │
Upsample ×2 + Skip connection      │
    │                               │
    ▼                               ▼
E_circ [B, L, D]          E_mirna [B, M, D]
    │                               │
    └──────────────┬────────────────┘
                   ▼
       Cross-Attention  (Q: E_circ, K/V: E_mirna)
                   │
                   ▼
      MultiKernelCNN Feature Enhancer (k=3,5,7)
                   │
                   ▼
         Conv1D Site Head
                   │
                   ▼
    Binding Site Prediction [B, L, 2]
```

---

## CircMACBlock

각 CircMACBlock은 세 개의 병렬 브랜치와 Adaptive Router로 구성된다.

```
Input x  [B, L, D]
    │
    ▼
in_proj  (Linear: D → 4D)
    │
    ├──────────── Q, K, V ──────────────┐
    │                                   │
    │             base                  │
    │               │                   │
    ▼               ▼              ▼    │
┌────────┐    ┌──────────┐   ┌─────────┐
│ Attn   │    │  Mamba   │   │  Circ-  │
│ Branch │    │  Branch  │   │   CNN   │
└────────┘    └──────────┘   └─────────┘
    │               │              │
 RMSNorm        RMSNorm        RMSNorm
    │               │              │
    └───────────────┼──────────────┘
                    ▼
           Adaptive Router
      gates = Softmax(Linear([mean(attn),
                              mean(mamba),
                              mean(cnn)]))
                    │
              weighted sum
                    │
                    ▼
            out_proj  (D → D)
                    │
            + residual (x)
                    │
                    ▼
            Output [B, L, D]
```

### Branch 1 — Attention (Multi-Head Self-Attention + Circular Bias)

circRNA 전체 서열에 걸친 **장거리 의존성**을 모델링한다. 표준 attention에 **circular relative position bias**를 추가하여 원형 위상을 반영한다.

```
Attention score(i, j) = QᵢKⱼᵀ / √d + bias(i, j)

bias(i, j) = −slope × d_circular(i, j)
d_circular(i, j) = min(|i−j|, L−|i−j|)
```

- 선형 거리 `|i−j|` 대신 **원형 최소 거리** 사용
- BSJ 양끝 위치(pos 0, pos L−1)가 서로 가깝게 인식됨
- 파라미터: `n_heads=8`, `head_dim = d_model / n_heads`

### Branch 2 — Mamba (Selective State Space Model)

**순차적 패턴**을 causal 방향으로 학습한다. Structured SSM 기반으로 장거리 순차 의존성을 선형 복잡도로 처리한다.

```
선택적 상태 공간 모델:
  h_t = A·h_{t−1} + B·x_t
  y_t = C·h_t

파라미터: d_state=16, d_conv=4, expand=2
```

- NTP(Next Token Prediction) 사전학습과 방향이 일치
- 긴 circRNA 서열(최대 1022 nt)에서도 효율적

### Branch 3 — Circular CNN (Depthwise + Circular Padding)

**BSJ 경계를 포함한 국소 모티프**를 추출한다. 일반 zero-padding 대신 **circular padding**을 사용하여 서열 끝과 시작이 연결된 것처럼 처리한다.

```
Zero padding (기존):   [0, 0, A, U, G, C, A, U, 0, 0]
Circular padding:     [A, U, A, U, G, C, A, U, A, U]
                       ↑↑ 끝부분이 앞으로        ↑↑ 시작이 뒤로
```

- Depthwise Conv1D: `groups=d_model`, `kernel_size=7`
- BSJ에 걸친 결합 모티프도 온전하게 포착 가능

### Adaptive Router

세 브랜치의 출력을 **위치별로 동적**으로 가중 합산한다.

```python
router_input = concat([mean(attn_out), mean(mamba_out), mean(cnn_out)])
               # [B, L, 3D]
gates = Softmax(Linear(Linear(router_input)))
        # [B, L, 3]
output = Σ gates[..., i] * branch_out[i]
```

---

## Multiscale (Down/Up Sampling)

HyMBA 스타일의 전역 다운샘플링을 사용해 **긴 서열에서의 계산 효율**을 높이고 전역 컨텍스트를 압축한다.

```
원본 길이 L
    │  Conv1D(stride=2)
    ▼
길이 L/2   → N × CircMACBlock
    │  Interpolate × 2
    ▼
길이 L    + Skip connection (원본)
```

---

## Interaction: Cross-Attention

circRNA 표현 `E_circ`가 Query, miRNA 표현 `E_mirna`가 Key/Value가 되어 **miRNA 서열을 조건으로 한 circRNA 위치별 표현**을 생성한다.

```
Q = E_circ  [B, L_circ, D]
K = E_mirna [B, L_mirna, D]
V = E_mirna [B, L_mirna, D]

context = Softmax(QKᵀ/√d) · V  [B, L_circ, D]
output  = concat(context, Q) → Linear → [B, L_circ, D]
```

---

## Site Prediction Head

**MultiKernelCNN**으로 상호작용 특징을 강화한 후 **Conv1D Head**로 위치별 예측을 수행한다.

```
interaction features [B, L, D]
    │
MultiKernelCNN (k=3, 5, 7 병렬 → 평균 + Residual + Norm)
    │
Conv1D(D→D/2, k=3) → BN → GELU → Dropout
    │
Conv1D(D/2→2, k=1)
    │
per-position logits [B, L, 2]
```

---

## 하이퍼파라미터 요약

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `d_model` | 128 | 모델 차원 |
| `n_layer` | 6 | CircMACBlock 레이어 수 |
| `n_heads` | 8 | Attention 헤드 수 |
| `max_len` | 1022 | 최대 입력 길이 (nt) |
| `conv_kernel_size` | 7 | CNN 커널 크기 |
| `d_state` | 16 | Mamba 상태 차원 |
| `d_conv` | 4 | Mamba conv 커널 |
| `mamba_expand` | 2 | Mamba 확장 비율 |

---

## Ablation Flags

```bash
--no_attn           # Attention branch 비활성화
--no_mamba          # Mamba branch 비활성화
--no_conv           # CNN branch 비활성화
--no_circular_rel_bias   # Circular relative bias 제거
# no_circular_pad는 CircularCNNBranch의 circular=False로 제어
```

---

## 파라미터 수 (d_model=128, n_layer=6 기준)

| 구성 요소 | 파라미터 수 (추정) |
|-----------|------------------|
| Embedding | ~0.5M |
| CircMACBlock × 6 | ~4.2M |
| Cross-Attention | ~0.3M |
| Site Head | ~0.2M |
| **Total** | **~5.2M** |
