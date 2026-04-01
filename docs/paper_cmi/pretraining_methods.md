# Self-Supervised Pretraining Strategies for CircRNA Representation Learning

## 개요

CircMAC은 다운스트림 circRNA-miRNA 결합 예측 전에 circRNA 서열로부터 표현을 사전학습한다.
5가지 자기지도 학습 기법을 단독 및 조합으로 비교하며, 이 중 CPCL은 circRNA의 원형 위상에 특화된 방법이다.

---

## 전체 구조

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        df_pretrain (160K circRNA isoforms)                  │
│              RNAsubopt --circ 로 2차 구조 레이블 포함                        │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
              ┌─────────────────┼──────────────────┐
              │                 │                   │
    ┌─────────▼────────┐ ┌──────▼──────┐ ┌────────▼────────┐
    │  서열 기반 학습   │ │ 구조 기반   │ │  위상 기반 학습  │
    │  MLM  │  NTP     │ │ SSP│Pairing │ │     CPCL        │
    └─────────┬────────┘ └──────┬──────┘ └────────┬────────┘
              │                 │                  │
              └─────────────────┼──────────────────┘
                                │
                   ┌────────────▼────────────┐
                   │     CircMAC Backbone     │
                   │  (Attn + Mamba + CNN)   │
                   └────────────┬────────────┘
                                │
                   ┌────────────▼────────────┐
                   │   Fine-tuning: 결합     │
                   │   예측 (sites task)     │
                   └─────────────────────────┘
```

---

## 1. MLM — Masked Language Modeling

**학습 목표:** 무작위로 마스킹된 뉴클레오타이드를 양방향 문맥으로 복원

```
원본 서열:   A  U  G  C  U  A  G  C  U  U  A
                  │           │
마스킹 (15%): A [M] G  C  U [M] G  C  U  U  A
                  │           │
예측 정답:       U            A

손실: CrossEntropy( ŷ_masked, y_masked )
PAD 위치는 마스킹 제외 (fix: attention_mask 활용)
```

- BERT 방식의 표준 자기지도 학습
- 양방향 문맥 → Attention 브랜치와 CNN 브랜치 활성화
- 마스킹 비율: 15%

---

## 2. NTP — Next Token Prediction

**학습 목표:** 이전 토큰들을 보고 다음 뉴클레오타이드를 자동회귀적으로 예측

```
입력 (causal):  [0]  A   U   G   C   U   A
                 ↓   ↓   ↓   ↓   ↓   ↓
예측 정답:       A   U   G   C   U   A  [EOS]

시점 t 에서: 토큰 0..t 를 보고 토큰 t+1 예측
손실: CrossEntropy( ŷ_{t+1}, y_{t+1} )  for all t
```

- GPT 방식의 자동회귀 예측
- CircMAC의 **Mamba 브랜치 (causal SSM)** 와 학습 방향이 일치
- PAD 위치는 레이블에서 제외 (-100)

---

## 3. SSP — Secondary Structure Prediction

**학습 목표:** 각 위치의 염기쌍 여부를 이진 분류

```
circRNA 서열:  A  U  G  C  G  C  A  U  G  C
2차 구조:      (  (  .  .  )  )  (  .  )  .
                                        
레이블:        1  1  0  0  1  1  1  0  1  0
              paired         unpaired

손실: BinaryCrossEntropy( ŷ_i, ss_labels_i )
구조 레이블 출처: RNAsubopt --circ (원형 RNA 전용)
```

- dot-bracket → 이진 레이블 변환
- circRNA의 **구조적 특성** 인코딩
- `df_pretrain`의 `ss_labels` 컬럼 사용

---

## 4. Pairing — Base Pairing Matrix Reconstruction

**학습 목표:** 서열 내 모든 위치쌍의 염기쌍 여부를 행렬로 재구성

```
서열 길이 L 에 대해 L×L 행렬 예측:

        A  U  G  C  G  C
    A [ 0  0  0  0  0  0 ]
    U [ 0  0  0  0  1  0 ]  ← U-A 쌍
    G [ 0  0  0  0  0  1 ]  ← G-C 쌍
    C [ 0  0  0  0  0  0 ]
    G [ 0  1  0  0  0  0 ]  ← G-U 쌍
    C [ 0  0  1  0  0  0 ]

손실: BCEWithLogits( P̂_{ij}, pairing_{ij} )
CircularPairingHead 사용 (CircMAC 전용)
```

- SSP보다 정밀: **어느 위치와 쌍을 이루는지** 명시적 학습
- 장거리 염기쌍 관계 포착
- `df_pretrain`의 `pairing` 컬럼 사용 (sentinel=-1 for unpaired)

---

## 5. CPCL — Circular Permutation Contrastive Learning

**학습 목표:** 동일 circRNA의 다른 회전(rotation) 버전이 같은 표현을 갖도록 학습

```
원형 circRNA:  ← A - U - G - C - U - A - G - C →
                  ↑___________________________|

회전 전 (offset=0):  A  U  G  C  U  A  G  C
회전 후 (offset=3):  C  U  A  G  C  A  U  G
                     (같은 분자, 다른 시작점)

                  CircMAC Encoder
                 ↙              ↘
           z₁ = f(x)      z₂ = f(x_rot)

목표: sim(z₁, z₂) ↑   sim(z₁, z_neg) ↓

손실: NT-Xent (InfoNCE)
  L = -log [ exp(sim(z₁,z₂)/τ) / Σ exp(sim(z₁,z_neg)/τ) ]
```

- circRNA **고유 기법**: 시작 위치가 생물학적으로 임의적임
- 회전 불변 표현 (rotation-invariant representation) 학습
- `df_pretrain`의 `rotation_offset` 컬럼으로 회전 추적
- 배치 내 다른 샘플을 negative로 사용

---

## 비교 요약

```
┌──────────┬───────────────────┬──────────────┬────────────┬──────────────┐
│  기법    │    입력           │   예측 대상  │  circRNA   │   관련 브랜치│
│          │                   │              │  특화 여부 │              │
├──────────┼───────────────────┼──────────────┼────────────┼──────────────┤
│  MLM     │ 마스킹된 서열     │ 마스킹 토큰  │    ✗       │ Attn, CNN   │
│  NTP     │ 이전 토큰들       │ 다음 토큰    │    ✗       │ Mamba       │
│  SSP     │ 서열 + 구조 레이블│ 이진 구조    │    △       │ Attn, CNN   │
│  Pairing │ 서열 + pairing    │ L×L 행렬     │    △       │ Attn        │
│  CPCL    │ 서열 + 회전 버전  │ 표현 유사도  │    ✅      │ 전체        │
└──────────┴───────────────────┴──────────────┴────────────┴──────────────┘
△: 구조 정보가 필요하므로 일반 RNA에도 적용 가능하나 circRNA용 RNAsubopt --circ 사용
```

---

## 조합 실험

| 설정 | 기법 조합 | 비고 |
|------|----------|------|
| No PT | — | 베이스라인 |
| MLM | MLM only | 표준 LM |
| NTP | NTP only | 자동회귀 LM |
| SSP | SSP only | 구조 학습 |
| Pairing | Pairing only | 정밀 구조 학습 |
| CPCL | CPCL only | 위상 학습 |
| MLM+NTP | MLM + NTP | 양방향 + 자동회귀 |
| All | MLM+NTP+SSP+Pairing+CPCL | 전체 조합 |

---

## Figure 작성 가이드

논문 Figure로 만들 때 권장 구성:

```
Figure X. Self-supervised pretraining strategies for circRNA.

(a) MLM: 마스킹된 서열 위에 [MASK] 표시, 화살표로 예측 방향
(b) NTP: 왼쪽→오른쪽 causal 화살표, 각 위치에서 다음 토큰 예측
(c) SSP: 서열 위에 dot-bracket 구조 표시, 각 위치에 0/1 레이블
(d) Pairing: 서열을 원형으로 배치, 쌍을 이루는 위치 간 호(arc) 연결
(e) CPCL: 원형 서열 두 개 (다른 시작점), 인코더 통과 후 벡터 유사도 최대화
```

**도구 추천:**
- BioRender / Inkscape: 전문적인 생물정보학 스타일 그림
- draw.io (diagrams.net): 무료, 웹 기반, 논문 품질
- Matplotlib (Python): 프로그래매틱하게 Pairing arc 그림 생성 가능
