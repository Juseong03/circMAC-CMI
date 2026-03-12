# Results Section Writing Guide
> 실험 결과가 나오면 이 guide를 따라 Results를 작성하세요.

---

## 전체 Results 구성 (권장 순서)

```
4. Experiments
   4.1 Experimental Setup          ← Methods에서 이미 일부 작성됨
   4.2 Main Results (Exp1 + Exp3)  ← 핵심 테이블 2개
   4.3 Pretraining Strategy (Exp2) ← CircMAC-specific novelty
   4.4 Ablation Study (Exp4)       ← 각 component 기여도
   4.5 Design Analysis (Exp5, 6)   ← interaction, site head
```

---

## 4.1 Experimental Setup (짧게)

- Dataset statistics (sample 수, positive ratio 등)
- Evaluation metrics 정의
- Baseline 목록

**작성 포인트:**
> "All models are trained with 3 random seeds and results are
> reported as mean ± standard deviation."

---

## 4.2 Main Results

### Table 1: Encoder Architecture Comparison (Exp1)

**목적:** CircMAC가 같은 조건에서 다른 아키텍처보다 뛰어남을 보임

**구성 (from scratch, sites task 기준):**

| Model | Type | Acc | F1 | Prec | Recall | #Params |
|-------|------|-----|-----|------|--------|---------|
| LSTM | RNN | | | | | |
| Transformer | Attention | | | | | |
| Mamba | SSM | | | | | |
| Hymba | Hybrid | | | | | |
| **CircMAC (ours)** | Tri-branch | | | | | |

**작성 포인트:**
- **Bold** = best, <u>underline</u> = second best
- CircMAC가 특히 Recall에서 우수할 경우 → binding site는 false negative가 더 위험하므로 Recall 강조
- 파라미터 수도 표기해서 efficiency 어필
- 예시 문장:
  > "CircMAC achieves the highest F1 score of XX.X% among all
  > from-scratch encoders, outperforming the best baseline Hymba
  > by +X.X points, demonstrating the benefit of circular-aware
  > multi-branch architecture."

---

### Table 2: Comparison with Pretrained RNA Models (Exp3)

**목적:** CircMAC-PT가 95M+ 파라미터 모델과 비교해 경쟁력 있음을 보임

**구성:**

| Model | Mode | Params | max_len | Cov. | F1 | Recall | AUROC |
|-------|------|--------|---------|------|-----|--------|-------|
| RNABERT† | Frozen | 0.5M | 440 | 37.6% | | | |
| RNABERT† | Trainable | 0.5M | 440 | 37.6% | | | |
| RNAErnie†† | Frozen | 86M | 510 | 48.2% | | | |
| RNAErnie†† | Trainable | 86M | 510 | 48.2% | | | |
| RNA-FM | Frozen | 95M | 1024 | 100% | | | |
| RNA-FM | Trainable | 95M | 1024 | 100% | | | |
| RNA-MSM | Frozen | 96M | 1024 | 100% | | | |
| RNA-MSM | Trainable | 96M | 1024 | 100% | | | |
| **CircMAC-PT** | Fine-tuned | **3M** | 1022 | **100%** | | | |

> † Limited by positional encoding; sequences >440 nt are truncated.
> †† Sequences >510 nt are truncated.

**작성 포인트 (3가지 서사 중 결과에 맞는 것 선택):**

**Case A — CircMAC-PT가 최고 성능:**
> "CircMAC-PT achieves the best F1 of XX.X%, surpassing the strongest
> pretrained baseline RNA-FM (trainable) by +X.X points, while using
> 31× fewer parameters. Notably, RNABERT and RNAErnie are limited to
> 37.6% and 48.2% sequence coverage respectively, which may explain
> their lower performance on full-length circRNA sequences."

**Case B — CircMAC-PT가 RNA-FM/MSM과 대등:**
> "CircMAC-PT achieves competitive performance (F1: XX.X%) compared to
> RNA-FM trainable (XX.X%), despite using 31× fewer parameters and
> being specifically designed for circular topology. Models limited
> to shorter sequences (RNABERT, RNAErnie) consistently underperform,
> supporting the importance of full-length sequence modeling."

**Case C — CircMAC-PT가 frozen보다 좋지만 trainable 대형 모델보다 낮음:**
> "While trainable RNA-FM achieves the highest absolute performance,
> CircMAC-PT offers a substantially better efficiency-performance
> tradeoff: X.X% F1 with only 3M parameters vs. XX.X% with 95M,
> making it more practical for deployment. Furthermore, CircMAC-PT
> benefits from circular-specific pretraining that general RNA models
> cannot provide."

---

## 4.3 Pretraining Strategy Analysis (Exp2)

**목적:** CPCL이 circRNA-specific 이점을 제공함을 보임

### Table 3: Pretraining Objectives Ablation

| Config | MLM | NTP | SSP | CPCL | Pairing | F1 | ΔF1 vs No-PT |
|--------|-----|-----|-----|------|---------|-----|--------------|
| No Pretrain | - | - | - | - | - | | (baseline) |
| MLM | ✓ | | | | | | |
| MLM + NTP | ✓ | ✓ | | | | | |
| MLM + SSP | ✓ | | ✓ | | | | |
| MLM + CPCL | ✓ | | | ✓ | | | |
| MLM + Pairing | ✓ | | | | ✓ | | |
| MLM+NTP+CPCL+Pair | ✓ | ✓ | | ✓ | ✓ | | |

**작성 포인트:**
- **CPCL의 효과**를 반드시 언급 (논문의 핵심 novelty):
  > "Among all pretraining objectives, CPCL provides the largest
  > improvement when combined with MLM (+X.X F1), confirming that
  > circular permutation invariance is a meaningful inductive bias
  > for circRNA representation learning."
- MLM 단독 대비 각 추가 objective의 marginal gain 분석
- Best config이 Exp3 CircMAC-PT에 사용됐음을 명시

---

## 4.4 Ablation Study (Exp4)

**목적:** 각 component가 실제로 필요함을 증명

### Table 4: Component Ablation

| Configuration | Removed | F1 | ΔF1 |
|--------------|---------|-----|-----|
| **Full CircMAC** | — | | — |
| w/o Attention | Attention branch | | |
| w/o Mamba | Mamba branch | | |
| w/o CNN | CNN branch | | |
| w/o Circular Rel. Bias | Circular bias → linear | | |
| Attention only | Mamba + CNN | | |
| Mamba only | Attention + CNN | | |
| CNN only | Attention + Mamba | | |

**작성 포인트 (중요도 순):**

1. **Circular Rel. Bias 효과** (논문 핵심):
   > "Removing the circular relative position bias from attention
   > results in a drop of X.X F1 points, demonstrating that standard
   > linear position encodings fail to capture the continuity at
   > the back-splice junction."

2. **Multi-branch vs. single-branch:**
   > "Single-branch variants (Attention-only, Mamba-only, CNN-only)
   > all underperform the full model, confirming that complementary
   > modeling—global context, sequential patterns, and local
   > motifs—are each essential."

3. **가장 중요한 branch 파악:**
   - Attention-only vs. Mamba-only vs. CNN-only 중 어느 것이 가장 높은지
   - 가장 많이 떨어지는 ablation = 가장 중요한 component

---

## 4.5 Interaction Mechanism & Site Head (Exp5, Exp6)

이 두 실험은 보통 **짧게 병합**해서 작성합니다.

### Table 5: Design Choice Validation

| Experiment | Variant | F1 |
|-----------|---------|-----|
| **Interaction** | Concat | |
| | Elementwise | |
| | **Cross-Attention (ours)** | |
| **Site Head** | Linear | |
| | **Conv1d multi-kernel (ours)** | |

**작성 포인트:**
> "Cross-attention outperforms simpler interaction mechanisms
> (concat: −X.X, elementwise: −X.X), as it enables each circRNA
> position to directly attend to the full miRNA sequence.
> Similarly, the multi-kernel convolutional head improves over
> linear projection (+X.X F1), capturing binding motifs at
> multiple spatial scales."

---

## 작성 순서 권장

실험 결과 나오는 대로:

```
Step 1: 표 숫자 채우기
  → logs/에서 best F1 추출 (grep "best_f1" logs/**/*.log)

Step 2: Table 1, 2 먼저 작성 (핵심 결과)

Step 3: Pretraining analysis (Table 3)
  → CPCL 효과 강조

Step 4: Ablation (Table 4)
  → 가장 많이 떨어지는 것 = 핵심 기여

Step 5: Exp5, 6 (Table 5, 짧게)

Step 6: 전체 흐름 연결하는 문단 작성
```

---

## 결과 추출 명령어

```bash
# 모든 실험 best F1 추출
grep -r "best_f1\|test_f1\|'f1'" logs/ | grep -v "^Binary"

# Exp1 결과
grep "test_f1" logs/exp1/*.log

# Exp2 결과 (finetune)
grep "test_f1" logs/exp2/finetune/*.log

# Exp3 결과
grep "test_f1" logs/exp3/*.log

# Exp4 ablation
grep "test_f1" logs/exp4/*.log
```

---

## Figures 권장 (논문 임팩트 향상)

| Figure | 내용 | 데이터 소스 |
|--------|------|------------|
| Fig 1 | Framework overview (circRNA → CircMAC → site pred) | — |
| Fig 2 | Linear vs Circular distance matrix 비교 | 수식으로 생성 |
| Fig 3 | Main results bar chart (Exp1, Exp3) | Table 1, 2 |
| Fig 4 | Pretraining ablation + Component ablation | Table 3, 4 |
| Fig 5 | Binding site prediction example (sequence visualization) | Test set 샘플 |
| Fig 6 | Router gate distribution (Attn/Mamba/CNN 비율) | router gates 분석 |
