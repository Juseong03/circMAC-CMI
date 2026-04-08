# CircMAC Experiment Results
> 모든 실험 결과 정리 (logs_0403 기반, 2025-04)
> mean ± std (n=3 seeds), 기준: macro F1 Score (threshold sweep)

---

## 실험 목록 및 상태

| 실험 | 설명 | 상태 |
|------|------|------|
| EXP1 | Encoder Architecture Comparison | ✅ 완료 (일부 pending) |
| EXP2v4 | Pretraining Strategy Comparison | ⚠️ MLM/CPCL/ALL pending |
| EXP3 | CircMAC-PT (pretrained backbone) | ⚠️ s1 only / pending |
| EXP4 | CircMAC Ablation Study | ✅ 완료 |
| EXP5 | Interaction Mechanism | ✅ 완료 |
| EXP6 | Site Prediction Head | ✅ 완료 |

---

## EXP1: Encoder Architecture Comparison

### 1-A. From-Scratch Encoders (max_len=1022, BS=128)

| Model | s1 | s2 | s3 | **Mean** | **Std** |
|-------|-----|-----|-----|---------|---------|
| **CircMAC** | 0.7442 | 0.7454 | 0.7457 | **0.7451** | 0.0008 |
| Mamba | 0.7128 | 0.7144 | 0.7006 | 0.7093 | 0.0062 |
| HyMBA | 0.7021 | 0.7035 | 0.6807 | 0.6954 | 0.0101 |
| LSTM | 0.6521 | 0.6561 | 0.6491 | 0.6524 | 0.0035 |
| Transformer | 0.6044 | 0.6212 | 0.6029 | 0.6095 | 0.0083 |

> 실험명: `exp1_circmac_s{1,2,3}`, `exp1_mamba_s{1,2,3}`, `exp1_hymba_s{1,2,3}`, `exp1_lstm_s{1,2,3}`, `exp1_transformer_s{1,2,3}`

---

### 1-B. Pretrained RNA LMs — Frozen (Fair, max_len=438)

> rnabert 최대 길이(438nt)에 맞춰 모든 모델을 동일 조건으로 비교

| Model | Params | s1 | s2 | s3 | **Mean** | **Std** |
|-------|--------|-----|-----|-----|---------|---------|
| RNABERT | 113M | 0.5949 | 0.5929 | 0.6016 | 0.5965 | 0.0038 |
| RNAErnie | 86M | 0.6144 | 0.6107 | 0.6168 | **0.6140** | 0.0025 |
| RNA-FM | 95M | 0.6092 | 0.6073 | 0.6120 | 0.6095 | 0.0020 |
| RNA-MSM | 96M | 0.6057 | 0.6083 | 0.6119 | 0.6086 | 0.0025 |

> 실험명: `exp1_fair_frozen_{rnabert,rnaernie,rnafm,rnamsm}_s{1,2,3}`

---

### 1-C. Pretrained RNA LMs — Frozen (Max, max_len=1022)

> rnabert 제외, 나머지 모델을 각자 최대 길이(1022)에서 평가

| Model | s1 | s2 | s3 | **Mean** | **Std** |
|-------|-----|-----|-----|---------|---------|
| RNAErnie | 0.6124 | 0.6105 | 0.6178 | **0.6136** | 0.0031 |
| RNA-FM | 0.6046 | 0.6076 | 0.6026 | 0.6049 | 0.0021 |
| RNA-MSM | 0.6056 | 0.6065 | 0.6052 | 0.6058 | 0.0006 |

> 실험명: `exp1_max_frozen_{rnaernie,rnafm,rnamsm}_s{1,2,3}`

---

### 1-D. Pretrained RNA LMs — Trainable (Fair, max_len=438)

| Model | s1 | s2 | s3 | **Mean** | **Std** |
|-------|-----|-----|-----|---------|---------|
| RNABERT | 0.6009 | 0.6021 | 0.6029 | 0.6020 | 0.0008 |
| RNAErnie | 0.5948 | 0.5890 | 0.5949 | 0.5929 | 0.0027 |
| **RNA-FM** | **0.6421** | **0.6404** | **0.6376** | **0.6400** | 0.0019 |
| RNA-MSM | 0.6316 | 0.6361 | 0.6269 | 0.6315 | 0.0038 |

> 실험명: `exp1_fair_trainable_{rnabert,rnaernie,rnafm,rnamsm}_s{1,2,3}`

---

### 1-E. Pretrained RNA LMs — Trainable (Max, max_len=1022)

| Model | s1 | s2 | s3 | **Mean** | **Std** |
|-------|-----|-----|-----|---------|---------|
| RNAErnie | 0.6040 | 0.5958 | 0.6032 | 0.6010 | 0.0037 |
| RNA-FM | — | — | — | OOM (pending re-run, BS=8) | — |
| RNA-MSM | — | — | — | OOM (pending re-run, BS=8) | — |

> 실험명: `exp1_max_trainable_{rnaernie,rnafm,rnamsm}_s{1,2,3}`

---

### EXP1 핵심 비교 (시각화용)

| Model | Mode | Len | Mean F1 |
|-------|------|-----|---------|
| **CircMAC** | scratch | 1022 | **0.7451** |
| Mamba | scratch | 1022 | 0.7093 |
| HyMBA | scratch | 1022 | 0.6954 |
| LSTM | scratch | 1022 | 0.6524 |
| Transformer | scratch | 1022 | 0.6095 |
| RNA-FM | trainable | 438 | 0.6400 |
| RNA-MSM | trainable | 438 | 0.6315 |
| RNABERT | trainable | 438 | 0.6020 |
| RNAErnie | frozen | 1022 | 0.6136 |
| RNAErnie | frozen | 438 | 0.6140 |
| RNA-FM | frozen | 438 | 0.6095 |
| RNA-MSM | frozen | 438 | 0.6086 |
| RNABERT | frozen | 438 | 0.5965 |

---

## EXP2: Pretraining Strategy Comparison

### EXP2v4 (최신, BS=32 finetune, BS=192 pretrain)

> **논문 메인 테이블용**

| Pretraining | s1 | s2 | s3 | **Mean** | **Std** | Δ vs No-PT |
|-------------|-----|-----|-----|---------|---------|-----------|
| **Pairing** | **0.7704** | **0.7743** | **0.7693** | **0.7713** | 0.0022 | **+1.64pp** |
| SSP | 0.7603 | 0.7701 | 0.7571 | 0.7625 | 0.0056 | +0.76pp |
| No-PT | 0.7531 | 0.7586 | 0.7529 | 0.7549 | 0.0026 | — |
| MLM+NTP | 0.7183 | 0.7183 | 0.6899 | 0.7088 | 0.0134 | −4.61pp |
| MLM | — | — | — | pending | — | — |
| CPCL | — | — | — | pending | — | — |
| ALL (combined) | — | — | — | pending | — | — |

> 실험명: `exp2v4_{nopt,ssp,pair,mlm_ntp,mlm,cpcl,all}_sites_s{1,2,3}`

---

### EXP2v3 (이전 버전, 참고용)

> BS=32 finetune, 동일 아키텍처. v4와 비교하면 No-PT 기준선이 약간 높음 (BS 차이 없음, 데이터 버전 차이)

| Pretraining | s1 | s2 | s3 | **Mean** | **Std** |
|-------------|-----|-----|-----|---------|---------|
| **Pairing** | **0.7629** | **0.7744** | **0.7583** | **0.7652** | 0.0067 |
| SSP | 0.7597 | 0.7673 | 0.7594 | 0.7621 | 0.0037 |
| MLM | 0.7573 | 0.7581 | 0.7510 | 0.7555 | 0.0031 |
| No-PT | 0.7576 | 0.7595 | 0.7558 | 0.7576 | 0.0016 |
| MLM+NTP | 0.7457 | 0.7522 | 0.7396 | 0.7458 | 0.0051 |
| CPCL | — | — | — | not completed | — |
| NTP | — | — | — | not completed | — |

> 실험명: `exp2v3_{nopt,mlm,ssp,pair,mlm_ntp}_sites_s{1,2,3}`

---

## EXP3: CircMAC-PT (pretrained backbone fine-tuning)

### 3-A. CircMAC-PT (CPCL 사전학습 weights 사용)

| | s1 | s2 | s3 | Mean |
|-|-----|-----|-----|------|
| CircMAC-PT | — | — | — | s1만 실행 중 (early training) |

> 실험명: `exp3_circmac_pt_s{1,2,3}`
> ⚠️ EXP2v4 완료 후 best PT(Pairing) weights로 재실행 필요

---

### 3-B. Encoder별 비교 (native max_len, EXP3 기준)

> EXP1과 차이: rnabert=438, rnaernie/rnafm/rnamsm=1022 (각자 native limit)

| Model | Mode | s1 | s2 | s3 | **Mean** | **Std** |
|-------|------|-----|-----|-----|---------|---------|
| **CircMAC** | scratch | 0.7457 | 0.7414 | 0.7390 | **0.7420** | 0.0028 |
| Mamba | scratch | 0.7128 | 0.7144 | 0.7006 | 0.7093 | 0.0062 |
| HyMBA | scratch | 0.7037 | 0.7063 | 0.6908 | 0.7003 | 0.0066 |
| LSTM | scratch | 0.6521 | 0.6575 | 0.6485 | 0.6527 | 0.0038 |
| Transformer | scratch | 0.6044 | 0.6212 | 0.6029 | 0.6095 | 0.0083 |
| RNA-MSM | trainable | 0.6411 | 0.6393 | 0.6265 | 0.6356 | 0.0064 |
| RNA-FM | trainable | 0.6289 | 0.6384 | — | 0.6337† | — |
| RNAErnie | trainable | 0.6022 | 0.5911 | 0.6105 | 0.6013 | 0.0079 |
| RNABERT | trainable | 0.6009 | 0.6021 | 0.6029 | 0.6020 | 0.0008 |
| RNAErnie | frozen | 0.6120 | 0.6144 | 0.6129 | 0.6131 | 0.0010 |
| RNA-FM | frozen | 0.6046 | 0.6076 | 0.6026 | 0.6049 | 0.0021 |
| RNA-MSM | frozen | 0.6052 | 0.6061 | 0.6052 | 0.6055 | 0.0004 |
| RNABERT | frozen | 0.5827 | 0.5696 | 0.5905 | 0.5809 | 0.0088 |

> † RNA-FM trainable: s3 OOM, 2-seed 평균
> 실험명: `exp3_{model}_{frozen,trainable,sites}_s{1,2,3}`

---

## EXP4: CircMAC Ablation Study

### 전체 결과 (max_len=1022, BS=128)

| Variant | Description | s1 | s2 | s3 | **Mean** | **Std** | **Δ vs Full** |
|---------|-------------|-----|-----|-----|---------|---------|--------------|
| **Full (CircMAC)** | 전체 모델 | 0.7459 | 0.7449 | 0.7449 | **0.7452** | 0.0005 | — |
| No Circ Bias | Attention circular bias 제거 | 0.7435 | 0.7408 | 0.7381 | 0.7408 | 0.0022 | −0.44pp |
| No Circ Pad | CNN circular padding 제거 | 0.7438 | 0.7445 | 0.7459 | 0.7447 | 0.0009 | −0.05pp |
| No Attn | Attention branch 제거 | 0.7401 | 0.7407 | 0.7389 | 0.7399 | 0.0008 | −0.53pp |
| No Mamba | Mamba branch 제거 | 0.6844 | 0.6850 | 0.6785 | 0.6826 | 0.0030 | **−6.26pp** |
| No Conv | CNN branch 제거 | 0.6969 | 0.6798 | 0.6939 | 0.6902 | 0.0074 | **−5.50pp** |
| Attn Only | Mamba+CNN 제거 | 0.6317 | 0.7065 | 0.7005 | 0.6796 | 0.0331 | −6.56pp |
| Mamba Only | Attn+CNN 제거 | 0.7025 | 0.6988 | 0.7006 | 0.7006 | 0.0015 | −4.46pp |
| CNN Only | Attn+Mamba 제거 | 0.6731 | 0.6740 | 0.6645 | 0.6705 | 0.0043 | −7.47pp |

> 실험명: `exp4_{full,no_attn,no_mamba,no_conv,no_circular_bias,no_circular_pad,attn_only,mamba_only,cnn_only}_s{1,2,3}`

**해석:**
- Mamba 제거 (−6.3pp) > Conv 제거 (−5.5pp) >> Attn 제거 (−0.5pp)
- Circular 구성요소: Circ Bias (−0.44pp), Circ Pad (−0.05pp) → 개별 효과는 작지만 전체 circular architecture의 일부
- Attn-only의 높은 분산(std=0.033)은 attention만으로 불안정함을 시사

---

## EXP5: Interaction Mechanism

| Interaction | Description | s1 | s2 | s3 | **Mean** | **Std** | **Δ** |
|-------------|-------------|-----|-----|-----|---------|---------|-------|
| **Cross-Attention** | circRNA→miRNA cross-attn | **0.7464** | **0.7472** | **0.7454** | **0.7463** | 0.0008 | — |
| Concat | [circRNA; miRNA] → Linear | 0.7206 | 0.7162 | 0.7165 | 0.7178 | 0.0020 | −2.85pp |
| Elementwise | circRNA ⊙ miRNA | 0.7112 | 0.7147 | 0.7018 | 0.7092 | 0.0056 | −3.71pp |

> 실험명: `exp5_{cross_attention,concat,elementwise}_s{1,2,3}`

---

## EXP6: Site Prediction Head

| Head | Description | s1 | s2 | s3 | **Mean** | **Std** | **Δ** |
|------|-------------|-----|-----|-----|---------|---------|-------|
| **Conv1D** | Multi-scale (k=3,5,7) + Conv1D×2 | **0.7450** | **0.7460** | **0.7460** | **0.7457** | 0.0005 | — |
| Linear | Direct projection | 0.7360 | 0.7436 | 0.7337 | 0.7378 | 0.0043 | −0.79pp |

> 실험명: `exp6_{conv1d,linear}_s{1,2,3}`

---

## 종합 비교 (논문 Table 1용)

| Model | Setting | Params | F1 Mean | F1 Std |
|-------|---------|--------|---------|--------|
| **CircMAC-PT (Pairing)** | pretrain+FT | ~5.2M | **0.7713** | 0.0022 |
| CircMAC-PT (SSP) | pretrain+FT | ~5.2M | 0.7625 | 0.0056 |
| CircMAC-PT (No-PT) | FT only (BS=32) | ~5.2M | 0.7549 | 0.0026 |
| **CircMAC** | scratch (BS=128) | ~5.2M | **0.7451** | 0.0008 |
| Mamba | scratch | ~4.1M | 0.7093 | 0.0062 |
| HyMBA | scratch | ~4.4M | 0.6954 | 0.0101 |
| LSTM | scratch | ~1.8M | 0.6524 | 0.0035 |
| Transformer | scratch | ~3.6M | 0.6095 | 0.0083 |
| RNA-FM (trainable) | fine-tuned LM | 95M | 0.6400 | 0.0019 |
| RNA-MSM (trainable) | fine-tuned LM | 96M | 0.6315 | 0.0038 |
| RNABERT (trainable) | fine-tuned LM | 113M | 0.6020 | 0.0008 |
| RNAErnie (frozen) | frozen LM | 86M | 0.6136 | 0.0031 |

---

## Pending 실험 목록

| 실험 | 내용 | 비고 |
|------|------|------|
| EXP2v4 MLM | MLM 단독 사전학습 | 서버 진행 중 |
| EXP2v4 CPCL | CPCL 단독 사전학습 | 서버 진행 중 |
| EXP2v4 ALL | 5가지 목표 결합 | 서버 진행 중 |
| EXP3 CircMAC-PT | Pairing PT → FT | EXP2v4 완료 후 실행 |
| EXP1 RNA-FM max trainable | max_len=1022, BS=8 | OOM → BS=8 재실험 필요 |
| EXP1 RNA-MSM max trainable | max_len=1022, BS=8 | OOM → BS=8 재실험 필요 |

---

## 시각화에 사용할 실험명 (logs/ 경로 기준)

### Figure A: Encoder Comparison (Bar chart)
```
# From-scratch
logs/circmac/exp1_circmac_s{1,2,3}/{seed}/training.json
logs/mamba/exp1_mamba_s{1,2,3}/{seed}/training.json
logs/hymba/exp1_hymba_s{1,2,3}/{seed}/training.json
logs/lstm/exp1_lstm_s{1,2,3}/{seed}/training.json
logs/transformer/exp1_transformer_s{1,2,3}/{seed}/training.json

# RNA LMs (fair frozen)
logs/rnabert/exp1_fair_frozen_rnabert_s{1,2,3}/{seed}/training.json
logs/rnaernie/exp1_fair_frozen_rnaernie_s{1,2,3}/{seed}/training.json
logs/rnafm/exp1_fair_frozen_rnafm_s{1,2,3}/{seed}/training.json
logs/rnamsm/exp1_fair_frozen_rnamsm_s{1,2,3}/{seed}/training.json

# RNA LMs (fair trainable)
logs/rnabert/exp1_fair_trainable_rnabert_s{1,2,3}/{seed}/training.json
logs/rnafm/exp1_fair_trainable_rnafm_s{1,2,3}/{seed}/training.json
logs/rnamsm/exp1_fair_trainable_rnamsm_s{1,2,3}/{seed}/training.json
```

### Figure B: Pretraining Strategy (Bar chart)
```
logs/circmac/exp2v4_nopt_sites_s{1,2,3}/{seed}/training.json
logs/circmac/exp2v4_ssp_sites_s{1,2,3}/{seed}/training.json
logs/circmac/exp2v4_pair_sites_s{1,2,3}/{seed}/training.json
logs/circmac/exp2v4_mlm_ntp_sites_s{1,2,3}/{seed}/training.json
# pending:
logs/circmac/exp2v4_mlm_sites_s{1,2,3}/{seed}/training.json
logs/circmac/exp2v4_cpcl_sites_s{1,2,3}/{seed}/training.json
logs/circmac/exp2v4_all_sites_s{1,2,3}/{seed}/training.json
```

### Figure C: Ablation Study (Bar chart)
```
logs/circmac/exp4_{full,no_attn,no_mamba,no_conv}_s{1,2,3}/{seed}/training.json
logs/circmac/exp4_{no_circular_bias,no_circular_pad}_s{1,2,3}/{seed}/training.json
logs/circmac/exp4_{attn_only,mamba_only,cnn_only}_s{1,2,3}/{seed}/training.json
```

### Figure D: Interaction + Site Head (Bar chart)
```
logs/circmac/exp5_{cross_attention,concat,elementwise}_s{1,2,3}/{seed}/training.json
logs/circmac/exp6_{conv1d,linear}_s{1,2,3}/{seed}/training.json
```

### Figure E: BSJ-Proximal Analysis (Bar chart, model inference 필요)
```
saved_models/circmac/exp1_circmac_s1/1/train/epoch/{best_epoch}/model.pth   ← CircMAC
saved_models/mamba/exp1_mamba_s1/1/train/epoch/{best_epoch}/model.pth        ← Mamba
saved_models/transformer/exp1_transformer_s1/1/train/epoch/{best_epoch}/model.pth
```

### Figure F: Case Study (model inference 필요)
```
saved_models/circmac/exp1_circmac_s1/1/train/epoch/{best_epoch}/model.pth   ← CircMAC (best)
saved_models/mamba/exp1_mamba_s1/1/train/epoch/{best_epoch}/model.pth        ← baseline 비교
```

---

## Best Epoch 참조 (logs/에서 자동 탐지 가능)

`logs/{model}/{exp}/{seed}/training.json` → `final` key의 첫 번째 key = best epoch

| Experiment | Best Epoch (seed 1) |
|------------|-------------------|
| exp1_circmac_s1 | training.json `final` 참조 |
| exp1_mamba_s1 | training.json `final` 참조 |
| exp2v4_pair_sites_s1 | training.json `final` 참조 |

```bash
# 자동 추출 예시
python -c "
import json, glob
for f in sorted(glob.glob('logs/circmac/exp1_circmac_s*/*/training.json')):
    d = json.load(open(f))
    ep = list(d['final'].keys())[0]
    print(f'{f}: best_epoch={ep}')
"
```
