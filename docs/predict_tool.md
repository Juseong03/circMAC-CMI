# predict.py — CircMAC Inference Tool

## 개요

기존 `training.py` / `trainer.py`는 학습·평가 전용이었고, 새로운 circRNA–miRNA 쌍을 넣으면 binding site를 바로 예측해주는 **독립 추론 도구**가 없었다. `predict.py`는 이를 위해 새로 작성된 파일이다.

---

## 기존 코드에서 변경·재활용한 부분

### 재활용한 코드 (수정 없음)

| 모듈 | 재활용 내용 |
|---|---|
| `trainer.Trainer` | `forward()`, `forward_target()`, `forward_cross_attention()`, `forward_task()` 메서드 그대로 호출 |
| `data.CircRNABindingSitesDataset` | 시퀀스를 토크나이징·패딩하는 데이터셋 클래스 그대로 사용 |
| `utils_config.get_model_config` | 모델 아키텍처 config 생성 |
| `utils.get_device` | GPU/CPU 선택 |

### 새로 작성한 부분 (`predict.py` 전체)

기존 코드 어디에도 없던 기능들이다.

---

## 세부 구현 내용

### 1. 모델 로딩 (`load_model`)

```python
def load_model(ckpt_path, model_name, device) -> Trainer:
```

**기존과 차이점:**
- `training.py`는 `Trainer`를 생성한 뒤 데이터셋을 먼저 만들고 `train_model()` 루프를 돌린다.
- `predict.py`는 학습 루프 없이 **checkpoint만 로드**하고 `model.eval()` 상태로 유지한다.
- checkpoint에서 embedding weight shape을 읽어 `vocab_size`를 자동 추론 → 사용자가 별도로 지정하지 않아도 된다.

```python
# vocab_size 자동 추론
for k, v in state.items():
    if "embedding" in k and "weight" in k and v.ndim == 2 and v.shape[-1] == D_MODEL:
        vocab_size = int(v.shape[0])
        break
```

---

### 2. 단일 쌍 추론 (`infer_pair`)

```python
def infer_pair(trainer, circ_seq, mirna_seq, max_len) -> np.ndarray
```

**기존과 차이점:**
- `trainer.py`의 `predict()` 루프는 DataLoader 배치 단위로 돌며 loss 계산·메트릭 집계를 같이 한다.
- `infer_pair`는 **1쌍만** 처리하고 loss 없이 확률값 배열만 반환한다.
- `sites` 레이블을 dummy(전부 0)로 채워 데이터셋에 넣는다 — 실제 정답 없이도 동작하도록.
- 출력 텐서의 CLS/SEP 토큰 길이 차이를 `probs_np[:target_len]`로 정리.

**추론 파이프라인:**
```
circRNA seq, miRNA seq
    ↓ CircRNABindingSitesDataset (tokenize, pad)
    ↓ Trainer.forward()           → circRNA embedding
    ↓ Trainer.forward_target()    → miRNA embedding
    ↓ Trainer.forward_cross_attention()
    ↓ Trainer.forward_task(task='sites')
    → logits (1, L, 2) or (1, L)
    → softmax / sigmoid → per-nucleotide probabilities [0, 1]
```

---

### 3. Site 추출 및 출력 컬럼 생성 (`extract_sites`)

```python
def extract_sites(probs, circ_seq, mirna_seq, circ_id, mirna_id,
                  threshold, bsj_window, min_site_len) -> pd.DataFrame
```

이 함수가 output 컬럼들을 만드는 핵심이다. 기존 코드 어디에도 없었다.

#### 동작 방식

1. **Thresholding**: `prob >= threshold`인 위치를 1로 마킹
2. **Contiguous run 추출**: 연속된 1의 구간 `[start, end)`를 사이트로 정의
3. **각 사이트별 컬럼 계산**:

| 컬럼 | 계산 방법 |
|---|---|
| `site_start` | 연속 구간의 0-based 시작 위치 (BSJ = position 0) |
| `site_end` | 연속 구간의 0-based 배타적 끝 위치 |
| `site_length` | `site_end - site_start` |
| `site_score` | 구간 내 확률의 **평균값** |
| `peak_position` | 구간 내 확률이 최대인 절대 위치 (`start + argmax`) |
| `peak_probability` | `peak_position`에서의 확률값 |
| `BSJ_relation` | `distance_to_BSJ <= bsj_window` 이면 `"BSJ-adjacent"`, 아니면 `"distal"` |
| `distance_to_BSJ` | 사이트 양쪽 끝과 BSJ(0번) 사이의 **원형 최단 거리** |
| `circRNA_site_sequence` | `circ_seq[site_start:site_end]` |
| `miRNA_sequence` | 전체 miRNA 서열 |

#### BSJ 거리 계산 (원형)

circRNA는 원형이므로 BSJ(위치 0)까지의 거리를 선형과 wrap-around 두 방향으로 계산해 최솟값을 취한다:

```python
dist_start_fwd = start          # 정방향: site 시작 → BSJ
dist_end_fwd   = end - 1        # 정방향: site 끝 → BSJ  
dist_start_rev = L - start      # 역방향: wrap-around
dist_to_bsj    = min(dist_start_fwd, dist_end_fwd, dist_start_rev)
```

4. **Rank 부여**: `site_score` 내림차순 정렬 후 1부터 번호 부여

---

### 4. 입력 방식 3가지

#### A. 커맨드라인 단일 쌍
```bash
python predict.py \
  --circRNA GUGCACAUUC... --circRNA_id circFANCA \
  --miRNA   GUGAGGAGG...  --miRNA_id   hsa-miR-6858-5p \
  --model_path saved_models/circmac/max_circmac_pairing_s1/train/model.pth
```

#### B. 배치 CSV
```bash
python predict.py --input pairs.csv --out results.csv
```
CSV 필수 컬럼: `circRNA`, `miRNA` / 선택: `circRNA_id`, `miRNA_id`

#### C. FASTA 파일 쌍 (1:1 매핑)
```bash
python predict.py \
  --circRNA_fasta circrna.fa \
  --miRNA_fasta   mirna.fa \
  --out results.csv
```

---

### 5. 주요 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--threshold` | 0.5 | nucleotide를 binding으로 분류하는 확률 기준 |
| `--bsj_window` | 40 | `BSJ-adjacent` 판단 거리 (nt) |
| `--max_len` | 1022 | circRNA 최대 처리 길이 (초과 시 truncation) |
| `--min_site_len` | 1 | 보고할 최소 사이트 길이 (nt) |
| `--top_n` | None (전체) | pair당 상위 N개 사이트만 출력 |
| `--out_format` | csv | 출력 포맷: `csv` / `tsv` / `json` |
| `--model_name` | circmac | 모델 아키텍처 키 |

---

## 출력 예시

```
circRNA_id,miRNA_id,rank,site_start,site_end,site_length,site_score,peak_position,peak_probability,BSJ_relation,distance_to_BSJ,circRNA_site_sequence,miRNA_sequence
circFANCA,hsa-miR-6858-5p,1,99,131,32,0.73421,112,0.91203,BSJ-adjacent,27,CACGGCUGGCCGACCUCAAGG...,GUGAGGAGGGGCUGGCAGGGAC
circFANCA,hsa-miR-6858-5p,2,27,58,31,0.61033,41,0.84512,BSJ-adjacent,0,UCUCCACCCACCCCUGGUUCCC...,GUGAGGAGGGGCUGGCAGGGAC
```

---

## 기존 코드 미변경 사항

- `trainer.py`: 수정 없음
- `data.py`: 수정 없음
- `models/`: 수정 없음
- `training.py`: 수정 없음

`predict.py`는 완전히 독립된 신규 파일로, 기존 학습·평가 파이프라인에 영향을 주지 않는다.
