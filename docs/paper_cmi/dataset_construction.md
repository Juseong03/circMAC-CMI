# Dataset Construction

## 개요

본 문서는 circRNA-miRNA 결합 예측 모델의 사전학습 및 실험 데이터셋 구성 과정을 기술한다.
데이터셋은 크게 세 가지로 구성된다.

| 데이터셋 | 파일 | 행 수 | 용도 |
|---------|------|-------|------|
| `df_pretrain` | `extracted_data/df_pretrain.pkl` | 1,988,672 | 2차 구조 기반 사전학습 |
| `df_train` | `extracted_data/df_train_final.pkl` | 45,272 | circRNA-miRNA 결합 예측 학습 |
| `df_test` | `extracted_data/df_test_final.pkl` | 17,708 | circRNA-miRNA 결합 예측 평가 |

---

## 1. 원본 데이터 소스

| 파일 | 설명 | 출처 |
|------|------|------|
| `isoform_human.csv` | circRNA isoform 좌표 및 엑손 정보 | FL-circAS DB |
| `human_sequence_v3.0` | circRNA ID별 전체 서열 | circAtlas v3.0 |
| `human_bed_v3.0.txt` | circRNA 게놈 좌표 (BED 형식) | circAtlas v3.0 |
| `BSJ_human.csv` | Back-Splice Junction 정보 | FL-circAS DB |
| `hg38.fasta` | 인간 참조 게놈 (GRCh38) | UCSC |
| `miRNA_binding_sites_in_circRNA_sequence.human.tsv` | circRNA-miRNA 결합 사이트 annotation | FL-circAS DB |
| `mature.fa` | miRBase mature miRNA 서열 | miRBase v22 |

---

## 2. 공통 전처리: circRNA 서열 추출

**노트북**: `[0]data.ipynb`, `[1]Filter data.ipynb`
**출력**: `df_circ.json` (40,255 unique isoform)

### 2-1. circAtlas 로드 및 QC

```
human_sequence_v3.0 + human_bed_v3.0.txt
        ↓ inner merge on circAtlas_ID
결측/unknown 서열 제거
strand 유효성 확인 (+/- 만 허용)
좌표 정수화 및 start < end 검증
허용 염색체: chr1~chr22, chrX, chrY
BSJ_ID 생성: chr|start|end|strand
중복 BSJ_ID 제거 (first keep)
        ↓
df_atlas (circAtlas_ID, BSJ_ID, sequence)
```

### 2-2. isoform 필터링

- `algorithm` 컬럼에 isoCirc 및 CIRI-long 모두 포함된 isoform만 사용
  (두 알고리즘 모두에서 검출된 신뢰도 높은 isoform)
- 필요 컬럼: BSJ_ID, isoform_ID, chr, exon_start, exon_end, strand, len

### 2-3. 게놈 서열 추출

`pyfaidx`를 이용해 hg38.fasta에서 각 isoform의 엑손 좌표 기반으로 서열 추출.

```python
# 엑손 서열 연결
sequence = ''.join(genome[chr][start-1:end].seq for start, end in zip(exon_starts, exon_ends))

# 음가닥 처리: 역상보 + T→U 변환
if strand == '-':
    sequence = reverse_complement(sequence)
sequence = sequence.replace('T', 'U')
```

길이 불일치(`len(sequence) != isoform['len']`) 발생 시 경고 출력 후 제외.

### 2-4. 필터링 및 저장

- 허용 염색체 필터 재적용
- 최종 컬럼: BSJ_ID, isoform_ID, circRNA
- 출력: `extracted_data/circRNA_0813.json` (147,329행, 40,255 unique isoform)

---

## 3. df_pretrain 구성

**노트북**: `[2][1] Secondary_structure3.ipynb`, `[3] Build pretrain dataset.ipynb`

### 3-1. 2차 구조 예측

RNAsubopt (`--circ` 옵션으로 원형 RNA 처리)를 사용해 isoform별 suboptimal 구조를 예측한다.

```bash
# extracted_data/ss/run_RNAFold_subopt.sh
RNAsubopt --circ -p 50 < circRNAs_chr*.fasta > results_rnafold_subopt/
```

- `--circ`: 원형 RNA topology 적용
- `-p 50`: isoform당 50개의 suboptimal 구조 샘플링
- 입력: 염색체별 FASTA (`circRNAs_chr*.fasta`)
- 출력: `extracted_data/ss/results_rnafold_subopt/*.txt`

예측 결과를 파싱하여 `df_circ_ss_subopts.pkl`로 저장 (39,782 isoform × 50구조 = 1,989,100행).

별도로 RNAFold (MFE optimal 구조 1개)도 예측하여 `df_circ_ss.pkl`로 저장 (29,973행).
→ df_circ_ss의 모든 isoform은 df_circ_ss_subopts에 포함됨.

### 3-2. 구조 레이블 생성

dot-bracket 표기법에서 세 가지 레이블을 생성한다.

**pairing** (list[int]): 각 위치의 짝 위치 인덱스. 비짝은 `-1`.

```python
# get_pairing_info
pairing = [-1] * L
stack = []
for i, char in enumerate(dot_bracket):
    if char == '(':
        stack.append(i)
    elif char == ')' and stack:
        j = stack.pop()
        pairing[i] = j
        pairing[j] = i
```

**ss_labels** (list[int]): 이진 구조 레이블.

```
0 = unpaired ('.')
1 = paired   ('(' or ')')
```

**ss_labels_multi** (list[int]): 염기쌍 종류를 구분하는 다중 클래스 레이블.

| 값 | 의미 |
|----|------|
| 0 | unpaired |
| 1 | paired (기타) |
| 2 | A-U opening `(` |
| 3 | A-U closing `)` |
| 4 | G-C opening `(` |
| 5 | G-C closing `)` |
| 6 | G-U opening `(` |
| 7 | G-U closing `)` |

### 3-3. df_pretrain 최종 구성

**노트북**: `[3] Build pretrain dataset.ipynb`

```
df_circ_ss_subopts.pkl (1,989,100행)
        ↓
컬럼 정규화
  - circRNA_id → isoform_ID
  - chr 추출 (isoform_ID.split('|')[0])
  - energy = 0.0 (suboptimal은 에너지 값 없음)
  - is_optimal = False
        ↓
완전 unpaired 구조 제거 (set(structure)=={'.'})
  → 428개 제거
        ↓
pairing sentinel 교정
  - 기존 sentinel=0은 위치 0과의 쌍과 구별 불가
  - ss_labels[i]==0 이면 pairing[i]=-1 로 교정
        ↓
df_pretrain.pkl 저장
```

**최종 스펙:**

| 항목 | 값 |
|------|-----|
| 행 수 | 1,988,672 |
| unique isoforms | 39,781 |
| 구조 수/isoform | 50 |
| circRNA 길이 | min=85, max=2,042, mean≈500 |
| 컬럼 | isoform_ID, circRNA, chr, structure, energy, length, pairing, ss_labels, ss_labels_multi, is_optimal, rotation_offset |

---

## 4. df_train / df_test 구성

**노트북**: `[2] Construct dataset.ipynb`

### 4-1. miRNA annotation 로드

```
miRNA_binding_sites_in_circRNA_sequence.human.tsv
        ↓
컬럼 표준화: query_id→miRNA_ID, reference_id→isoform_ID
isoform_ID 접두어 보정: chr 없으면 prepend
score, energy 수치 변환 및 결측 제거
```

miRBase `mature.fa`에서 human miRNA 서열 로드 후 annotation과 merge.

### 4-2. Positive 샘플 정의

AGO-CLIP 지지 데이터만을 Positive로 사용한다.

```python
df_pos = df_anno[
    (df_anno['score']       >= 155) &   # Miranda score
    (df_anno['energy']      <= -20) &   # binding energy (kcal/mol)
    (df_anno['AGO_support'] >= 1)       # AGO-CLIP 지지
]
```

### 4-3. Negative 샘플 정의

RNAhybrid 예측 결과를 활용해 negative를 구성한다.

**Train negative** (엄격한 기준):
```python
mfe < -25.0 and p_value < 0.05 and alignment_length >= 18
```

**Test negative** (완화된 기준):
```python
mfe < -20.0 and p_value < 0.1  and alignment_length >= 18
```

Positive pair와 겹치는 (isoform_ID, miRNA_ID) 조합은 negative에서 제외.

### 4-4. circRNA 길이 필터

```python
150 <= len(circRNA) <= 1000
```

### 4-5. Train/Test 분리

Positive pair 중 10,000개를 랜덤 샘플링하여 test set으로 고정.
동일한 (isoform_ID, miRNA_ID) pair가 train과 test에 동시 등장하지 않도록 분리.

```
df_pos_test  = df_pos.sample(n=10000, random_state=42)
df_pos_train = df_pos[~df_pos['pair'].isin(test_pairs)]
```

> ⚠️ circRNA isoform 수준에서는 train/test 간 4,633개 겹침 존재.
> pair 단위 분리는 보장되나, 동일 circRNA 서열이 양쪽에 등장할 수 있음.

### 4-6. Binding mask 생성

pair별로 여러 binding site를 하나의 binary mask로 통합한다.

```python
# bandwidth=2: 결합 위치 주변 ±2nt 확장
mask = create_binding_mask(seq_len, ref_start, ref_end, bandwidth=2)

# 동일 pair의 여러 site → OR 연산으로 통합
final_mask = np.maximum.reduce(all_masks)
```

### 4-7. 최종 데이터셋 스펙

**df_train_final.pkl**

| 항목 | 값 |
|------|-----|
| 행 수 (pairs) | 45,272 |
| label=1 (Positive) | 19,674 |
| label=0 (Negative) | 25,598 |
| circRNA 길이 범위 | 150~1,000 |
| 컬럼 | isoform_ID, miRNA_ID, sites(mask), label, circRNA, binding, length, n_binding_site, ratio_binding_site |

**df_test_final.pkl**

| 항목 | 값 |
|------|-----|
| 행 수 (pairs) | 17,708 |
| label=1 (Positive) | 7,838 |
| label=0 (Negative) | 9,870 |
| circRNA 길이 범위 | 150~1,000 |
| 컬럼 | 동일 |

---

## 5. 데이터 흐름 요약

```
[원본 소스]
  isoform_human.csv + human_sequence_v3.0 + human_bed_v3.0.txt + hg38.fasta
           │
           ▼ [0]data.ipynb / [1]Filter data.ipynb
  df_circ.json (40,255 unique isoform, circRNA 서열)
           │
     ┌─────┴──────────────────────────────────────────┐
     │                                                │
     ▼ [2][1] Secondary_structure3.ipynb              ▼ [2] Construct dataset.ipynb
  RNAsubopt --circ -p50                    miRNA_binding_sites.tsv + mature.fa
     │                                                │
     ▼                                                ▼
  df_circ_ss_subopts.pkl                  Positive (AGO-CLIP 지지)
  (39,782 isoform × 50구조)              + Negative (RNAhybrid)
     │                                                │
     ▼ [3] Build pretrain dataset.ipynb              │
  전처리 (sentinel 교정, unpaired 제거)              │
     │                                    길이 필터 (150~1000)
     ▼                                    Train/Test 분리 (pair 단위)
  df_pretrain.pkl                         Binding mask 생성
  (1,988,672행)                                       │
                                          ┌───────────┴───────────┐
                                          ▼                       ▼
                                  df_train_final.pkl      df_test_final.pkl
                                  (45,272 pairs)          (17,708 pairs)
```

---

## 6. 주요 주의사항

1. **pairing sentinel**: 기존 pkl 파일의 pairing은 `0`을 "비짝" sentinel로 사용. 위치 0과 쌍을 이루는 뉴클레오타이드의 경우 `pairing[j]=0`이 되어 비짝과 구별 불가. `[3]` 노트북에서 `ss_labels` 기반으로 `-1`로 교정 완료.

2. **energy 타입**: `df_circ_ss.pkl`의 `energy` 컬럼은 `str` 타입 (`"-91.40"`). 사용 시 `float` 변환 필요. `df_pretrain.pkl`에서는 `0.0`으로 통일.

3. **Pretrain ↔ Downstream 서열 겹침**:
   - pretrain ∩ train circRNA: 80% (8,841/11,053)
   - pretrain ∩ test circRNA: 77% (4,959/6,450)
   - 사전학습 효과의 엄밀한 평가를 위해서는 test circRNA를 사전학습에서 제외하는 것을 권장.

4. **Train ↔ Test circRNA 겹침**: 4,633개의 circRNA isoform이 양쪽에 존재. pair 단위 분리는 보장되나 circRNA 서열 수준 leakage 가능성 있음.
