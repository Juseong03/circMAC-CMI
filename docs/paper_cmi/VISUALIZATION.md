# Binding Site Visualization — 코드 및 결과물 정리

> 최종 업데이트: 2026-05-06

---

## 1. 개요

circRNA–miRNA binding site prediction 결과를 시각화하는 파이프라인.  
크게 **두 단계**로 구성됨:

1. **Inference → CSV 저장** (`plot_binding_visualization.py`)
2. **CSV → Figure 생성** (`plot_from_csv.py`, `plot_combined_cases.py`)

---

## 2. 스크립트 목록

| 파일 | 역할 |
|------|------|
| `run_viz_all_models.sh` | 9개 모델 전체 inference + 시각화 실행 (Step 1+2 통합) |
| `plot_binding_visualization.py` | 모델 inference 수행 → `binding_visualization_*.csv` 저장 |
| `plot_from_csv.py` | CSV로부터 단일 circRNA 시각화 (heatmap / overlay / per_model / bsj_zoom / metrics) |
| `plot_combined_cases.py` | 3개 케이스 × 3개 모델그룹 통합 figure 생성 |
| `compute_thresholds.py` | validation set으로 모델별 F1-optimal threshold 계산 |

---

## 3. 모델 및 Threshold

`model_thresholds_s1.json` (seed=1, validation set F1-optimal):

| 모델 | Threshold |
|------|-----------|
| circmac | 0.8691 |
| mamba | 0.8940 |
| lstm | 0.8223 |
| transformer | 0.8191 |
| hymba | 0.8654 |
| rnabert | 0.0259 |
| rnaernie | 0.3448 |
| rnamsm | 0.1132 |
| rnafm | 0.3698 |

> pretrained RNA LM (rnabert, rnamsm)은 threshold가 매우 낮음 → thr=0.5와 val-thr 간 metrics 차이가 큼

**모델 그룹 정의:**

| 그룹 | 포함 모델 |
|------|-----------|
| `encoder` | circmac, mamba, lstm, transformer, hymba |
| `pretrained` | circmac, rnabert, rnaernie, rnamsm, rnafm |
| `all` | 위 9개 전부 |

---

## 4. Case Study 대상

| Case | isoform_ID | 유전자 | 대표 miRNA | n(binding pairs) |
|------|-----------|--------|-----------|-----------------|
| circCDYL2 | `chr4\|84678168` | CDYL2 | hsa-miR-449a | 4 (34b, 34c, 449a, 449b) |
| circMAPK1 | `chr22\|21799012` | MAPK1 | hsa-miR-12119 | 2 (12119, 4721) |
| circAPP   | `chr21\|25954590` | APP   | hsa-miR-5001-3p | 3 (5001-3p, 1236-3p, 4732-5p) |

**대표 miRNA 선정 기준:** 생물학적 중요성 + circMAC이 잘 맞추는 pair

---

## 5. Metrics 설명

모든 metrics는 **nucleotide-level strict** (tol=0, gap=0) 기준:

| Metric | 설명 |
|--------|------|
| **Recall** | GT binding 위치 중 예측이 맞은 비율 (exact position match) |
| **Precision** | 예측 binding 위치 중 실제 GT인 비율 |
| **F1** | 2 × Precision × Recall / (Precision + Recall) |
| **AUROC** | position-level ROC AUC (threshold 무관) |

> **주의:** AUROC는 per-pair (단일 miRNA–circRNA) 기준으로 계산됨.  
> 대표 pair를 circMAC이 잘 맞추는 것으로 선택했기 때문에 AUROC가 높게 나올 수 있음.  
> 예: circMAPK1 miR-4721에서 min(pred at binding)=0.8119 > max(pred at non-binding)=0.8022 → AUROC=1.0  
> 같은 circRNA의 다른 pair(miR-147b-3p)는 AUROC=0.40.

**Threshold 비교:** Figure 내 bar chart에서 두 버전 표시
- **solid** = val-set optimal threshold (model_thresholds_s1.json)
- **`///` hatched** = threshold=0.5 (고정)

---

## 6. Inference CSV

위치: `docs/paper_cmi/results/{run_tag}/binding_visualization_{tag}_with_pred.csv`

| run_tag | 내용 |
|---------|------|
| `chr4_84678168_84679116_test_binding_only_bsjw20_s1` | circCDYL2, test set, binding pair만 |
| `chr22_21799012_21805850_21807664_test_binding_only_bsjw20_s1` | circMAPK1 |
| `chr21_25954590_25955627_25975070_25975954_25982344_test_binding_only_bsjw20_s1` | circAPP |

CSV 컬럼: `isoform_ID, miRNA_ID, length, position, nucleotide, ground_truth, bsj_adjacent, pred_circmac, pred_mamba, ..., pred_rnafm`

---

## 7. 결과 파일 구조

### 7-1. 통합 Combined Figure (논문용 메인)

위치: `docs/paper_cmi/paper_figures_v2/`

| 파일명 | 내용 | pairs | model group |
|--------|------|-------|-------------|
| `viz_combined_cases.png/pdf` | 3 cases × all 9 models | single rep | all |
| `viz_combined_cases_encoder.png/pdf` | 3 cases × encoder 5 | single rep | encoder |
| `viz_combined_cases_pretrained.png/pdf` | 3 cases × pretrained 5 | single rep | pretrained |
| `viz_combined_cases_all.png/pdf` | 3 cases × all 9 | all pairs avg | all |
| `viz_combined_cases_all_encoder.png/pdf` | 3 cases × encoder 5 | all pairs avg | encoder |
| `viz_combined_cases_all_pretrained.png/pdf` | 3 cases × pretrained 5 | all pairs avg | pretrained |

**Combined figure 구성 (6-row layout):**

```
Row 0: GT heatmap        (miRNA × position, red)
Row 1: circMAC heatmap   (prediction)
Row 2: Circular diagram  (대표 miRNA, circMAC)
Row 3: 5' BSJ zoom       (±50 nt, all models overlay)
Row 4: 3' BSJ zoom       (±50 nt, all models overlay)
Row 5: 2×2 Metrics grid  (Recall / Precision / F1 / AUROC)
         └─ Recall/Prec/F1: grouped bar (solid=val-thr, ///=thr0.5)
         └─ AUROC: single bar (threshold-independent)
```

### 7-2. Heatmap Comparison Figure

| 파일명 | 내용 |
|--------|------|
| `viz_heatmap_comparison.png/pdf` | GT + 9모델 heatmap 나열, 3 cases |
| `viz_heatmap_comparison_encoder.png/pdf` | encoder 5모델 버전 |
| `viz_heatmap_comparison_pretrained.png/pdf` | pretrained 5모델 버전 |
| `viz_heatmap_comparison_all*.png/pdf` | all-pairs 버전 (×3 model groups) |

### 7-3. Circular Comparison Figure

| 파일명 | 내용 |
|--------|------|
| `viz_circular_comparison.png/pdf` | 3 cases 원형 다이어그램 비교 |
| `viz_circular_comparison_encoder.png/pdf` | encoder group |
| `viz_circular_comparison_pretrained.png/pdf` | pretrained group |
| `viz_circular_comparison_all*.png/pdf` | all-pairs 버전 (×3 model groups) |

### 7-4. Per-case 개별 파일

위치: `docs/paper_cmi/paper_figures_v2/{case}/`

| Case 디렉토리 | 대상 | pairs |
|--------------|------|-------|
| `chr4_main/` | circCDYL2 | 대표 pair (miR-449a) |
| `mapk1_main/` | circMAPK1 | 대표 pair (miR-12119) |
| `app_main/` | circAPP | 대표 3 pairs |
| `app_supp/` | circAPP | 전체 binding pairs |

각 case 디렉토리 내 카테고리별 서브폴더:

```
{case}/
├── bsj_zoom/        viz_bsj_zoom_*       BSJ ±50nt 확대 (all models)
├── heatmap/         viz_heatmap_*        GT + model heatmap
├── overlay/         viz_overlay_*        multi-model overlay
├── per_model/       viz_model_{name}_*   모델별 개별 linear + circular
├── per_pair/        viz_pair_{mirna}_*   miRNA pair별 상세 (line plot)
└── site_metrics/    viz_site_metrics_*   Recall/Prec/F1/AUROC bar chart
```

---

## 8. 실행 방법

### Step 1: Inference CSV 생성 (최초 1회)

```bash
# 예시: circCDYL2
./docs/paper_cmi/run_viz_all_models.sh 0 1 "chr4|84678168" 20 0.5 0.3 0 test

# 예시: circMAPK1
./docs/paper_cmi/run_viz_all_models.sh 0 1 "chr22|21799012" 20 0.5 0.3 0 test

# 예시: circAPP
./docs/paper_cmi/run_viz_all_models.sh 0 1 "chr21|25954590" 20 0.5 0.3 0 test
```

> CSV가 이미 있으면 Step 1 생략 가능.

### Step 2: Threshold 계산 (최초 1회)

```bash
python docs/paper_cmi/compute_thresholds.py --seed 1 --device 0 --model_root models_for_viz
# 출력: docs/paper_cmi/model_thresholds_s1.json
```

### Step 3: Combined Figure 생성

```bash
# 단일 대표 pair, encoder + pretrained 그룹
python docs/paper_cmi/plot_combined_cases.py \
    --pairs single --model_group encoder pretrained --mode combined

# 모든 버전 (6개 combined + heatmap + circular)
python docs/paper_cmi/plot_combined_cases.py \
    --pairs both --model_group all encoder pretrained --mode all \
    --out_dir docs/paper_cmi/paper_figures_v2
```

### Step 4: Per-case 개별 Figure 생성

```bash
# circCDYL2 예시 (threshold file 자동 적용)
python docs/paper_cmi/plot_from_csv.py \
    --csv docs/paper_cmi/results/chr4_84678168_84679116_test_binding_only_bsjw20_s1/binding_visualization_chr4_84678168_84679116_with_pred.csv \
    --isoform "chr4|84678168" \
    --bsj_w 20 --zoom_w 50 --tol 0 --gap 0 \
    --plots all \
    --out_dir docs/paper_cmi/paper_figures_v2/chr4_main
```

---

## 9. 주요 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `--bsj_w` | 20 | BSJ adjacent 판정 window (±bp) |
| `--zoom_w` | 50 | BSJ zoom 시각화 범위 (±bp) |
| `--tol` | 0 | GT tolerance (strict=0) |
| `--gap` | 0 | Prediction gap-fill (strict=0) |
| `--top_mirna` | 12 | Heatmap에 표시할 상위 miRNA 수 |
| `--circ_model` | circmac | Combined figure Row 1/2 에 사용할 모델 |
| `--threshold_file` | auto | val-set threshold JSON 경로 |

---

## 10. 알려진 특이사항

- **AUROC ≈ 1.0**: 대표 pair에서 circMAC의 min(pred@binding) > max(pred@non-binding)이 간신히 성립하는 경우 발생. cherry-picked pair라는 점에 유의. 같은 circRNA의 다른 pairs에서는 AUROC가 0.40 수준도 존재.
- **rnabert val-thr = 0.026**: 매우 낮은 threshold → thr=0.5와 metrics 차이 큼 (recall ↓, precision ↑).
- **app_supp vs app_main**: app_supp는 5개 pairs 전체, app_main은 3개 주요 pairs만.
