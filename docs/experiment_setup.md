# Experiment Setup (Final — Paper Version)

> Last updated: 2026-03-05
> CircMAC final version: circular=False (no circular padding), circular_rel_bias=ON

---

## Overview

| Exp | Research Question | Description | Runs | Script |
|-----|-----------------|-------------|------|--------|
| **1** | RQ1: CircMAC vs other encoders | Encoder Architecture Comparison | **15** | `exp1_encoders.sh` |
| **2** | RQ2: Best pretraining strategy | Pretraining Task Analysis | **27** | `exp2_s{N}_gpu{N}.sh` |
| **3** | RQ3: CircMAC-PT vs pretrained RNA models | Pretrained Model Comparison (fair + max) | **48** | `exp3_fair.sh` + `exp3_max.sh` |
| **4** | RQ4: CircMAC component contributions | Ablation Study | **24** | `exp4_ablation.sh` |
| **5** (supp) | RQ5: Best interaction mechanism | Interaction Mechanism Comparison | **9** | `exp5_interaction.sh` |
| **6** (supp) | RQ6: Best site head | Site Head Structure Comparison | **6** | `exp6_site_head.sh` |
| | | **Total** | **129** | |

All scripts in: `scripts/final/`

---

## Final CircMAC Architecture

```
CircMAC (Final):
  ├─ Attention branch     (circular relative bias = ON)
  ├─ Mamba branch
  ├─ CNN branch           (circular padding = OFF ← ablation showed OFF is better)
  └─ 3-branch Router
```

**Key changes from preliminary:**
- `circular=False`: circular padding in CNN removed (ablation Exp4 showed +0.28pp F1)
- Gradient bug in pretraining fixed (`.item()` bug in UncertaintyWeightingLoss)

---

## Dataset

| | Train | Test |
|--|-------|------|
| Total pairs | 45,272 | 17,708 |
| Positive (binding=1) | 19,674 | 7,838 |
| **Sites task** (binding=1 only) | **19,674** | **7,838** |

**Sequence lengths** (circRNA): mean=601nt, min=150nt, max=1000nt
**miRNA lengths**: mean=22nt (range: 16–28nt)

### Data Availability by max_len

| max_len | Train samples | % |
|---------|--------------|---|
| 440 (RNABERT limit) | ~7,400 | 37.6% |
| 510 (RNAErnie limit) | ~9,479 | 48.2% |
| 1,022 (CircMAC / RNA-FM / RNA-MSM) | 19,674 | 100% |

### Pretraining Data (Exp2)

| File | Description | Samples |
|------|-------------|---------|
| `df_circ_ss_5.pkl` | 5 stochastic secondary structures per circRNA | ~149,249 |

---

## Task

All experiments use **task = sites** (token-level binding site prediction).

```
Input:  circRNA sequence (c_1,...,c_L) + miRNA sequence (m_1,...,m_M)
Output: Site labels y = (y_1,...,y_L), y_i ∈ {0,1}
```

---

## Common Hyperparameters

| Parameter | Value | Note |
|-----------|-------|------|
| d_model | 128 | |
| n_layer | 6 | |
| optimizer | adamw | |
| lr | 1e-4 | |
| epochs | 150 | |
| earlystop | 20 | |
| seeds | 1, 2, 3 | report mean ± std |
| interaction | cross_attention | fixed for all exps |
| target_model | rnabert | miRNA encoder (frozen) |
| site_head | conv1d | except Exp6 |
| max_len | 1022 | except Exp3 fair |
| batch_size | 128 | see per-exp notes |

---

## Evaluation Metrics

### Primary (Sites Task)
| Metric | Note |
|--------|------|
| **F1 (macro)** | Main metric |
| **F1 (positive)** | Binding class F1 |
| AUROC | |
| AUPRC | |
| MCC | Matthews Correlation Coefficient |
| Span-F1 | F1 on contiguous binding spans |

### Threshold
- Sweep [0.1, 0.9] on validation set, select by F1 macro

---

## Experiment 1: Encoder Architecture Comparison (RQ1)

> Which encoder architecture is best for circRNA binding site prediction?

### Models (all from scratch, no pretraining)

| Model | Type | Key Feature |
|-------|------|-------------|
| LSTM | RNN | Bidirectional sequential processing |
| Transformer | Attention | Global self-attention |
| Mamba | SSM | State Space Model |
| Hymba | Hybrid | Attention + Mamba |
| **CircMAC** | Hybrid | **Attention + Mamba + CNN + Circular-aware** |

### Setup
- batch_size: 128 (Transformer: 64 to avoid OOM)
- **5 models × 3 seeds = 15 runs**

```bash
./scripts/final/exp1_encoders.sh [GPU_ID]
```

### Result Table (LaTeX)
```latex
\begin{table}
\caption{Encoder architecture comparison (from scratch, task=sites)}
\begin{tabular}{llccccc}
\toprule
Model & Type & $F_1^{macro}$ & $F_1^{pos}$ & AUROC & AUPRC & $F_1^{span}$ \\
\midrule
LSTM          & RNN     & & & & & \\
Transformer   & Attn    & & & & & \\
Mamba         & SSM     & & & & & \\
Hymba         & Hybrid  & & & & & \\
\textbf{CircMAC} & \textbf{Hybrid} & & & & & \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Experiment 2: Pretraining Strategy Analysis (RQ2)

> Which self-supervised pretraining objectives benefit circRNA binding site prediction?

### Configs

| Config | Pretrain Tasks | Flags | Server |
|--------|---------------|-------|--------|
| No-PT | None (baseline) | — | S1-G0 |
| MLM | Masked LM | `--mlm` | S1-G0 |
| MLM+NTP | + Next Token Pred | `--mlm --ntp` | S1-G1 |
| MLM+SSP | + Secondary Structure | `--mlm --ssp` | S2-G0 |
| MLM+CPCL | + Circular Perm. CL | `--mlm --cpcl` | S2-G1 |
| MLM+Pair | + Base Pairing Matrix | `--mlm --pairing` | S3-G0 |
| **MLM+NTP+CPCL+Pair** | Combined | `--mlm --ntp --cpcl --pairing` | S3-G1 |

### Pretraining Hyperparameters
| Parameter | Value |
|-----------|-------|
| data | df_circ_ss_5 (5 SS per RNA, ~149K samples) |
| batch_size | 128 |
| lr | 5e-4 |
| w_decay | 0.01 |
| optimizer | adamw |
| epochs | 200 |
| earlystop | 30 |
| max_len | 1022 |
| seed | 42 (single pretrain run) |

### Fine-tuning: same as common HP (lr=1e-4, bs=128, epochs=150)

### Run Count
- Pretrain: 6 runs (one per PT config)
- Finetune: 7 configs × 3 seeds = 21 runs
- **Total: 27 runs**

```bash
# Server distribution:
./scripts/final/exp2_s1_gpu0.sh 0   # No-PT + MLM (7 runs)
./scripts/final/exp2_s1_gpu1.sh 1   # MLM+NTP (4 runs)
./scripts/final/exp2_s2_gpu0.sh 0   # MLM+SSP (4 runs)
./scripts/final/exp2_s2_gpu1.sh 1   # MLM+CPCL (4 runs)
./scripts/final/exp2_s3_gpu0.sh 0   # MLM+Pair (4 runs)
./scripts/final/exp2_s3_gpu1.sh 1   # MLM+NTP+CPCL+Pair (4 runs)
```

### Result Table (LaTeX)
```latex
\begin{table}
\caption{Effect of pretraining strategy on binding site prediction}
\begin{tabular}{lcccc}
\toprule
Pretrain Config & $F_1^{macro}$ & $F_1^{pos}$ & AUROC & $\Delta F_1$ \\
\midrule
No pretrain & & & & baseline \\
\midrule
MLM & & & & \\
MLM + NTP & & & & \\
MLM + SSP & & & & \\
MLM + CPCL & & & & \\
MLM + Pairing & & & & \\
\textbf{MLM + NTP + CPCL + Pair} & & & & \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Experiment 3: Pretrained RNA Model Comparison (RQ3)

> Does CircMAC with circRNA-specific pretraining outperform general-purpose RNA models?

### Models

| Model | Source | Pretrain Data | Hidden | Max Len | Params |
|-------|--------|---------------|--------|---------|--------|
| RNABERT | multimolecule | RNAcentral | 120 | 440 | ~1M |
| RNA-FM | multimolecule | RNAcentral | 640 | 1024 | ~95M |
| RNAErnie | multimolecule | RNAcentral | 768 | 512 | ~86M |
| RNA-MSM | multimolecule | RNAcentral | 768 | 1024 | ~96M |
| **CircMAC-PT** | **Ours** | **circRNA** | 128 | 1022 | ~3M |

### Part 1 — Fair (max_len=440, identical data)

| Mode | Models | Runs |
|------|--------|------|
| Frozen | RNABERT, RNA-FM, RNAErnie, RNA-MSM | 12 |
| Trainable | RNABERT, RNA-FM, RNAErnie, RNA-MSM | 12 |
| Fine-tuned | CircMAC-PT | 3 |
| **Subtotal** | | **27** |

### Part 2 — Max (model-specific max_len, skip RNABERT)

| Mode | Models | Runs |
|------|--------|------|
| Frozen | RNA-FM, RNAErnie, RNA-MSM | 9 |
| Trainable | RNA-FM (bs=16), RNAErnie, RNA-MSM (bs=16) | 9 |
| Fine-tuned | CircMAC-PT | 3 |
| **Subtotal** | | **21** |

**Total Exp3: 48 runs**
Note: Exp3 CircMAC-PT requires best PT config from Exp2.

```bash
./scripts/final/exp3_fair.sh [GPU_ID] [BEST_PT_CONFIG]  # 27 runs
./scripts/final/exp3_max.sh  [GPU_ID] [BEST_PT_CONFIG]  # 21 runs
# BEST_PT_CONFIG default: "mlm_ntp_cpcl_pair"
```

---

## Experiment 4: Ablation Study (RQ4)

> How does each CircMAC component contribute?

### Configurations

| Config | Attn | Mamba | CNN | Circ. Bias | Flags |
|--------|:----:|:-----:|:---:|:----------:|-------|
| **Full CircMAC** | O | O | O | O | (none) |
| w/o Attention | X | O | O | — | `--no_attn` |
| w/o Mamba | O | X | O | O | `--no_mamba` |
| w/o CNN | O | O | X | O | `--no_conv` |
| w/o Circ. Bias | O | O | O | X | `--no_circular_rel_bias` |
| Attention only | O | X | X | O | `--no_mamba --no_conv` |
| Mamba only | X | O | X | — | `--no_attn --no_conv` |
| CNN only | X | X | O | — | `--no_attn --no_mamba` |

Note: Circular padding is OFF by default (design decision from preliminary ablation).

**8 configs × 3 seeds = 24 runs**

```bash
./scripts/final/exp4_ablation.sh [GPU_ID]
```

### Result Table (LaTeX)
```latex
\begin{table}
\caption{Ablation study of CircMAC components}
\begin{tabular}{lccccc}
\toprule
Configuration & $F_1^{macro}$ & $F_1^{pos}$ & AUROC & $F_1^{span}$ & $\Delta F_1$ \\
\midrule
\textbf{Full CircMAC} & & & & & — \\
\midrule
\multicolumn{6}{l}{\textit{Branch ablation}} \\
\quad w/o Attention   & & & & & \\
\quad w/o Mamba       & & & & & \\
\quad w/o CNN         & & & & & \\
\midrule
\multicolumn{6}{l}{\textit{Circular feature ablation}} \\
\quad w/o Circular Bias & & & & & \\
\midrule
\multicolumn{6}{l}{\textit{Single branch}} \\
\quad Attention only  & & & & & \\
\quad Mamba only      & & & & & \\
\quad CNN only        & & & & & \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Experiment 5: Interaction Mechanism (RQ5, Supplementary)

> Which circRNA-miRNA interaction mechanism is most effective?

| Mechanism | Flag |
|-----------|------|
| Concatenation | `--interaction concat` |
| Element-wise | `--interaction elementwise` |
| **Cross-attention** | `--interaction cross_attention` |

**3 × 3 seeds = 9 runs**

```bash
./scripts/final/exp5_interaction.sh [GPU_ID]
```

---

## Experiment 6: Site Head Structure (RQ6, Supplementary)

> Conv1D vs Linear classifier for token-level site prediction?

| Head | Flag |
|------|------|
| Linear | `--site_head_type linear` |
| **Conv1D** | `--site_head_type conv1d` |

**2 × 3 seeds = 6 runs**

```bash
./scripts/final/exp6_site_head.sh [GPU_ID]
```

---

## Server Distribution (3 servers × 2 GPUs)

| Server | GPU 0 | GPU 1 |
|--------|-------|-------|
| **Server 1** | Exp1: Encoder comparison (15) | Exp4: Ablation (24) |
| **Server 2** | Exp2 S2-G0: MLM+SSP (4) + Exp3 Fair part (13) | Exp2 S2-G1: MLM+CPCL (4) + Exp3 Max (17) |
| **Server 3** | Exp2 S3-G0: MLM+Pair (4) + Exp5+6 (15) | Exp2 S3-G1: MLM+NTP+CPCL+Pair (4) + Exp2 S1-G0: baseline+MLM (7) |

> Note: Exp3 CircMAC-PT rows require Exp2 best model. Run Exp3 without CircMAC-PT first (24 runs), then add CircMAC-PT rows after Exp2 finishes.

### Execution Order

```
Immediately (independent):
  Server 1 GPU 0: Exp1 (exp1_encoders.sh)
  Server 1 GPU 1: Exp4 (exp4_ablation.sh)
  Server 2 GPU 0: Exp2 S2-G0 (exp2_s2_gpu0.sh)
  Server 2 GPU 1: Exp2 S2-G1 (exp2_s2_gpu1.sh)
  Server 3 GPU 0: Exp2 S3-G0 (exp2_s3_gpu0.sh) → then Exp5+6
  Server 3 GPU 1: Exp2 S3-G1 (exp2_s3_gpu1.sh)
  + Exp2 S1-G0 and S1-G1 somewhere

After Exp2 completes (need best PT model):
  Exp3 CircMAC-PT rows (exp3_fair.sh, exp3_max.sh)

Exp3 non-CircMAC rows can run anytime:
  exp3_fair.sh → skip CircMAC-PT if PT_PATH not ready (auto-warns)
  exp3_max.sh  → same
```

---

## Exp Name Convention

| Exp | Pattern | Example |
|-----|---------|---------|
| Exp1 | `exp1_{model}_s{seed}` | `exp1_circmac_s1` |
| Exp2 PT | `exp2_pt_{config}` | `exp2_pt_mlm_cpcl` |
| Exp2 FT | `exp2_{config}_{task}_s{seed}` | `exp2_mlm_cpcl_sites_s1` |
| Exp2 base | `exp2_nopt_{task}_s{seed}` | `exp2_nopt_sites_s2` |
| Exp3 fair | `exp3_fair_{mode}_{model}_s{seed}` | `exp3_fair_frozen_rnafm_s1` |
| Exp3 max | `exp3_max_{mode}_{model}_s{seed}` | `exp3_max_trainable_rnamsm_s3` |
| Exp4 | `exp4_{config}_s{seed}` | `exp4_no_attn_s2` |
| Exp5 | `exp5_{interaction}_s{seed}` | `exp5_cross_attention_s1` |
| Exp6 | `exp6_{head}_s{seed}` | `exp6_conv1d_s3` |

---

## Total Run Summary

| Experiment | Main Runs | Notes |
|------------|-----------|-------|
| Exp1 Encoder | 15 | |
| Exp2 Pretraining | 27 (6 PT + 21 FT) | |
| Exp3 Fair | 27 | |
| Exp3 Max | 21 | CircMAC-PT rows need Exp2 |
| Exp4 Ablation | 24 | |
| **Main Total** | **114** | |
| Exp5 Interaction | 9 | Supplementary |
| Exp6 Site Head | 6 | Supplementary |
| **Grand Total** | **129** | |

---

## Reproducibility Checklist

- [x] Seeds: 1, 2, 3 (report mean ± std)
- [x] Data splits: Fixed pkl files (`df_train_final.pkl`, `df_test_final.pkl`)
- [x] Hyperparameters: Documented above
- [x] Scripts: `scripts/final/`
- [x] CircMAC: `circular=False` (no circular CNN padding)
- [x] Pretraining: gradient bug fixed (no `.item()` in loss accumulation)
- [ ] Environment: `requirements.txt`
- [ ] Hardware: GPU model, VRAM
- [ ] Model parameter counts
