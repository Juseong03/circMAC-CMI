# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CMI-MAC is a deep learning project for predicting **circRNA-miRNA binding sites and interactions**. It supports multiple neural network architectures (Mamba, Transformer, LSTM, Hymba, CircMAC) and both supervised training and self-supervised pretraining.

## Commands

### Supervised Training
```bash
# Single run
python training.py --model_name circmac --device 0 --task sites --seed 1 --d_model 128 --batch_size 128 --interaction cross_attention --verbose

# Batch run (binding, sites, both tasks with seeds 1,2,3)
./run_training.sh <model_name> <device> <rc:True|False>
```

### Self-Supervised Pretraining
```bash
# Task keys: mlm, ntp, ssp, ssl, sslm, cpcl, bsj_mlm, mlm_cpcl_bsj_pair, full, etc.
./run_pretraining.sh <data_file:df_circ_ss|df_circ_ss_5> <model_name:circmac> <task_key> <device> <exp_name>

# Recommended for circRNA:
./run_pretraining.sh df_circ_ss circmac mlm_cpcl_bsj_pair 0 experiment_name
```

### Combined Training with Target Model
```bash
./run_both.sh <model_name> <device> <target_model:rnabert> <exp_name> <batch_size>
```

## Architecture

```
data.py (KmerTokenizer, CircRNABindingSitesDataset, CircRNASelfDataset)
    ↓
models/model.py (ModelWrapper)
    ├─ Backbone: circmac, mamba, transformer, hymba, lstm, rnabert, etc.
    ├─ Interaction: concat, elementwise, cross_attention
    └─ Task Heads: BindingHead (binary), SiteHead (token-level), CircularPairingHead
    ↓
trainer.py (train/pretrain loops, evaluation, checkpointing)
```

### Key Model Files
- `models/circmac.py`: **CircMAC v3** - HyMBA + Circular CNN (circRNA specialized)
- `models/hymba.py`: Hymba (Attention + Mamba hybrid)
- `models/model.py`: ModelWrapper that combines backbone + heads
- `models/modules.py`: Embeddings, attention, normalization components
- `models/heads.py`: BindingHead, SiteHead, SSLHead, PairingHead, CircularPairingHead

### Tasks
- **binding**: Binary classification (does the pair bind?)
- **sites**: Token-level classification (which positions bind?) — **main focus**
- **both**: Multi-task learning (binding + sites)

### Pretraining Objectives
- **MLM**: Masked Language Modeling
- **NTP**: Next Token Prediction
- **SSP**: Secondary Structure Prediction
- **ss_labels / ss_labels_multi**: Structure label prediction
- **pairing**: Base pairing matrix reconstruction (CircularPairingHead for circmac)
- **CPCL**: Circular Permutation Contrastive Learning (circRNA specialized)
- **BSJ_MLM**: Back-Splice Junction focused MLM (circRNA specialized)

## Data

- Training data: `./data/df_train_final.pkl`, `./data/df_test_final.pkl`
- RBP data: `./data/binding_RBP.json`
- Pretraining data: `df_circ_ss`, `df_circ_ss_5` (with secondary structure)

## Configuration

Model configs are defined in `utils_config.py`:
- `CircMACConfig`: d_model, n_layer, n_heads, circular, use_multiscale, conv_kernel_size
- `MambaConfig`: d_model, n_layer, d_state, d_conv, expand
- `Transformer2Config`: d_model, n_layer, n_heads, d_head
- `PretrainedConfig`: for RNABERT, RNAErnie, RNAFM wrappers

Default hyperparameters: `d_model=128`, `batch_size=128`, `n_layer=6`, `max_len=1022`

## CircMAC Ablation Flags
```bash
--no_circular_window    # Disable circular padding in CNN
--no_circular_rel_bias  # Disable circular relative bias in attention
--no_attn               # Disable Attention branch
--no_mamba              # Disable Mamba branch
--no_conv               # Disable Conv branch
```

## Experiments
| Exp | Description | Script |
|-----|-------------|--------|
| 1 | Pretrained Model Comparison | `exp1_pretrained_models.sh` |
| 2 | Pretraining Strategy | `exp2_pretraining.sh` |
| 3 | Encoder Architecture Comparison | `exp3_encoder_comparison.sh` |
| 4 | CircMAC Ablation Study | `exp4_ablation.sh` |
| 5 | Interaction Mechanism | `exp5_interaction.sh` |
| 6 | Site Head Structure | `exp6_site_head.sh` |

## Documentation
- `docs/overview.md`: Project overview and architecture
- `docs/experiments.md`: Experiment design and schedule
- `docs/writings_for_paper.md`: Paper writing materials and templates
