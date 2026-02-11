# CMI-MAC

**Circular RNA-MicroRNA Interaction with Multi-branch Attention and Circular-aware Modeling**

A deep learning framework for predicting circRNA-miRNA binding sites using a novel circular-aware architecture.

## Key Features

- **CircMAC Architecture**: 3-branch hybrid (Attention + Mamba + Circular CNN)
- **Circular-aware Design**: Circular padding, circular relative bias
- **Site-First Approach**: Binding site prediction as main task, binding derived from sites
- **Specialized Pretraining**: CPCL (Circular Permutation CL), BSJ_MLM

## Project Structure

```
cmi_mac/
├── models/              # Model architectures
│   ├── circmac.py       # CircMAC (main model)
│   ├── heads.py         # Task heads (SiteHead, UnifiedSiteHead)
│   ├── model.py         # ModelWrapper
│   └── modules.py       # Embeddings, Attention, CrossAttention
├── trainer.py           # Training loop
├── training.py          # Supervised training CLI
├── pretraining.py       # Self-supervised pretraining CLI
├── scripts/             # Experiment scripts
├── docs/                # Documentation
└── data/                # Datasets (not included in repo)
```

## Quick Start

### Training

```bash
python training.py \
    --model_name circmac \
    --task sites \
    --interaction cross_attention \
    --d_model 128 \
    --n_layer 6 \
    --batch_size 128 \
    --epochs 150 \
    --device 0
```

### Run Experiments

```bash
./scripts/exp1_pretrained_models.sh 0    # GPU 0
./scripts/exp3_encoder_comparison.sh 1   # GPU 1
```

### Check Progress

```bash
python check_results.py --summary
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

## Model Comparison

| Model | Attention | Mamba | CNN | Circular |
|-------|-----------|-------|-----|----------|
| LSTM | X | X | X | X |
| Transformer | O | X | X | X |
| Mamba | X | O | X | X |
| Hymba | O | O | X | X |
| **CircMAC** | **O** | **O** | **O** | **O** |

## Documentation

- [Overview](docs/overview.md) - Architecture and design
- [Experiments](docs/experiments.md) - Experiment design and results
- [Paper Draft](docs/draft.md) - Writing notes

## Requirements

- Python 3.8+
- PyTorch 2.0+
- mamba-ssm
- einops
- transformers
- multimolecule
