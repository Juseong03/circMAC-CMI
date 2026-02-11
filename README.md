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
│   ├── heads.py         # Task heads (UnifiedSiteHead)
│   └── model.py         # ModelWrapper
├── trainer.py           # Training loop
├── training.py          # Training CLI
├── scripts/             # Experiment scripts
├── docs/                # Documentation
└── data/                # Datasets
```

## Quick Start

### Training with Unified Head (Site-First Approach)

```bash
python training.py \
    --model_name circmac \
    --task sites \
    --use_unified_head \
    --binding_pooling mean \
    --d_model 128 \
    --n_layer 6 \
    --batch_size 128 \
    --epochs 150 \
    --device 0
```

### Run All Experiments

```bash
bash scripts/run_all.sh 0  # GPU 0
```

### Check Progress

```bash
python check_results.py --summary
```

## Experiments

| Exp | Description | Script |
|-----|-------------|--------|
| 1 | Pretrained Model Comparison | `exp1_pretrained_comparison.sh` |
| 2 | Encoder Architecture Comparison | `exp2_encoder_comparison.sh` |
| 3 | Ablation Study | `exp3_ablation.sh` |
| 4 | Pretraining Task Analysis | `exp4_pretraining_tasks.sh` |

## Model Comparison

| Model | Attention | Mamba | CNN | Circular |
|-------|-----------|-------|-----|----------|
| LSTM | X | X | X | X |
| Transformer | O | X | X | X |
| Mamba | X | O | X | X |
| Hymba | O | O | X | X |
| TTHymba | O | O | X | X |
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

## License

MIT License
