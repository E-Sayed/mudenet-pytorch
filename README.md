# MuDeNet-PyTorch

Production-grade PyTorch reproduction of **MuDeNet** (Multi-patch Descriptor Network) for unsupervised visual anomaly detection and segmentation.

> Campos-Romero, M. et al. (2026). *MuDeNet: A multi-patch descriptor network for anomaly modeling.* Information Fusion, 132, 104214.

MuDeNet is a knowledge-distillation-based framework that uses multi-scale embedding maps with structural and logical anomaly detection branches. It achieves state-of-the-art results on standard industrial inspection benchmarks.

## Key Features

- **Full MuDeNet pipeline** -- distillation, training, inference, and evaluation
- **Three datasets supported** -- MVTec AD (15 categories), MVTec LOCO (5 categories), VisA (12 categories)
- **Reproducible** -- seeded RNG, deterministic training, per-category augmentation configs matching the paper
- **Production-quality** -- type-hinted, tested (300+ tests), linted, documented
- **Clean CLI** -- three subcommands (`distill`, `train`, `evaluate`) with YAML config and CLI overrides
- **Visualization** -- anomaly map overlays and side-by-side comparison panels

## Installation

**Prerequisites:** Python 3.10+, CUDA-capable GPU recommended (CPU training is possible but slow).

```bash
# Clone the repository
git clone <repository-url>
cd mudenet-pytorch

# Install in development mode
pip install -e ".[dev]"
```

To verify the installation:

```bash
python -m mudenet --help
pytest tests/ -v
```

## Quick Start

MuDeNet training has two stages: (1) distill a teacher network from a pretrained WideResNet-50, then (2) train the full model (student networks + autoencoder) end-to-end.

```bash
# Stage 1: Distill teacher from WideResNet-50
mudenet distill --config configs/mvtec_ad.yaml --device cuda

# Stage 2: Train S1, autoencoder, and S2 end-to-end
mudenet train --config configs/mvtec_ad.yaml --category bottle \
    --checkpoint runs/mvtec_ad/teacher_distilled.pt

# Evaluate on test set
mudenet evaluate --config configs/mvtec_ad.yaml --category bottle \
    --checkpoint runs/mvtec_ad/bottle/end_to_end.pt
```

All commands accept `--device`, `--seed`, and `--output-dir` overrides. See `mudenet <subcommand> --help` for full option lists.

## Project Structure

```
src/mudenet/
├── config/              # YAML config schema and loading
│   ├── schema.py        # Dataclass config definitions
│   └── loading.py       # YAML parsing with CLI overrides
├── models/              # Neural network architectures
│   ├── common.py        # Stem, ResidualBlock shared components
│   ├── teacher.py       # TeacherNetwork (also used for S1, S2)
│   ├── autoencoder.py   # Encoder + DecoderEnsemble + Autoencoder
│   └── feature_extractor.py  # WideResNet-50 feature extraction
├── data/                # Dataset loaders and transforms
│   ├── datasets.py      # MVTecAD, MVTecLOCO, VisA dataset classes
│   ├── transforms.py    # Train/eval/mask transforms
│   └── utils.py         # DataLoader factory with reproducibility
├── training/            # Training loops
│   ├── losses.py        # All loss functions (Eqs. 3, 5, 7, 8, 16)
│   ├── distillation.py  # Stage 1: teacher distillation
│   └── trainer.py       # Stage 2: end-to-end training
├── inference/           # Anomaly scoring
│   ├── scoring.py       # Structural and logical score maps (Eqs. 9-10)
│   └── pipeline.py      # Full inference pipeline with normalization
├── evaluation/          # Metrics and reporting
│   ├── metrics.py       # Image-AUROC, Pixel-AUROC, PRO, sPRO
│   └── reporting.py     # JSON results and formatted tables
├── visualization/       # Anomaly map visualization
│   └── overlay.py       # Heatmap overlays, side-by-side panels
├── utils/               # General utilities
│   └── seed.py          # Reproducibility (seed all RNGs)
└── cli/                 # Command-line interface
    ├── __main__.py      # Subcommand dispatcher
    ├── distill.py       # Stage 1 CLI
    ├── train.py         # Stage 2 CLI
    └── evaluate.py      # Evaluation CLI
```

## Configuration

MuDeNet uses a layered YAML configuration system. The `configs/` directory provides dataset-specific defaults:

| Config File | Dataset | Categories |
|---|---|---|
| `configs/default.yaml` | Generic defaults | -- |
| `configs/mvtec_ad.yaml` | MVTec AD | 15 |
| `configs/mvtec_loco.yaml` | MVTec LOCO | 5 |
| `configs/visa.yaml` | VisA | 12 |

### CLI Overrides

Any config value can be overridden from the command line:

```bash
# Override category, device, and seed
mudenet train --config configs/mvtec_ad.yaml \
    --category hazelnut \
    --device cpu \
    --seed 123
```

### Configuration Sections

| Section | Key Parameters |
|---|---|
| `model` | `num_channels` (128), `latent_dim` (32), `num_levels` (3), `image_size` (256) |
| `training` | `num_epochs` (500), `batch_size` (8), `learning_rate` (1e-3), `seed` (42) |
| `distillation` | `backbone` (wide_resnet50_2), `num_epochs` (500) |
| `data` | `dataset_type`, `data_root`, `category`, `augmentations` |
| `inference` | `normalization` (min_max), `validation_ratio` (0.1) |

## Datasets

### MVTec AD

Download from [mvtec.com](https://www.mvtec.com/company/research/datasets/mvtec-ad).

```
data/mvtec_ad/
├── bottle/
│   ├── train/good/
│   ├── test/good/
│   ├── test/broken_large/
│   ├── test/broken_small/
│   ├── test/contamination/
│   └── ground_truth/broken_large/
├── cable/
│   └── ...
└── ... (15 categories total)
```

**Categories:** bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper

### MVTec LOCO

Download from [mvtec.com](https://www.mvtec.com/company/research/datasets/mvtec-loco).

```
data/mvtec_loco/
├── breakfast_box/
│   ├── train/good/
│   ├── validation/good/
│   ├── test/good/
│   ├── test/logical_anomalies/
│   ├── test/structural_anomalies/
│   └── ground_truth/...
└── ... (5 categories total)
```

**Categories:** breakfast_box, juice_bottle, pushpin, screw_bag, splicing_connector

### VisA

Download from [GitHub](https://github.com/amazon-science/spot-diff).

```
data/visa/
├── candle/
│   ├── train/good/
│   ├── test/good/
│   ├── test/bad/
│   └── ground_truth/...
└── ... (12 categories total)
```

**Categories:** candle, capsules, cashew, chewinggum, fryum, macaroni1, macaroni2, pcb1, pcb2, pcb3, pcb4, pipe_fryum

## Data Augmentation

Per-category augmentation settings follow the paper (Tables A.16--A.18). The YAML configs include the full augmentation mapping. Key augmentations:

| Augmentation | Description |
|---|---|
| `horizontal_flip` | Random horizontal flip |
| `vertical_flip` | Random vertical flip |
| `rotation` | Random rotation |
| `color_jitter` | Brightness, contrast, saturation jitter |

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{camposromero2026mudenet,
  title     = {MuDeNet: A multi-patch descriptor network for anomaly modeling},
  author    = {Campos-Romero, M. and others},
  journal   = {Information Fusion},
  volume    = {132},
  pages     = {104214},
  year      = {2026},
  publisher = {Elsevier},
}
```

## License

This project is licensed under the MIT License. See the [pyproject.toml](pyproject.toml) for details.
