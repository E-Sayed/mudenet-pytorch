# Architecture Overview

## Project

**Name:** mudenet-pytorch

**Description:** Production-grade PyTorch reproduction of MuDeNet (Campos-Romero et al., Information Fusion 2026) — an unsupervised multi-patch descriptor network for visual anomaly detection and segmentation in industrial settings.

**Paper:** Campos-Romero, M., Carranza-García, M., Sips, R.-J., & Riquelme, J.C. (2026). MuDeNet: A multi-patch descriptor network for anomaly modeling. *Information Fusion*, 132, 104214.

**Target Users:** Researchers and practitioners working on visual anomaly detection who need a clean, reproducible MuDeNet implementation.

---

## Tech Stack

- **Language:** Python 3.10+
- **Framework:** PyTorch + torchvision
- **Dependencies:** numpy, scipy, scikit-learn, Pillow, tqdm, matplotlib, PyYAML
- **Dev tooling:** pytest, ruff, mypy, pre-commit
- **Packaging:** pyproject.toml with `src/mudenet/` layout

---

## Architecture Summary

MuDeNet is a teacher-student framework with two specialized modules:

### Teacher Network (T)
- Lightweight CNN from paper Figure 2: stem (7×7 conv + AvgPool) + 3 blocks of residual sub-blocks
- Internal channels: 64; output channels: C=128 (via 1×1 projections)
- Block depths: (1, 2, 2) residual blocks per level; kernel sizes: (3, 3, 5)
- Produces 3 spatially-aligned embedding maps (C @ 128×128) at receptive fields 16, 32, 64
- Params: 666,496 per network (2.67 MB); 3 networks share same architecture (T, S1, S2)
- Initialized via knowledge distillation from WideResNet50 (ImageNet-pretrained)
- Frozen during main training

### Structural Module (T + S1)
- Student S1 has the same architecture as T
- Trained to replicate T's outputs on nominal images
- Discrepancies at inference reveal local/structural anomalies

### Logical Module (A + S2)
- Autoencoder A (Figure 3): encoder (6 strided convs: 3→32→64→C→C→C→Z) compresses to Z=32 dim latent; ensemble of L=3 decoders (6 transposed convs each) reconstruct embedding maps
- Student S2 has the same architecture as T, trained against A's outputs
- Bottleneck forces global context — failures reveal logical anomalies

### Inference
- Structural score: ||T - S1|| per level
- Logical score: ||A - S2|| per level
- Min-max normalize both, sum per level, average across levels

---

## Modules

| Module | Location | Purpose |
|--------|----------|---------|
| Config | `src/mudenet/config/` | Typed dataclass schema, YAML loading, argparse overrides |
| Models | `src/mudenet/models/` | Teacher, Autoencoder, WideResNet50 feature extractor |
| Data | `src/mudenet/data/` | Dataset loaders (MVTec AD, LOCO, VisA), transforms |
| Training | `src/mudenet/training/` | Stage 1 distillation, Stage 2 end-to-end training, losses |
| Inference | `src/mudenet/inference/` | Anomaly scoring, normalization, full pipeline |
| Evaluation | `src/mudenet/evaluation/` | AUROC, PRO, sPRO metrics, result reporting |
| Utilities | `src/mudenet/utils/` | Reproducibility, checkpoints, visualization, logging |
| CLI | `src/mudenet/cli/` | Entry points: distill, train, evaluate |

---

## Constraints

- Production-grade code quality (clean API, documentation, error handling)
- Must support evaluation on MVTec AD (15 categories), MVTec LOCO (5 categories), VisA (12 categories)
- Intended for open-source use — code must be readable and well-documented
- No hard version pinning; compatible with recent PyTorch releases

---

## Deliverables

Layered approach:
1. **Code:** Full pipeline for all stages (distillation, training, inference, evaluation)
2. **Pre-distilled teacher weights:** Downloadable checkpoint to skip Stage 1
3. **Fully trained checkpoints:** All 32 categories across 3 datasets

---

## Decisions

Architecture decisions will be documented in `/docs/architecture/decisions/` as they are made.

Use the ADR (Architecture Decision Record) format:
- Context: What situation prompted this decision?
- Decision: What did we decide?
- Consequences: What are the implications?
