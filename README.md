# MuDeNet-PyTorch

PyTorch implementation of **MuDeNet** — a teacher-student anomaly detection framework based on multi-scale patch descriptors.

> Campos-Romero, M., Carranza-García, M., Sips, R.-J., & Riquelme, J.C. (2026).
> *MuDeNet: A multi-patch descriptor network for anomaly modeling.*
> Information Fusion, 132, 104214.

---

**Status: work in progress.** The full pipeline runs (distillation → training → inference → evaluation), but results don't match the paper yet — still optimizing. Progress is tracked in the [optimization log](docs/architecture/optimization-log.md), and architectural decisions made under uncertainty are recorded in the [assumptions register](docs/architecture/assumptions-register.md).

---

## What is MuDeNet?

MuDeNet detects anomalies in images by training lightweight CNNs to mimic a distilled teacher network. At test time, anything the students fail to reconstruct is flagged as anomalous. It has two branches:

- **Structural branch** — a student network (S1) trained to match the teacher (T) directly. Catches local defects like scratches or dents.
- **Logical branch** — an autoencoder (A) bottlenecks the teacher's embeddings, and a second student (S2) learns the autoencoder's outputs. The bottleneck forces global reasoning, so this branch picks up things like missing components or wrong arrangements.

Both branches produce multi-scale anomaly maps that get fused into a single score per pixel.

The method targets industrial inspection datasets: MVTec AD, MVTec LOCO, and VisA (32 categories total).

## Installation

Requires Python 3.10+ and a CUDA GPU (CPU works but is painfully slow).

```bash
git clone https://github.com/E-Sayed/mudenet-pytorch.git
cd mudenet-pytorch
pip install -e ".[dev]"

python -m mudenet --help
```

## Usage

Two stages: distill a teacher from a pretrained WideResNet-50, then train the full model (S1 + autoencoder + S2) against the frozen teacher.

```bash
# Stage 1: distill teacher
mudenet distill --config configs/mvtec_ad.yaml --device cuda

# Stage 2: train the model
mudenet train --config configs/mvtec_ad.yaml --category bottle \
    --checkpoint runs/mvtec_ad/teacher_distilled.pt

# Evaluate
mudenet evaluate --config configs/mvtec_ad.yaml --category bottle \
    --checkpoint runs/mvtec_ad/bottle/end_to_end.pt --visualize
```

`--visualize` saves side-by-side panels (original / ground truth / anomaly heatmap) per test image. All commands accept `--device`, `--seed`, `--output-dir`. See `mudenet <command> --help`.

## Configuration

YAML configs in `configs/`. Base settings in `default.yaml`, per-dataset overrides:

| Config | Dataset | Categories |
|---|---|---|
| `configs/mvtec_ad.yaml` | MVTec AD | 15 |
| `configs/mvtec_loco.yaml` | MVTec LOCO | 5 |
| `configs/visa.yaml` | VisA | 12 |

CLI overrides:

```bash
mudenet train --config configs/mvtec_ad.yaml --category hazelnut --seed 123
```

Per-category augmentation settings follow Tables A.16–A.18 in the paper.

## Datasets

Download separately:

- **MVTec AD** (15 categories) — [mvtec.com](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- **MVTec LOCO** (5 categories) — [mvtec.com](https://www.mvtec.com/company/research/datasets/mvtec-loco)
- **VisA** (12 categories) — [GitHub](https://github.com/amazon-science/spot-diff)

Expected layout (MVTec AD example):

```
data/mvtec_ad/
├── bottle/
│   ├── train/good/
│   ├── test/good/
│   ├── test/broken_large/
│   └── ground_truth/broken_large/
├── cable/
│   └── ...
└── ...
```

## Project layout

```
src/mudenet/
├── models/         # Teacher network, autoencoder, WRN-50 feature extractor
├── training/       # Distillation (stage 1), end-to-end training (stage 2), losses
├── inference/      # Anomaly scoring and full inference pipeline
├── evaluation/     # Metrics (image-AUROC, pixel-AUROC, PRO, sPRO) and reporting
├── data/           # Dataset classes and transforms
├── config/         # YAML schema and loading
├── visualization/  # Heatmap overlays
├── utils/          # Seed management, etc.
└── cli/            # Entry points (distill, train, evaluate)
```

## Notation mapping

See [NOTATION.md](NOTATION.md) for paper symbol → code identifier mapping.

## Citation

```bibtex
@article{camposromero2026mudenet,
  title   = {MuDeNet: A multi-patch descriptor network for anomaly modeling},
  author  = {Campos-Romero, M. and Carranza-Garc{\'\i}a, M. and Sips, R.-J. and Riquelme, J.C.},
  journal = {Information Fusion},
  volume  = {132},
  pages   = {104214},
  year    = {2026},
}
```

## License

MIT. See [pyproject.toml](pyproject.toml).
