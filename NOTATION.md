# Paper-to-Code Notation Mapping

Reference for reading the code alongside the paper:
> Campos-Romero, M. et al. (2026). MuDeNet: A multi-patch descriptor network for anomaly modeling. *Information Fusion*, 132, 104214.

## Networks

| Paper | Code | Description |
|-------|------|-------------|
| T | `TeacherNetwork` | Teacher network (stem + 3 residual blocks) |
| S1 | `TeacherNetwork` (instance) | Structural student — same architecture as T |
| S2 | `TeacherNetwork` (instance) | Logical student — same architecture as T |
| A | `Autoencoder` | Asymmetric autoencoder (encoder + decoder ensemble) |
| E | `Encoder` | Encoder subnetwork of A |
| D | `Autoencoder.decoders` (`ModuleList[Decoder]`) | Ensemble of L decoders in A |

## Tensors and Variables

| Paper | Code | Shape | Description |
|-------|------|-------|-------------|
| I | `image` / `images` | (B, 3, 256, 256) | Input RGB image |
| X^l | `embedding_maps[l]` | (B, C, H, W) | Teacher embedding map at level l |
| X^l_S1 | `student1_maps[l]` | (B, C, H, W) | Structural student embedding at level l |
| X^l_A | `autoencoder_maps[l]` | (B, C, H, W) | Autoencoder reconstruction at level l |
| X^l_S2 | `student2_maps[l]` | (B, C, H, W) | Logical student embedding at level l |
| y | `latent` | (B, Z) | Autoencoder latent vector |
| E | `distill_target` | (B, C, H1, W1) | Pre-distillation target from WideResNet50 |
| S^l_str | `structural_scores[l]` | (B, H, W) | Structural anomaly score at level l |
| S^l_log | `logical_scores[l]` | (B, H, W) | Logical anomaly score at level l |
| S | `anomaly_map` | (B, H, W) | Final fused anomaly score map |

## Hyperparameters

| Paper | Code | Default | Description |
|-------|------|---------|-------------|
| C | `num_channels` | 128 | Embedding channel dimensionality |
| Z | `latent_dim` | 32 | Autoencoder latent space dimensionality |
| L | `num_levels` | 3 | Number of embedding levels / receptive fields |
| — | `receptive_fields` | [16, 32, 64] | Receptive field sizes per level |
| — | `image_size` | 256 | Input image size (square) |

## Equations

| Equation | Code Location | Description |
|----------|---------------|-------------|
| Eq. 1 | `TeacherNetwork.forward()` | Teacher produces L embedding maps |
| Eq. 2 | `TeacherNetwork.forward()` | Student S1 produces L embedding maps |
| Eq. 3 | `losses.py` | Structural loss: Frobenius norm T vs S1 |
| Eq. 4 | `Autoencoder.forward()` | Encoder + decoder reconstruction |
| Eq. 5 | `losses.py` | Autoencoder loss: Frobenius norm T vs A |
| Eq. 6 | `TeacherNetwork.forward()` | S2 forward pass (same arch as T) |
| Eq. 7 | `losses.py` | Logical student loss: Frobenius norm A vs S2 |
| Eq. 8 | `losses.py` | Composite end-to-end loss |
| Eq. 9 | `scoring.py` | Structural anomaly score |
| Eq. 10 | `scoring.py` | Logical anomaly score |
| Eq. 11 | `pipeline.py` | Per-level combined score |
| Eq. 12 | `pipeline.py` | Final averaged anomaly map |
| Eq. 13–15 | `feature_extractor.py` | Pre-distillation feature fusion |
| Eq. 16 | `distillation.py` | Teacher distillation loss |

## Dimension Conventions

- `B` — batch size
- `C` — channels (embedding dimensionality, default 128)
- `H, W` — spatial height and width (embedding maps: 128×128)
- `L` — number of levels (default 3)
- `Z` — latent dimensionality (default 32)
