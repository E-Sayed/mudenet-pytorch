# Optimization Log

> **Purpose:** Track every experiment in the optimization phase — what we tested, why, results, and conclusions. Single source of truth for the journey from baseline to paper-matching metrics.
>
> **Benchmark category:** Cable (MVTec AD) — used throughout until metrics match, then scale to all categories.

---

## Baseline (pre-optimization)

Trained end-to-end for 500 epochs with distilled teacher. No Gaussian smoothing.

| Metric  | Result | Paper  | Gap      |
|---------|--------|--------|----------|
| I-AUROC | 89.2%  | 98.9%  | −9.7 pp  |
| P-AUROC | 96.8%  | 98.3%  | −1.5 pp  |
| PRO     | 82.0%  | 90.6%  | −8.6 pp  |

**Checkpoint:** `runs/mvtec_ad/cable/end_to_end.pt` (500 epochs, seed 42)

---

## Experiment 1: Gaussian Smoothing Sigma Sweep

**Date:** 2026-02-17
**Hypothesis:** Post-inference Gaussian smoothing reduces pixel-level noise, improving image-level scoring (spatial max is less sensitive to single-pixel spikes). The optimal sigma may differ from the initial default of 4.0.
**What changed:** No code or model changes — only the `smoothing_sigma` inference parameter.
**Retraining required:** No

### Results

| Sigma | I-AUROC | P-AUROC | PRO  |
|-------|---------|---------|------|
| 0.0   | 89.2    | 96.8    | 82.0 |
| 2.0   | 90.7    | 96.9    | 82.3 |
| 4.0   | 91.0    | 97.1    | 82.1 |
| 6.0   | 90.5    | 97.2    | 81.6 |
| 8.0   | 90.1    | 97.2    | 80.7 |

### Analysis

- **I-AUROC** peaks at sigma=4.0 (+1.8pp over no smoothing). Higher sigma over-smooths anomaly peaks, reducing discriminative power for image-level classification.
- **P-AUROC** is nearly flat across the range (96.8–97.2%), already close to the paper's 98.3%. Smoothing has minimal effect on pixel ranking.
- **PRO** peaks at sigma=2.0 (82.3%) but the difference from 4.0 is marginal (+0.2pp). Higher sigma actively hurts PRO by blurring defect boundaries, which degrades connected-component overlap.

### Conclusion

Sigma=4.0 is the best overall trade-off for cable. The default was already optimal. Remaining gaps (I-AUROC −7.9pp, PRO −8.5pp) cannot be closed through post-processing — they require model-level changes.

### Code changes

- Added `--smoothing-sigma` CLI argument to evaluate subcommand (`src/mudenet/cli/evaluate.py`)

---

## Experiment 2: Detach Autoencoder Maps + Remove Final Decoder ReLU

**Date:** 2026-02-17
**Hypothesis:** Two independent model-level issues compound to degrade metrics:
1. **A-014 (detach):** L_2 should only update θ_S2. Without `.detach()`, conflicting gradients from L_A (push autoencoder to reconstruct teacher) and L_2 (push autoencoder to be easy for S2 to match) degrade both networks.
2. **A-012 (remove final ReLU):** Teacher embeddings come from bare 1×1 projections (no activation) and can be negative. The decoder's final ReLU clamps outputs ≥ 0, creating a systematic reconstruction floor.

**What changed:**
- `src/mudenet/training/trainer.py` line 155: `autoencoder_maps` detached before `logical_loss`
- `src/mudenet/models/autoencoder.py` Decoder class: removed final `nn.ReLU` from `nn.Sequential`

**Retraining required:** Yes (500 epochs, same distilled teacher)
**Checkpoint:** `runs/mvtec_ad/cable/v2_detach_norelu/end_to_end.pt`

### Results

| Metric  | v1 Baseline | v2 (this exp) | Paper  | v1→v2 gain | Remaining gap |
|---------|-------------|---------------|--------|------------|---------------|
| I-AUROC | 91.0%       | 95.5%         | 98.9%  | +4.5 pp    | −3.4 pp       |
| P-AUROC | 97.1%       | 97.8%         | 98.3%  | +0.7 pp    | −0.5 pp       |
| PRO     | 82.1%       | 84.3%         | 90.6%  | +2.2 pp    | −6.3 pp       |

### Analysis

- **I-AUROC** improved significantly (+4.5pp), confirming these were real model-level issues. The detach fix likely had the larger impact — conflicting gradients were actively harming training.
- **P-AUROC** now within 0.5pp of the paper — essentially closed. The model segments anomalies well.
- **PRO** improved (+2.2pp) but retains the largest gap (−6.3pp). PRO measures per-component overlap, which is sensitive to false positive regions on nominal images and boundary precision on defective images.
- Both changes combined closed ~57% of the I-AUROC gap, ~58% of the P-AUROC gap, and ~27% of the PRO gap.
- Note: paper reports averages over 3 independent runs; our results are single runs.

### Conclusion

Phase 2 delivered substantial gains. P-AUROC is near-closed. I-AUROC and PRO still have meaningful gaps. The remaining PRO gap (−6.3pp) is the primary concern — likely requires changes to the feature extraction / normalization layer (Phase 3+).

---

## Experiment 3: Global Z-Score Normalization (Negative Result)

**Date:** 2026-02-17
**Hypothesis:** Per-sample per-channel z-score normalization washes out magnitude differences between images. Switching to global per-channel statistics (computed over the entire training set) would preserve relative activation differences, providing richer distillation targets and improving downstream metrics.

**What changed:**
- `FeatureExtractor`: added `compute_global_stats(dataloader, device)` — single pass over the training set to compute per-channel mean/std in float64, stored as registered buffers. `forward()` branches on whether global stats are available (backward-compatible).
- `distill.py`: called `compute_global_stats` before `train_distillation`.
- `distillation.py`: saved global stats in teacher checkpoint.

**Re-distillation required:** Yes (distillation targets change)
**Retraining required:** Yes (500 epochs each for distillation and training)

### Results

| Metric  | v2 Baseline | v3 (this exp) | Paper  | v2→v3 change | Remaining gap |
|---------|-------------|---------------|--------|--------------|---------------|
| I-AUROC | 95.5%       | 91.4%         | 98.9%  | −4.1 pp      | −7.5 pp       |
| P-AUROC | 97.8%       | 97.3%         | 98.3%  | −0.5 pp      | −1.0 pp       |
| PRO     | 84.3%       | 81.5%         | 90.6%  | −2.8 pp      | −9.1 pp       |

### Analysis

- All three metrics regressed. I-AUROC dropped below even the v1 baseline (91.0%).
- Per-sample normalization was not the problem — it was actually helping by giving the teacher a consistent, well-scaled target for every image.
- Global normalization introduced high variance in target norms across images. Images with channels that activate far from the dataset mean produced targets with large absolute values, making the distillation task harder and noisier.
- The paper's "z-score based on ImageNet statistics" most likely means "z-score normalization of features from the ImageNet-pretrained backbone" — describing the features' origin, not the statistics' scope.

### Conclusion

**Negative result.** Global z-score normalization is strictly worse than per-sample normalization. Code changes reverted; per-sample normalization confirmed as correct. A-011 resolved.

### Code changes

None retained — all changes reverted via `git restore`.

---

## Experiment 4: (next)

**Planned:** Roadmap item 4 — MaxPool vs AvgPool in stem (A-002). Requires re-distillation.
