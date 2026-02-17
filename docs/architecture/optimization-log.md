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

## Screening Methodology

**Problem:** Full 500-epoch runs take ~90 minutes per seed (distillation + training). Testing each experiment with 3 seeds at full budget is prohibitively slow for iterative exploration.

**Approach:** Two-phase optimization:

- **Phase A — Screening (100 epochs, 3 seeds).** Each experiment is distilled for 100 epochs and trained for 100 epochs, across seeds 42, 123, and 7. Results are evaluated and compared as 3-run averages against the screening baseline. This reduces per-seed time to ~20 minutes (~1 hour per 3-seed experiment). Absolute metrics will be lower than 500-epoch results, but relative differences between experiments are preserved.
- **Phase B — Full run (500 epochs, 3 seeds).** The best configuration from Phase A is trained at full budget to produce paper-comparable metrics.

**Why 100 epochs:** With cable (~224 images, batch_size=8, ~28 steps/epoch), 100 epochs provides ~2,800 gradient steps. This is past the initial rapid-descent phase (epochs 1–30) and well into the steady-improvement phase, where relative ordering of configurations is predictive of final results. Shorter runs (e.g. 10 epochs / ~280 steps) risk being dominated by initialization effects.

**Seeds:** 42, 123, 7 — used consistently across all screening experiments.

---

## Screening Baseline (100 epochs, 3 seeds)

**Date:** 2026-02-17
**Purpose:** Establish 100-epoch reference metrics for the current v2 configuration (detach + no final ReLU, per-sample z-score, sigma=4.0). All subsequent screening experiments are compared against these 3-run averages.

**Config:** v2 codebase (current `main`), `configs/cable_screen.yaml`, 100-epoch distillation + 100-epoch training per seed.

| Seed | I-AUROC | P-AUROC | PRO  |
|------|---------|---------|------|
| 42   | 94.5    | 97.7    | 85.2 |
| 123  | 92.8    | 96.6    | 81.4 |
| 7    | 92.2    | 96.3    | 84.5 |
| **Mean** | **93.2** | **96.9** | **83.7** |

**Observations:**
- Seed 42 is consistently the strongest across all metrics. Our earlier single-seed 500-epoch results (seed 42: 95.5 / 97.8 / 84.3) came from the luckiest seed.
- Seed variance is significant: I-AUROC spans 2.3pp (92.2–94.5), PRO spans 3.8pp (81.4–85.2). This confirms that single-seed comparisons can be misleading.
- 100-epoch seed 42 (94.5 / 97.7 / 85.2) is close to 500-epoch seed 42 (95.5 / 97.8 / 84.3), validating 100 epochs as a screening proxy. PRO is slightly higher at 100 epochs, suggesting possible minor overfitting at longer training.

**Checkpoints:**
- `runs/mvtec_ad/cable/screen_baseline/seed42/`
- `runs/mvtec_ad/cable/screen_baseline/seed123/`
- `runs/mvtec_ad/cable/screen_baseline/seed7/`

---

## Experiment 4: Distillation Resolution Strategy — Downsample Teacher (A-009)

**Date:** (planned — after screening baseline)
**Hypothesis:** The current distillation upsamples the WRN50 target from 64×64 to 128×128 via bilinear interpolation, creating smooth/blurry training targets. The teacher learns to reproduce these blurry features, which propagates to the anomaly maps at inference. This hurts PRO specifically because PRO measures per-component boundary overlap, and blurred boundaries reduce overlap precision. The fact that P-AUROC (rank-based, boundary-insensitive) is near-closed (−0.5pp) while PRO retains a large gap (−6.3pp) is consistent with blur being the bottleneck.

**What changes:**
- `src/mudenet/training/distillation.py`: Instead of upsampling the target to 128×128, downsample each teacher map from 128×128 to 64×64 via `F.avg_pool2d(kernel_size=2)` before the loss. The teacher architecture and its 128×128 inference output are unchanged.

**Re-distillation required:** Yes (distillation loss computation changes)
**Retraining required:** Yes (different distilled teacher)
**Screening protocol:** 100-epoch distillation + 100-epoch training × 3 seeds. Compare 3-run averages to screening baseline.

### Expected impact

- **PRO** is the primary target — sharper distillation targets should produce sharper teacher embeddings, leading to sharper anomaly maps and better boundary overlap.
- **I-AUROC** may improve if sharper features help image-level discrimination.
- **P-AUROC** is already near-closed — minimal movement expected.

### Results (pending)

| Seed | I-AUROC | P-AUROC | PRO  |
|------|---------|---------|------|
| 42   | —       | —       | —    |
| 123  | —       | —       | —    |
| 7    | —       | —       | —    |
| **Mean** | — | — | — |
| **Baseline mean** | 93.2 | 96.9 | 83.7 |
| **Delta** | — | — | — |

**Checkpoints:**
- `runs/mvtec_ad/cable/screen_exp4/seed42/`
- `runs/mvtec_ad/cable/screen_exp4/seed123/`
- `runs/mvtec_ad/cable/screen_exp4/seed7/`

If the result is negative, revert via `git restore` and document.

---

## Optimization Roadmap (revised)

### Phase A — Screening (100 epochs × 3 seeds)

| Priority | Change | Assumption ID | Status |
|----------|--------|---------------|--------|
| ~~1~~ | ~~Sigma tuning~~ | ~~A-016~~ | ~~Done (full run)~~ |
| ~~2~~ | ~~Detach + remove final ReLU~~ | ~~A-014, A-012~~ | ~~Done (full run)~~ |
| ~~3~~ | ~~Global z-score normalization~~ | ~~A-011~~ | ~~Rejected (full run)~~ |
| ~~Baseline~~ | ~~Screening baseline (v2 config)~~ | ~~—~~ | ~~Done~~ |
| **4b** | **Distillation at 64×64 (downsample teacher)** | **A-009** | **Next** |
| 4a | MaxPool vs AvgPool in stem | A-002 | |
| 5 | Add BN to stem | A-005 | |
| 6 | BN/ReLU placement in residual blocks | A-003 | |

### Phase B — Full run (500 epochs × 3 seeds)

Best configuration from Phase A → full training → paper-comparable metrics.
