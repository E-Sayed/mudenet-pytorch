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

**Date:** 2026-02-17
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

### Results — REJECTED

| Seed | I-AUROC | P-AUROC | PRO  |
|------|---------|---------|------|
| 42   | 92.4    | 96.6    | 79.4 |
| 123  | 93.0    | 95.9    | 79.1 |
| 7    | 89.7    | 95.7    | 78.9 |
| **Mean** | **91.7** | **96.1** | **79.1** |
| **Baseline mean** | 93.2 | 96.9 | 83.7 |
| **Delta** | **−1.5** | **−0.8** | **−4.6** |

**Outcome:** All three metrics regressed. PRO dropped by 4.6pp — a uniform regression across all seeds (85.2→79.4, 81.4→79.1, 84.5→78.9), well outside seed variance. The hypothesis was wrong: bilinear upsampling of the target was not the bottleneck for PRO. The teacher needs the full 128×128 supervision signal; constraining the loss to 64×64 removes useful fine-scale gradient information.

**Action:** Code reverted via `git restore`. No changes retained.

---

## Experiment 5.5: Fix Cable Augmentation (A-017)

**Date:** 2026-02-18
**Hypothesis:** The cable config has two factual errors vs the paper's Table A.16: vertical flip is enabled (should be off — cables have a natural vertical orientation) and color jitter is disabled (should be on — cable anomalies include color-related defects).

**What changed:**
- `configs/cable.yaml` and `configs/cable_screen.yaml`: set `vertical_flip: false`, `color_jitter: true`
- No code changes — config-only fix

**Re-distillation required:** Yes (augmentations affect distillation training data)
**Retraining required:** Yes
**Screening protocol:** 100-epoch distillation + 100-epoch training × 3 seeds

### Results

| Seed | I-AUROC (base) | I-AUROC (fix) | P-AUROC (base) | P-AUROC (fix) | PRO (base) | PRO (fix) |
|------|----------------|---------------|----------------|---------------|------------|-----------|
| 7    | 89.7           | **94.5**      | 95.7           | **97.2**      | 78.9       | **87.2**  |
| 42   | 92.4           | **93.3**      | 96.6           | **97.6**      | 79.4       | **87.1**  |
| 123  | 93.0           | **94.9**      | 95.9           | **97.3**      | 79.2       | **85.8**  |
| **Mean** | **91.7**   | **94.2**      | **96.1**       | **97.4**      | **79.2**   | **86.7**  |
| **Delta** |          | **+2.5**      |                | **+1.3**      |            | **+7.5**  |

### Analysis

- **PRO** improved by +7.5pp — the single largest gain of any experiment so far. Removing V-Flip likely had the larger effect: flipping cables upside-down created implausible training samples that confused the teacher's spatial representations and degraded boundary precision.
- **I-AUROC** improved by +2.5pp. Seed 7 gained the most (+4.8pp), suggesting V-Flip was particularly harmful for that initialization.
- **P-AUROC** improved by +1.3pp, now within 0.9pp of paper — essentially closed.
- Seed variance narrowed significantly, indicating more stable training with correct augmentations.

### Remaining gap to paper

| Metric  | E5.5 Mean | Paper  | Gap      | Previous gap |
|---------|-----------|--------|----------|--------------|
| I-AUROC | 94.2%     | 98.9%  | −4.7 pp  | −7.2 pp      |
| P-AUROC | 97.4%     | 98.3%  | −0.9 pp  | −2.2 pp      |
| PRO     | 86.7%     | 90.6%  | −3.9 pp  | −11.4 pp     |

### Conclusion

Config error confirmed and fixed. The augmentation fix closed 65% of the PRO gap, 35% of the I-AUROC gap, and 59% of the P-AUROC gap. This is now the new screening baseline for subsequent experiments.

**Checkpoints:**
- `runs/mvtec_ad/cable/e5.5_aug_fix/seed7/`
- `runs/mvtec_ad/cable/e5.5_aug_fix/seed42/`
- `runs/mvtec_ad/cable/e5.5_aug_fix/seed123/`

---

## Experiment 5.6: Stem MaxPool vs AvgPool (A-002)

**Date:** 2026-02-18
**Hypothesis:** MaxPool2d preserves edge and high-frequency information by selecting the strongest activation in each 2×2 block. AvgPool smooths. The remaining PRO gap (−3.9pp) combined with near-closed P-AUROC (−0.9pp) is consistent with a spatial precision problem — MaxPool could sharpen feature representations and improve boundary detection.

**What changed:**
- `src/mudenet/models/common.py`: `Stem.pool` changed from `nn.AvgPool2d(2, 2)` to `nn.MaxPool2d(2, 2)`

**Re-distillation required:** Yes (teacher architecture changes)
**Retraining required:** Yes
**Screening protocol:** 100-epoch distillation + 100-epoch training × 3 seeds

### Results — REVERTED (neutral)

| Seed | I-AUROC (E5.5) | I-AUROC (MaxPool) | P-AUROC (E5.5) | P-AUROC (MaxPool) | PRO (E5.5) | PRO (MaxPool) |
|------|----------------|-------------------|----------------|-------------------|------------|---------------|
| 42   | 93.3           | **94.7** (+1.4)   | 97.6           | **98.0** (+0.4)   | 87.1       | **87.8** (+0.7) |
| 123  | 94.9           | **95.4** (+0.5)   | 97.3           | **97.5** (+0.2)   | 85.8       | 85.7 (−0.1)    |
| 7    | 94.5           | 92.8 (−1.7)      | 97.2           | 96.9 (−0.3)      | 87.2       | 84.0 (−3.2)    |
| **Mean** | **94.2**   | 94.3 (+0.1)      | **97.4**       | 97.5 (+0.1)      | **86.7**   | 85.8 (−0.9)    |

### Analysis

- Seeds 42 and 123 improved modestly on I-AUROC and P-AUROC, but seed 7 regressed significantly across all metrics (I-AUROC −1.7pp, PRO −3.2pp).
- Seed variance increased: I-AUROC span 2.6pp (was 1.6pp), PRO span 3.8pp (was 1.4pp). MaxPool makes training less stable.
- Mean PRO dropped by 0.9pp — the opposite of the intended effect.

### Conclusion

**Neutral result.** MaxPool does not meet the >1pp PRO improvement threshold and increases seed variance. Code reverted to AvgPool. A-002 resolved — AvgPool is the better choice.

### Code changes

None retained — reverted via `git restore`.

---

## Experiment 5.7: Distillation Target — Nearest-Neighbor Upsampling

**Date:** 2026-02-18
**Hypothesis:** The feature extractor outputs at 64×64 and is upsampled to 128×128 via bilinear interpolation for distillation. The paper describes feature fusion using the Kronecker product with an all-ones matrix — equivalent to nearest-neighbor upsampling. Bilinear fabricates smooth inter-pixel gradients that don't exist in the source 64×64 features; nearest-neighbor produces honest block-structured targets.

**What changed:**
- `src/mudenet/training/distillation.py` `_upsample_target()`: changed `mode="bilinear"` to `mode="nearest"` (removed `align_corners`)

**Re-distillation required:** Yes (distillation target interpolation changes)
**Retraining required:** Yes (different distilled teacher)
**Screening protocol:** 100-epoch distillation + 100-epoch training × 3 seeds, using E5.5 augmentations + E5.4 cosine LR

### Results — REVERTED (regression)

| Seed | I-AUROC (E5.4) | I-AUROC (NN) | P-AUROC (E5.4) | P-AUROC (NN) | PRO (E5.4) | PRO (NN) |
|------|----------------|--------------|----------------|---------------|------------|----------|
| 42   | 94.6           | **95.6** (+1.0) | 97.5        | 97.3 (−0.2)   | 86.1       | 86.5 (+0.4) |
| 123  | 95.0           | 94.5 (−0.5) | 97.3           | 97.5 (+0.2)   | 86.7       | 86.3 (−0.4) |
| 7    | 95.1           | 92.8 (−2.3) | 97.1           | 97.0 (−0.1)   | 87.3       | 84.2 (−3.1) |
| **Mean** | **94.9**   | 94.3 (−0.6) | **97.3**       | 97.3 (0.0)    | **86.7**   | 85.7 (−1.0) |

### Analysis

- **PRO** regressed by −1.0pp mean, with seed 7 dropping −3.1pp. The block-structured targets apparently make the distillation task harder — the teacher struggles to learn sharp 2×2 block discontinuities.
- **I-AUROC** regressed by −0.6pp mean. Seed 42 improved (+1.0pp) but seed 7 collapsed (−2.3pp), similar to the E5.6 pattern of seed 7 being particularly sensitive.
- **P-AUROC** flat (0.0pp mean change).
- Seed variance increased: I-AUROC span 2.8pp (was 0.5pp), PRO span 2.3pp (was 1.2pp). NN upsampling makes training less stable.

### Conclusion

**Negative result.** Nearest-neighbor upsampling produces worse results than bilinear. The bilinear smoothing provides better learning targets for the teacher — smooth inter-pixel gradients act as a form of regularization that helps the teacher generalize. Code reverted to bilinear.

### Code changes

None retained — reverted to bilinear in `_upsample_target()`.

**Checkpoints (for reference only — not used going forward):**
- `runs/mvtec_ad/cable/e5.7_nn_upsample/seed42/`
- `runs/mvtec_ad/cable/e5.7_nn_upsample/seed123/`
- `runs/mvtec_ad/cable/e5.7_nn_upsample/seed7/`

---

## Optimization Roadmap (revised 2026-02-18, post E5.7)

Full analysis and per-experiment specs: `docs/artifacts/exp5-findings.md`

### Completed experiments

| # | Change | Assumption ID | Result |
|---|--------|---------------|--------|
| 1 | Sigma tuning | A-016 | Done — sigma=4.0 kept |
| 2 | Detach + remove final ReLU | A-014, A-012 | Done — +4.5pp I-AUROC |
| 3 | Global z-score normalization | A-011 | Rejected — all metrics regressed |
| 4 | Distillation at 64×64 | A-009 | Rejected — PRO −4.6pp |
| E5.1 | PRO num_thresholds 300→1000 | — | **Kept** — measurement correction (screening PRO 83.7→79.2) |
| E5.2 | Bicubic anomaly map upsampling | — | Reverted — no effect (<0.1pp) |
| E5.3 | Smoothing at embedding resolution | — | Reverted — no effect (<0.1pp) |
| E5.6 | MaxPool vs AvgPool in stem | A-002 | Reverted — neutral (PRO −0.9pp, increased seed variance) |
| E5.7 | Distillation target NN upsample | — | Reverted — regression (I-AUROC −0.6pp, PRO −1.0pp) |
| E5.4 | Cosine LR schedule (end-to-end) | — | Marginal — I-AUROC +0.7pp, PRO flat; kept (low complexity) |

### Phase A — Screening (100 epochs × 3 seeds)

#### Tier 1: Free tests — COMPLETE

All three tested. Only E5.1 retained. See `docs/artifacts/exp5-findings.md` for full results.

#### Tier 2: Retrain only (use existing distilled teacher)

| ID | Change | Status |
|----|--------|--------|
| E5.4 | Cosine LR schedule (end-to-end only) | **Done — marginal (+0.7pp I-AUROC, PRO flat)** |

#### Tier 3: Full pipeline (re-distillation + retraining)

| ID | Change | Assumption ID | Status |
|----|--------|---------------|--------|
| E5.5 | Fix cable augmentation (remove V-Flip, add Color Jitter) | A-017 | **Done — +2.5pp I-AUROC, +7.5pp PRO** |
| E5.6 | MaxPool vs AvgPool in stem | A-002 | **Done — reverted (neutral, PRO −0.9pp)** |
| E5.7 | Distillation target nearest-neighbor upsample | — | **Done — reverted (regression, PRO −1.0pp)** |
| E5.8 | Add BN to stem | A-005 | Pending — next experiment |

#### Execution order

1. ~~**E5.1–E5.3 together**~~ — done; E5.1 kept, E5.2/E5.3 no effect
2. ~~**E5.5**~~ — done; +2.5pp I-AUROC, +7.5pp PRO (new screening baseline)
3. ~~**E5.6**~~ — done; neutral (PRO −0.9pp), reverted
4. ~~**E5.4**~~ — done; marginal (+0.7pp I-AUROC, PRO flat), kept
5. ~~**E5.7**~~ — done; regression (I-AUROC −0.6pp, PRO −1.0pp), reverted
6. **E5.8** — stem BatchNorm ← NEXT

#### Deprioritized (not worth testing)

| Change | Why |
|--------|-----|
| BN/ReLU placement in residual blocks (A-003) | <0.5pp expected; not worth re-distillation cost |
| BN on 1x1 projections (A-004) | Risky — scoring relies on embedding magnitudes |
| Separate optimizers for S1/A/S2 | No evidence of benefit |
| Weight decay, gradient clipping, LR warmup | No evidence of training instability |

### Phase B — Full run (500 epochs × 3 seeds)

Best configuration from Phase A → full training → paper-comparable metrics.
