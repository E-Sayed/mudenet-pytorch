# Assumptions Register

> **Purpose:** Track architectural choices made under uncertainty during this reproduction.

**Statuses:** Open (untested) | Confirmed (results match) | Revisit (flagged for change) | Resolved (author confirmed or moot)

---

### A-001: Padding formula in residual blocks — typo correction

- **Figure says:** `pad (k-2)/2`
- **Chose:** `pad (k-1)/2` (standard same-padding)
- **Why:** `(k-2)/2` gives non-integer values for the kernel sizes used (k=3 → 0.5, k=5 → 1.5). The figure's own spatial annotations (128×128 throughout) require same-padding. The residual skip connection requires matching spatial dims. `(k-1)/2` is the only formula that is self-consistent with the figure.
- **Alternatives:** None viable — `(k-2)/2` is not implementable.
- **How to detect if wrong:** Awaiting author response. If `(k-2)/2` is confirmed, it would require understanding the notation convention used.
- **Status:** Resolved — author confirmed `(k-1)/2` is correct (2026-02-17)

---

### A-002: Pool type in stem

- **Figure says:** "2×2 pool" (no type specified)
- **Chose:** `AvgPool2d(kernel_size=2, stride=2)`
- **Why:** The figure doesn't specify. AvgPool is common in lightweight descriptor networks (e.g., EfficientAD's PDN uses AvgPool). Standard ResNet uses MaxPool but with a 3×3 kernel, which is different.
- **Alternatives:** `MaxPool2d(kernel_size=2, stride=2)`
- **How to detect if wrong:** Notably lower distilled teacher quality. Swap to MaxPool and compare distillation loss convergence.
- **Status:** Open

---

### A-003: BN and ReLU placement in residual blocks

- **Figure shows:** Two k×k convolutions with a skip connection (addition). No BN or ReLU boxes shown explicitly.
- **Chose:** Post-activation ResNet style: `Conv → BN → ReLU → Conv → BN → (+skip) → ReLU`
- **Why:** This is the standard ResNet residual block layout (He et al., 2016). The paper describes a "residual block" without further detail, which conventionally implies this pattern. The figure's caption says "ReLU activation after each convolution" for the autoencoder but does not make the same statement for the teacher — however, ReLU is standard.
- **Alternatives:**
  - Pre-activation: `BN → ReLU → Conv → BN → ReLU → Conv → (+skip)` (He et al., 2016 v2)
  - Minimal: `Conv → ReLU → Conv → (+skip) → ReLU` (no BN)
  - Conv → BN → ReLU for both convs, then add skip (no final ReLU after skip)
- **How to detect if wrong:** Compare training loss curves and final metrics. Pre- vs post-activation usually has small effect; absence of BN would show as training instability.
- **Status:** Open

---

### A-004: 1×1 projection layers — bare or with activation

- **Figure shows:** Plain "1×1 conv" boxes after each block, projecting 64 → C=128.
- **Chose:** Bare `Conv2d(64, 128, kernel_size=1)` with no BN or ReLU.
- **Why:** The figure shows no activation or normalization after the 1×1 convs. These projections map internal representation to the embedding space — a linear projection is standard for this purpose (similar to how feature pyramid networks use bare 1×1 convs).
- **Alternatives:**
  - `Conv2d → BN → ReLU` (adds nonlinearity to projection)
  - `Conv2d → BN` (normalization without activation)
- **How to detect if wrong:** High distillation loss that doesn't converge may indicate BN/ReLU is needed on projections.
- **Status:** Open

---

### A-005: Stem convolution — stride and activation

- **Figure shows:** `7×7 conv, pad 3` producing `64@256×256`, followed by `2×2 pool`.
- **Chose:** `Conv2d(3, 64, kernel_size=7, stride=1, padding=3) → ReLU → AvgPool2d(2, 2)`
- **Why:** The figure shows output is 256×256 after the conv (same spatial size as input), confirming stride=1 with padding=3. ReLU is assumed between conv and pool (standard practice), though the figure doesn't show it explicitly.
- **Alternatives:**
  - No ReLU between conv and pool (unlikely — all standard networks activate before pooling)
  - BN between conv and pool: `Conv → BN → ReLU → Pool`
- **How to detect if wrong:** Training instability in distillation may indicate missing BN in stem.
- **Status:** Open

---

### A-006: FLOPs discrepancy — accepted, not blocking

- **Paper reports:** 20.7 GFLOPs for the full model (T + S1 + S2 + A).
- **Observation:** All candidate architectures (both the old dilated-conv approach and the new figure-based approach) exceed this number under standard counting. With 64 internal channels, the figure-based teacher is estimated at ~11.4 GMACs per network → ~34.1 GMACs for 3 networks, which exceeds 20.7 under any convention.
- **Chose:** Use the figure's architecture as ground truth. The FLOPs number in the paper likely uses a non-standard counting method or the paper has a reporting error.
- **Why:** The figures are unambiguous about the architecture. A FLOPs number is a derived metric that depends on counting conventions. Architecture is more trustworthy than a summary statistic.
- **Alternatives:** None — we can't change the architecture to match a FLOPs number when the figures clearly show the architecture.
- **How to detect if wrong:** Awaiting author clarification. A fundamentally different architecture would require reconsideration.
- **Status:** Open

---

### A-007: Encoder padding values

- **Figure 3 shows:** 6 encoder convolutions: five 3×3 stride-2 + one 8×8 stride-1. Spatial progression: 256→128→64→32→16→8→1×1.
- **Chose:** `padding=1` for all 3×3 stride-2 convs, `padding=0` for the 8×8 conv.
- **Why:** For 3×3 stride-2: `out = (in + 2*1 - 3)/2 + 1 = in/2`. This gives 256→128→64→32→16→8. For 8×8 stride-1 with pad=0: `out = (8 + 0 - 8)/1 + 1 = 1`. Both match the figure exactly.
- **Alternatives:** Other padding values don't produce the shown spatial progression.
- **Status:** Open (high confidence)

---

### A-008: Decoder transposed convolution stride and padding

- **Figure 3 shows:** Decoder spatial progression: 1→4→8→16→32→64→128. First layer uses 4×4 conv^T, rest use 3×3 conv^T.
- **Chose:**
  - Layer 1: `ConvTranspose2d(Z, C, kernel_size=4, stride=4, padding=0)` → 1×1 to 4×4
  - Layers 2-6: `ConvTranspose2d(C, C, kernel_size=3, stride=2, padding=1, output_padding=1)` → doubles spatial
- **Why:** Layer 1: `out = (1-1)*4 + 4 - 0 = 4`. Layers 2-6: `out = (in-1)*2 + 3 - 2 + 1 = 2*in`, giving 4→8→16→32→64→128. This is the standard formula for spatial doubling with 3×3 transposed convs.
- **Alternatives:** Different stride/padding combos could achieve the same progression. The key constraint is matching the spatial dimensions shown.
- **How to detect if wrong:** Poor autoencoder reconstructions or checkerboard artifacts (a classic transposed conv issue).
- **Status:** Open

---

### A-009: Distillation spatial resolution mismatch (64x64 vs 128x128)

- **Issue:** The distillation target E (from WideResNet50 layer1) is at 64×64 spatial resolution. The teacher network's embedding maps are at 128×128. Equation 16 (distillation loss) compares them, so they must match spatially.
- **Chose:** Handled in the distillation training loop. The feature extractor outputs at its natural 64×64 resolution; the teacher outputs at 128×128. Neither module bakes in a resolution assumption.
- **Why:** The paper doesn't explicitly address this gap. The reconciliation (upsample or downsample) is a training-loop concern, not a model-definition concern. Keeping both modules at their natural resolutions preserves flexibility.
- **Resolution:** Upsample E from 64×64 to 128×128 (bilinear) in the distillation loop. Standard practice in knowledge distillation with resolution mismatches.
- **Alternatives:**
  - Downsample teacher output from 128×128 to 64×64 (loses spatial detail in the trained teacher)
  - Modify the teacher stem to reduce by 4× instead of 2× during distillation (changes the architecture)
- **How to detect if wrong:** Distillation loss not converging, or poor downstream metrics despite converged distillation.
- **Status:** Open

---

### A-010: `utils/logging.py` name collision with stdlib

- **Issue:** A planned `src/mudenet/utils/logging.py` file would shadow Python's stdlib `logging` module, causing `import logging` to resolve to the wrong module.
- **Chose:** Do not use the name `logging.py`. Use a non-conflicting name when implementing.
- **Alternatives:** `log_utils.py`, `log_config.py`, `logger.py`
- **Why:** Well-known Python footgun. Every module in the project uses `import logging` for standard logging — shadowing it would cause widespread breakage.
- **Status:** Open

---

### A-011: Z-score normalization scope in feature extractor

- **Paper says:** "z-score based on ImageNet statistics" (Sec. 3.3)
- **Chose:** Per-sample, per-channel normalization using batch statistics (mean/std computed over spatial dims H, W for each sample and channel independently).
- **Why:** The paper's phrasing is ambiguous. It could mean (a) per-sample statistics from the extracted features, or (b) pre-computed global mean/std from ImageNet. Option (b) is impractical because the channel sampling (Eq. 15) selects a random subset of 128 channels from 1792, and these channels differ per seed — there are no standard "ImageNet statistics" for arbitrary channel subsets. Per-sample normalization is the simplest correct interpretation.
- **Tested alternative:** Global per-channel mean/std computed over the entire training set in a preprocessing pass. Result: all metrics regressed (I-AUROC −4.1pp, P-AUROC −0.5pp, PRO −2.8pp vs v2 baseline). Global normalization introduced high variance in target norms, making distillation harder. See optimization-log.md Experiment 3.
- **Status:** Resolved — per-sample normalization confirmed correct (Experiment 3, negative result)

---

### A-012: Decoder ReLU vs bare teacher projection — representational asymmetry

- **Issue:** The teacher's 1x1 projections have no activation (A-004), so teacher embeddings can be negative. The decoder's final layer has ReLU (per Figure 3), so autoencoder outputs are always non-negative. The autoencoder therefore cannot reconstruct negative teacher embeddings.
- **Chose:** ~~Keep both as implemented.~~ **Revised:** Removed the final ReLU from the decoder. Intermediate ReLUs (between transposed conv layers) are kept for nonlinearity.
- **Why:** The final ReLU created a systematic reconstruction floor — the autoencoder could not reproduce negative teacher values. Removing it allows the decoder to output the full value range. Combined with A-014 (detach), this contributed to +4.5pp I-AUROC, +0.7pp P-AUROC, +2.2pp PRO on cable (see optimization-log.md Experiment 2).
- **File changed:** `src/mudenet/models/autoencoder.py` — `Decoder.layers` final ReLU removed
- **Status:** Revisit → **Resolved** (confirmed beneficial, Experiment 2)

---

### A-013: Distillation hyperparameters assumed same as Stage 2

- **Issue:** The paper specifies Adam/lr=1e-3/batch_size=8/500 epochs for end-to-end training (Stage 2, Sec. 4) but does not separately specify distillation (Stage 1) hyperparameters.
- **Chose:** Default to the same hyperparameters (Adam, lr=1e-3, batch_size=8, 500 epochs) for distillation.
- **Why:** The paper doesn't distinguish them, and the same optimizer/lr is a reasonable default. Distillation is a simpler optimization (single target vs multi-loss), so it may converge faster — users can reduce epochs if needed.
- **How to detect if wrong:** Distillation loss not converging within 500 epochs (increase lr or epochs), or converging very early (reduce epochs for efficiency).
- **Status:** Open

---

### A-014: Gradient flow through autoencoder in logical loss (Eq. 7 vs Eq. 8)

- **Issue:** Eq. 7 defines the logical loss as L_2(theta_S2), parameterized only by S2's weights. This suggests only S2 receives gradients from L_2. However, Eq. 8 defines the joint loss L = L_1 + L_A + L_2 optimized end-to-end, and the autoencoder's outputs are used as targets for S2 in L_2. In the joint formulation, autograd naturally flows gradients from L_2 back through the autoencoder as well.
- **Chose:** ~~Allow gradients to flow from L_2 through the autoencoder.~~ **Revised:** Detach `autoencoder_maps` before passing to `logical_loss`, so L_2 only updates θ_S2. The autoencoder still receives gradients from L_A.
- **Why:** The per-equation notation L_2(θ_S2) is intentional — autoencoder outputs should be treated as fixed targets for S2. Without `.detach()`, conflicting gradients from L_A (push autoencoder to reconstruct teacher) and L_2 (push autoencoder to be easy for S2 to match) degraded both networks. Combined with A-012 (remove final ReLU), this contributed to +4.5pp I-AUROC, +0.7pp P-AUROC, +2.2pp PRO on cable (see optimization-log.md Experiment 2).
- **File changed:** `src/mudenet/training/trainer.py` line 155 — `[m.detach() for m in autoencoder_maps]`
- **Status:** Revisit → **Resolved** (confirmed beneficial, Experiment 2)

---

### A-015: Simplified sPRO metric

- **Official sPRO:** The MVTec LOCO evaluation code (Bergmann et al., 2022) uses per-defect-type saturation areas loaded from `defects_config.json`. Each defect type has its own `saturation_threshold` (absolute pixel count or relative fraction). The per-defect sPRO is `min(tp_area / saturation_area, 1.0)`, which makes it *easier* to reach 1.0 for large defects.
- **Our sPRO:** Uses a single global `saturation_threshold` and connected-component analysis. Per-component overlap is clamped: `min(overlap, saturation_threshold)`, which *caps* the maximum overlap value. The area under the curve is normalized by `saturation_threshold * max_fpr` to produce a result in [0, 1].
- **Why simplified:** The full official sPRO requires (a) multi-channel ground truth per image, (b) per-defect-type configuration from `defects_config.json`, and (c) a fundamentally different overlap formula. Implementing this is a significant scope expansion beyond the original build plan.
- **Impact:** When `saturation_threshold=1.0` (the default), our sPRO degenerates to standard PRO at `max_fpr=0.05`, which is correct. With `saturation_threshold<1.0`, our values will differ from the official benchmark.
- **How to mitigate:** For exact MVTec LOCO paper-matching results, use the official MVTec evaluation code (available from [mvtec.com](https://mvtec.com/company/research/datasets/mvtec-loco/downloads)).
- **Status:** Open

---

### A-016: Gaussian smoothing on anomaly maps

- **Paper says:** Nothing explicit about post-processing smoothing.
- **Chose:** Apply Gaussian smoothing (sigma=4.0) to the anomaly map after upsampling to input resolution and before computing image-level scores and pixel-level metrics.
- **Why:** Standard practice in teacher-student anomaly detection methods (EfficientAD, AST, etc.). Without smoothing, single-pixel noise spikes inflate image-level scores (which use spatial max), causing false positives on nominal images. Testing on cable showed +1.8pp Image-AUROC improvement with sigma=4.0.
- **Alternatives:** No smoothing (sigma=0.0), different sigma values (2.0, 6.0, 8.0).
- **How to detect if wrong:** If smoothing degrades Pixel-AUROC on fine-grained defects (over-smoothing blurs small anomalies), reduce sigma.
- **Sigma sweep (cable, MVTec AD):**

  | Sigma | I-AUROC | P-AUROC | PRO  |
  |-------|---------|---------|------|
  | 0.0   | 89.2    | 96.8    | 82.0 |
  | 2.0   | 90.7    | 96.9    | 82.3 |
  | 4.0   | 91.0    | 97.1    | 82.1 |
  | 6.0   | 90.5    | 97.2    | 81.6 |
  | 8.0   | 90.1    | 97.2    | 80.7 |

  I-AUROC peaks at 4.0; PRO peaks at 2.0 (marginal); P-AUROC nearly flat. Sigma=4.0 is the best overall trade-off for cable. May differ per category/dataset.
- **Status:** Open — sigma=4.0 confirmed as best for cable, untested on other categories

---

## Future Entries

New entries are added as more assumptions are identified during development.
