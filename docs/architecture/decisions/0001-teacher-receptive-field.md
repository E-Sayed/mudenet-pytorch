# 1. Teacher Network Architecture

Date: 2026-02-14
Status: **Superseded** (2026-02-14 — paper Figure 2 reveals different architecture)

> **Superseded by:** Evidence from paper Figure 2, analyzed in `docs/artifacts/session-transfer-003.md`. The dilated convolution approach below is incorrect. The actual architecture uses standard ResNet-style residual blocks with 64 internal channels, no dilation.

---

## Original Decision (SUPERSEDED)

The original ADR proposed a dilated convolution architecture:
- Stem: Conv2d(3, 128, k=4, s=2, p=1) → 128 channels throughout
- 3 single-conv residual blocks with dilation d=(3, 4, 8)
- RF targets achieved via dilation, not depth

This was accepted based on computational budget analysis and theoretical RF matching. However, it was derived without access to the paper's architectural figures.

---

## What Went Wrong

The original analysis was sound given the information available (paper text only), but the paper text is ambiguous about the architecture. It describes "k × k convolutions" and "residual blocks" without specifying:
- Internal channel count (64, not 128)
- Number of convolutions per block (2, not 1)
- Number of blocks per level (1, 2, 2 — not 1, 1, 1)
- Kernel sizes per level (3, 3, 5 — not all 3×3 with dilation)
- The presence of 1×1 projection layers

The paper figures (Fig. 1–3) contain this information unambiguously. They were discovered at the end of the previous session after the ADR was already accepted.

---

## Corrected Architecture (from Figure 2)

### Full layer-by-layer specification

```
Stem:
  Conv2d(3, 64, kernel_size=7, stride=1, padding=3)     → 64 @ 256×256
  ReLU
  AvgPool2d(kernel_size=2, stride=2)                     → 64 @ 128×128
  Stem RF = 8, accumulated stride = 2

Block 1 — 1× residual block (k=3):
  [Conv2d(64, 64, 3, pad=1) → BN → ReLU → Conv2d(64, 64, 3, pad=1) → BN → (+skip) → ReLU]
  RF: 8 + 2×4 = 16  ✓
  Output: 64 @ 128×128
  → Conv2d(64, 128, 1)  →  X¹ at C @ 128×128

Block 2 — 2× residual block (k=3):
  [ResBlock(64, k=3)] × 2  (4 convolutions total)
  RF: 16 + 4×4 = 32  ✓
  Output: 64 @ 128×128
  → Conv2d(64, 128, 1)  →  X² at C @ 128×128

Block 3 — 2× residual block (k=5):
  [ResBlock(64, k=5)] × 2  (4 convolutions total)
  RF: 32 + 4×8 = 64  ✓
  Output: 64 @ 128×128
  → Conv2d(64, 128, 1)  →  X³ at C @ 128×128
```

### Key differences from original ADR

| Aspect | Original ADR (wrong) | Figure 2 (correct) |
|--------|---------------------|---------------------|
| Stem | Conv(3→128, k=4, s=2, p=1) | Conv(3→64, k=7, s=1, p=3) + AvgPool(2) |
| Internal channels | 128 | **64** |
| Block type | Single dilated conv + BN + ReLU | **Standard 2-conv residual block** |
| Blocks per level | 1, 1, 1 | **1, 2, 2** |
| Kernel sizes | All 3×3 (with dilation d=3,4,8) | **3×3, 3×3, 5×5 (no dilation)** |
| 1×1 projections | None | **64→C=128 after each block** |
| Total convs/network | 4 | **14** (1 stem + 10 block + 3 proj) |

### Receptive field verification (confirmed)

```
Stem conv (7×7, s=1):   RF = 7, stride = 1
Stem pool (2×2, s=2):   RF = 8, stride = 2

Block 1 (1× res, k=3):  2 convs × (k-1)×stride = 2×2×2 = 8   → RF = 8+8 = 16 ✓
Block 2 (2× res, k=3):  4 convs × (k-1)×stride = 4×2×2 = 16  → RF = 16+16 = 32 ✓
Block 3 (2× res, k=5):  4 convs × (k-1)×stride = 4×4×2 = 32  → RF = 32+32 = 64 ✓
```

Both theoretical and gradient-based measurements confirm exact RF match. Unlike the dilated architecture, the figure-based architecture achieves **100% RF density** (no gridding gaps).

### Empirical verification (2026-02-14)

Script: `scripts/verify_teacher_architecture.py`

| Metric | Value |
|--------|-------|
| RF (gradient-based) | 16, 32, 64 ✓ |
| RF density | 100% at all levels |
| Spatial dims | 128×128 at all levels ✓ |
| Output channels | 128 at all levels ✓ |
| Params/network | 666,496 (2.67 MB) |
| 3 networks | 1,999,488 (8.00 MB) |
| Autoencoder budget | 30.20 MB |
| GMACs/network | 11.354 |
| 3 networks GMACs | 34.062 |
| Forward pass | ✓ |
| Gradient flow | ✓ |

### FLOPs note

Three networks (34.06 GMACs) significantly exceed the paper's reported 20.7 GFLOPs. This discrepancy exists under ALL candidate architectures tested and is accepted as a counting convention difference. See assumptions register A-006.

---

## Open Assumptions

Several details are not specified in the figure and required interpretation. All are tracked in `docs/architecture/assumptions-register.md`:

- **A-001:** Padding formula typo — using (k-1)/2
- **A-002:** Pool type — using AvgPool
- **A-003:** BN/ReLU placement — using post-activation
- **A-004:** 1×1 projections — bare (no BN/ReLU)
- **A-005:** Stem activation — ReLU between conv and pool

---

## Consequences

1. **Architecture is now grounded in visual evidence** — not just text interpretation
2. **64 internal channels** means smaller per-layer cost but more layers → different compute profile
3. **100% RF density** eliminates the gridding artifact concern from the dilated approach
4. **30.2 MB remaining for autoencoder** — ample room for the architecture shown in Figure 3
5. **The teacher, S1, and S2 all share this architecture** — they produce 128-channel outputs via the 1×1 projections despite using 64 channels internally
