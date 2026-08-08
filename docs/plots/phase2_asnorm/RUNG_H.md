# Rung H — does the Mamba decoder add anything on a fair substrate?

Branch `aug-ast-phases`. The project goal is to match or beat STgram-MFN **with a
MambaAD-adjacent architecture**, but the ladder had walked away from Mamba one defensible step
at a time, and by Phase 2 the best model contained none. Every prior Mamba verdict (rungs
C/D/E/F) was measured on the PNG pipeline that Phase 1 showed was handicapped — and that
pipeline had already produced one wrong conclusion ("input is not a lever"). So the verdict was
re-tested on the fbank substrate.

## Setup — a true A/B

Both arms identical except the decoder: fp32, `scan_type=sweep`, `lr_head=3e-4`, `lr_enc=5e-5`,
mixup 0.2, fbank 512×128 (time-pooled by 2), 30 epochs, 3 seeds, per-section Maha readout,
held-out 2-fold epoch selection.

| arm | model |
|---|---|
| baseline | ResNet34 → GAP → ArcFace |
| Rung H | ResNet34 teacher → MFF/OCE → **Mamba student** → GAP → ArcFace |

The existing `base512` runs were deliberately NOT reused as the control — they used bf16 and
`lr_head=1e-3`, which would have confounded the decoder with two other changes.

## Result: the Mamba decoder costs 2.2 AUROC

| arm | held-out mean-of-ID | test-selected | final epoch | vs STgram 90.75 |
|---|---|---|---|---|
| baseline (no decoder) | **91.01 ± 0.40** | 91.16 ± 0.26 | 90.68 ± 0.29 | **+0.26** |
| Rung H (Mamba) | **88.79 ± 0.72** | 88.88 ± 0.59 | 88.04 ± 0.78 | −1.96 |
| **decoder effect** | **−2.22** | −2.28 | −2.64 | |

The gap is ~3× the seed spread and holds under every selection rule. Rung H trained cleanly —
97–98% train accuracy, **zero** non-finite steps — so this is a valid run that is simply worse,
not a broken one.

**This is a much stronger negative than the PNG-era result.** Rungs C/E put the decoder at a tie
(E − B = +0.28 ± 0.50) against a baseline of ~86. Here it is a clear loss against a baseline
that beats STgram-MFN. The decoder is not neutral on this task; it actively destroys information
that the plain encoder's GAP preserves.

## It also cost three fixes to train at all

All three fail **silently** (no exception, no obvious error) — see the `mamba-fbank-gotchas`
note for detail:

1. **Autocast NaNs the selective-scan BACKWARD** while the forward stays finite. Post-mortem at
   the first bad step: loss, embeddings, inputs, all parameters and all optimizer state finite —
   only `gnorm=nan`. Symptom: epoch 1 clean, then ~1250/1258 steps skipped forever. Fix: fp32.
2. **`SCANS` assumes a square feature map** (`model/mambaad.py:53`, permutation over
   `size ** dim`). On a 128×32 map the flat length is 4096 = 64², so **no exception is raised**
   while the Hilbert curve — computed for a 64×64 layout — silently scrambles space. Only
   `sweep` (raster) is shape-agnostic. The repo's scan-curve ablation found all five curves
   equivalent, but that was measured on *square* inputs and does not license using them here.
3. **`lr_head=1e-3` is too high** for the Mamba student on fbank (it was fine for C/E on PNG).
   Not divergence — train accuracy *fell* (34 → 24 → 10) while loss fell. 3e-4 fixes it.

That fragility is itself a finding: the plain encoder trained first time at both learning rates
and in bf16.

## Verdict

On a fair, well-tuned substrate the Mamba decoder is **not** a lever for this task — it is a
2.2-point liability, on top of being markedly harder to train. The MambaAD framing does not
survive contact with MIMII once the input pipeline and readout are fixed.

The deployable result stands without it: **91.15 ± 0.18** (ResNet34, fbank 1024×128,
per-section Maha, honest held-out selection) versus STgram-MFN's 90.75.
