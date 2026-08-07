# Phase 0 — does spectrogram augmentation move Rung B?

Branch `aug-ast-phases`. Three arms, 3 seeds each, 50 epochs, otherwise identical to the
Rung B protocol (log-Mel, ResNet34, ArcFace over 23 machine sections, `--sub 2`):

| arm | augmentation | best-of-readout | winning readout |
|---|---|---|---|
| `no-aug` (control) | none | **85.89 ± 0.26** | `neg_cos` |
| `aug` | crop + mask + mixup 0.2 | 84.66 ± 0.57 | `maha_embed` |
| `aug-nomixup` | crop + mask | 83.21 ± 0.27 | mixed |

## The premise: the ladder never used augmentation

`MambaAD/configs/mambaad/mimii/_base.py:100` assigns `test_transforms =
train_transforms` — the *same list object* — so every rung A–F trained on deterministic
Resize→CenterCrop→ToTensor→Normalize views. Control train accuracy is 94.7 % at epoch 2 and
99.6 % at epoch 50 on a 23-way task. Phase 0 measures augmentation as its own lever before
the Phase 1 encoder swap, so a later AST delta cannot be confounded by a missing regularizer.

## Q1 — headline: negative

Paired per-seed deltas vs the control (the B/E seed-repeat puts run-to-run noise at
~0.3–0.6 pt, so judge against that):

| readout | `aug` | `aug-nomixup` |
|---|---|---|
| `neg_cos` | **−4.15** ± 1.61 | **−2.70** ± 0.28 |
| `logit_nll` | **+2.78** ± 0.93 | +0.11 ± 0.68 |
| `maha_embed` | **+2.83** ± 0.58 | +0.32 ± 0.64 |

Every effect is consistent in sign across all three seeds. **No augmentation configuration
beats the control on best-of-readout**, which is how the ladder is scored: 85.89 → 84.66 →
83.21. What augmentation does instead is *flip the readout ordering* — un-augmented,
`neg_cos` (85.89) beats Maha (81.83) by 4 pts; augmented, Maha (84.66) beats `neg_cos`
(81.73) by 3.

## Component decomposition — mixup is the useful half, masking is the harmful half

The `--mixup 0` arm was run to test the hypothesis that **mixup** caused the `neg_cos`
collapse (by training the embedding on inputs whose section assignment is ambiguous, which
is exactly the geometry `neg_cos` reads). **That hypothesis is refuted.** Removing mixup
neither restores `neg_cos` nor preserves the Maha gain:

```
crop + masking alone :  maha +0.32   neg_cos -2.70
adding mixup on top  :  maha +2.51   neg_cos -1.45
```

So the causal split is the reverse of the prediction. **Mixup supplies essentially the whole
Maha gain** and about a third of the `neg_cos` loss; **crop + SpecAugment masking supplies
most of the `neg_cos` damage and no benefit at all** — strictly worse than not augmenting.
If augmentation is used at all downstream, it should be mixup-first, and the masking/crop
component should be re-tuned or dropped rather than carried over by default.

## Q2/Q3 — a ladder claim partly corrected, and it is specifically mixup

| readout | peak epoch: no-aug → nomixup → aug | decay |
|---|---|---|
| `logit_nll` | 1.7 → 2.7 → **19.0** | 4.66 → 2.80 → 3.82 |
| `maha_embed` | 27.3 → 13.3 → **44.0** | 1.66 → 2.08 → 0.95 |
| `neg_cos` | 11.0 → 11.0 → 17.7 | 4.03 → 3.93 → 5.20 |

Rung B's recorded finding — "`logit_nll` overfits, train-acc 99 % by ep5, softmax
overconfident, monotonic decline" — is substantially a **missing-regularizer artifact**:
under mixup that readout's peak moves from epoch 1.7 to 19.0 and it gains +2.78. But the
`aug-nomixup` column shows this is *mixup specifically*, not augmentation in general (peak
stays at 2.7 without it). "AUROC peaks early" is therefore partly regularization deficit,
not purely a scoring artifact.

Train accuracy at epoch 50 still reaches 97–99 % under `standard`, so the regularization is
mild and "augment harder" remains untested. This is **not** a truncation effect: the `aug`
arm's last-10-epoch slope is +0.003 pts/epoch (plateaued) vs +0.147 for the control.

## The actually useful result: augmentation buys model diversity, not mean AUROC

Per-class, the arms are strongly complementary — `aug`+Maha owns slider (96.85 vs 86.95) and
valve, while the control owns ToyCar (93.35 vs 84.15) and ToyConveyor (70.12 vs 64.08):

| | fan | pump | slider | valve | ToyCar | ToyConveyor | mean |
|---|---|---|---|---|---|---|---|
| no-aug `neg_cos` | 82.29 | 87.35 | 86.95 | 95.26 | **93.35** | **70.12** | 85.89 |
| aug `maha_embed` | 80.21 | 87.13 | **96.85** | **95.54** | 84.15 | 64.08 | 84.66 |
| nomixup `neg_cos` | 80.96 | **91.36** | 90.26 | 90.88 | 87.43 | 58.22 | 83.19 |

Per-class best over the two full arms = **87.67**; over all three = **88.25**. For reference
the A–E cross-rung oracle over *five* rungs was 88.80. **Two ResNet34 runs differing only in
augmentation reach nearly the same oracle as the whole ladder did** — augmentation is an
unusually cheap diversity source for the Phase 2 fusion, even though it is a mean-AUROC
negative on its own.

Caveat, same as the A–E oracle: this peeks at the test set. Per-clip scores are not
persisted, only per-epoch AUROC, so the fair held-out-selection version (Rung F's mechanism)
cannot be computed from these runs and would need a re-run that dumps scores.

## Carry-over into Phase 1

1. **Do not put `standard` augmentation in the AST recipe by default.** The masking/crop half
   is a net negative here; only the mixup half earns its place.
2. **`neg_cos` is not a safe default readout under augmentation** — besides losing 4 pts it
   becomes unstable (seed std 2.01 vs 0.32 for the control). Report Maha alongside it.
3. **ToyConveyor is hurt by every augmentation arm** (70.12 → 58–64), consistent with the
   structural id_02 ceiling seen across the ladder and in STgram-MFN itself.
4. The augmented model is a **fusion candidate for Phase 2**, which is where its value is.

Reproduce: `python docs/plot_phase0_aug.py` (figure `phase0_aug.png`, tables `summary.csv`,
`per_class.csv`).
