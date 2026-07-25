# Rung A — frozen encoder + section-classifier head

**Question.** The gap decomposition (`docs/ABLATION_SUMMARY.md`) blames the residual
gap to STgram-MFN on the *learning objective*, not the encoder or input. Rung A is the
first, cleanest test: hold the encoder **frozen**, change **only** the objective — train a
small ArcFace head to classify the 23 free machine-section labels — and score the STgram
way (poor fit to a clip's own section = anomalous). Because the ImageNet ResNet34 features
never move, any lift is attributable to the *objective + readout* alone, with **zero
representation change**.

Probe: `diagnostics/section_classifier_probe.py`. Sweep: 100 epochs × ArcFace sub-clusters
{1, 2, 4}, log-Mel input. Figures: `per_class_grouped/`, `subcluster_sweep/`.

## Headline numbers (mean image-level AUROC, %, sub=2)

| readout | what it is | mean |
|---|---|---|
| `maha_concat_raw` | frozen ResNet34 + Mahalanobis (generative anchor) | **71.8** |
| `maha_embed` | Maha on the classifier-trained embedding | 67.0 (**−4.8**) |
| `logit_nll` | −log p(assigned section) | 78.7 |
| **`neg_cos`** | negative cosine to assigned section center | **80.0** |
| STgram-MFN | supervised audio SOTA (target) | 90.75 |

## Findings

1. **The objective is a real, measurable lever — but only as a *readout*.** Swapping the
   score to STgram-style fit-to-own-section lifts the mean **+8.2** (71.8 → 80.0) on the
   *identical frozen features*. That recovers roughly **40 %** of the ~19-point
   frozen→STgram gap without touching a single encoder weight.

2. **The gain is entirely in the readout, not a better representation.** Scoring the
   classifier's *learned embedding* with a distance metric (`maha_embed`) *drops* to 67.0
   (−4.8 vs the raw-feature anchor). A frozen head cannot reshape the feature manifold; it
   helps only by asking "does this clip fit its own section?", not by producing more
   separable features. This is the key negative result and the direct motivation for Rung B.

3. **ArcFace sub-clusters are a non-lever.** {1, 2, 4} → 80.2 / 80.0 / 80.2 (a 0.2-point
   spread — pure noise). The head saturates by ~epoch 60 (~71 % section accuracy), so Rung A
   has *converged* — no cheap "train longer / add clusters" knob remains.

4. **Per-class structure is complementary, not uniform.** The classification readout
   **rescues the classes Maha was weak on** (valve +21, fan +18, pump +13, ToyCar +8) but
   **regresses slider −9** (Maha's single best class) and leaves **ToyConveyor flat ~63**
   under either readout. The two readouts capture *different* failure modes — a hint that a
   fused score could beat either alone, and that ToyConveyor is hard for reasons neither
   addresses.

5. **~+11 AUROC to STgram remains, and it is not reachable from a frozen ImageNet
   encoder.** The best head-only result (80.0) still trails STgram-MFN (90.75) by ~11
   points. Since the readout is exhausted and sub-clusters don't help, the remaining gap
   must live in the **representation** (encoder adaptation → **Rung B**), and possibly in the
   **audio-native input** (learned Tgram / raw waveform) and the **MFN inductive bias** —
   none of which a frozen ImageNet backbone can supply.

## Decomposition of the frozen→STgram gap (mean AUROC)

```
frozen ResNet34 + Maha        71.8   ── anchor
  + classification readout    80.0   (+8.2)  ← Rung A: objective as a readout, encoder frozen
  ⋯ representation adaptation   ?     (Rung B: unfreeze encoder under the same ArcFace loss)
  ⋯ audio input + architecture  ?     (Tgram / raw wave; MFN)
STgram-MFN                    90.75  ── supervised target  (total gap +18.9)
```

## Caveats
- **Single-seed, and head training is not bit-reproducible** (CUDA non-determinism): across
  re-runs the head-dependent readouts drift ≲1 point, while the head-free `maha_concat_raw`
  anchor is exactly reproducible (71.80). Treat ≤1-point differences as noise — this is why
  the sub-cluster sweep is read as flat.
- The `maha_concat_raw` anchor (71.8, concat-GAP) is recomputed in-run for a byte-comparable
  baseline; it may differ slightly from the best logged frozen-Maha number.
- ToyConveyor is a known hard class across every ablation front, not specific to Rung A.

## Next step
**Rung B** — unfreeze ResNet34 and fine-tune it end-to-end under the same ArcFace section
loss. Rung A's converged ceiling (80.0) is the baseline Rung B must beat; the delta measures
how much of the remaining ~+11 is *representation adaptation* vs input/architecture.
