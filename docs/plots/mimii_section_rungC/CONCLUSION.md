# Rung C — Mamba student trained to classify sections

**Question.** Rungs A/B showed the section-classification *objective* is the dominant lever
(A: frozen ResNet + head → 80.0; B: fine-tune the ResNet encoder → 85.9). Both left MambaAD's
Mamba decoder out. Rung C puts it back: the actual MambaAD student — **frozen ResNet34 teacher
→ MFF/OCE fusion → MambaUPNet decoder** — trained to **classify** the 23 sections (ArcFace on
GAP-pooled student features) instead of to **reconstruct** teacher features (the native
CosLoss). Same input/feature/readouts/per-epoch eval as A/B. It is the first rung that exercises
the Mamba architecture, and answers: *does the Mamba decoder carry the discriminative objective
as well as simply fine-tuning the encoder?*

Trainer: `diagnostics/section_mamba_rungC.py`. Run: 50 epochs, ArcFace sub=2, lr 1e-3 (student
+ head are from scratch; teacher frozen), test AUROC every epoch, **0 divergence-guard skips**.
Figures: `auroc_vs_epoch/`, `per_class_best/`.

## Headline (mean image-level AUROC %, best epoch)

| readout | best epoch | mean |
|---|---|---|
| **`maha_embed`** | 9 | **84.4** |
| `neg_cos` | 43 | 81.9 |
| `logit_nll` | 38 | 75.0 |

Rung C best **84.4** vs Rung A 80.0 (**+4.4**), Rung B 85.9 (**−1.5**), STgram-MFN 90.75 (**−6.3**).

## Findings

1. **The Mamba decoder is a real but secondary lever — it beats a frozen head, and trails
   fine-tuning the encoder.** On the *same frozen teacher features* as Rung A, adding the
   trainable Mamba student lifts the best readout from 80.0 → 84.4 (**+4.4**). But fine-tuning
   the ResNet encoder (Rung B, 85.9) still wins by 1.5. **Representation adaptation of the
   encoder matters more than adding the Mamba decoder architecture.**

2. **The winning readout *flips* — the Mamba feature space prefers a distance readout.** Across
   the ladder the best readout changes: Rung A `neg_cos` ≫ `maha_embed` (frozen embedding is
   useless for distance); Rung B `neg_cos` ≥ `maha_embed` (discriminative wins); **Rung C
   `maha_embed` (84.4) > `neg_cos` (81.9)**. Even trained discriminatively, the Mamba decoder —
   architecturally a *reconstructive* module — produces a feature manifold that a plain
   Mahalanobis distance reads better than the model's own classification head. `neg_cos` on the
   Mamba student is noisy and weak (peaks late at ep43, wobbling 73–79).

3. **`maha_embed` peaks very early (ep9) — peaks-early, hard.** Section train-accuracy climbs
   past 97 % by epoch 5 while the useful AUROC peaks at ep9 then plateaus/declines. The
   per-epoch eval is again essential; a final-epoch reading (75.9 `neg_cos` / ~82 `maha_embed`)
   understates the model.

4. **Rung C and Rung B are complementary per class — and Rung C wins the residual class.** At
   its peak, Rung C (`maha_embed`) beats Rung B on **slider (93.1 vs 85.1, +8)** — the exact
   class that was Rung B's biggest gap to STgram — but loses on valve (−6.5), ToyCar (−4.4) and
   ToyConveyor (−4.6). The Mamba decoder's reconstructive geometry evidently suits slider's
   structure. This directly reinforces the downstream-fusion finding: a per-class / fused
   readout across the rungs would exceed any single one.

## Ladder so far (mean AUROC, each rung's best readout)

```
frozen ResNet + Maha         71.8
Rung A frozen + cls          80.0   (+8.2)   objective as a readout
Rung C Mamba student + cls   84.4   (+4.4 over A)   decoder architecture, frozen teacher
Rung B fine-tune ResNet      85.9   (+5.9 over A)   encoder representation ← best so far
STgram-MFN                   90.75            supervised target
```

**Reading:** encoder representation (B) > Mamba decoder (C) > frozen head (A). The Mamba
architecture helps, but it is not where the remaining gap lives — the encoder is the bigger
lever, and C on a *frozen* teacher can't access that.

## Caveats
- Single seed; from-scratch student, so run-to-run peak epoch will wander (the ~84 level and
  the `maha_embed` > `neg_cos` ordering are the claims, not "epoch 9").
- Rung C used a **frozen** teacher, as in real MambaAD. It therefore does not combine B's
  encoder adaptation with C's Mamba decoder — that combination is the natural next test.

## Next step
The obvious experiment the ladder now points to: **fine-tune the teacher *and* the Mamba
student jointly under the section loss** (B + C combined) — does encoder adaptation stack on
top of the Mamba decoder? And/or **Rung D** (keep the reconstruction distillation *and* add the
classification head, fused score), the faithful "MambaAD + labels" model. Given the per-class
complementarity (C wins slider, B wins the rest), a **fused readout** is also low-hanging fruit.
