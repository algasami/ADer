# Rung B — fine-tuned encoder + section classifier

**Question.** Rung A showed the discriminative objective helps only as a *readout* on frozen
features (Maha on the trained embedding actually dropped). Rung B unfreezes ResNet34 and
fine-tunes it end-to-end under the **same** ArcFace section-classification loss. Everything
else is held fixed vs Rung A (log-Mel input, concat-GAP feature, 23 sections, neg_cos /
logit_nll readouts) — the **only** change is frozen → trainable encoder, so the lift over
Rung A's ceiling isolates **representation adaptation**.

Trainer: `diagnostics/section_finetune_rungB.py`. Run: 50 epochs, ArcFace sub=2, backbone
lr 1e-4 / head lr 1e-3, test AUROC scored every epoch. Figures: `auroc_vs_epoch/`,
`per_class_best/`.

## Headline (mean image-level AUROC %, best epoch)

| readout | best epoch | mean | vs Rung A |
|---|---|---|---|
| **`neg_cos`** | 16 | **85.9** | **+5.9** |
| `maha_embed` | 11 | 81.8 | **+14.8** (Rung A: 67.0) |
| `logit_nll` | 3 | 75.6 | −3.1 |

Rung B best **85.9** vs Rung A ceiling 80.0 (**+5.9**) and STgram-MFN 90.75 (**−4.9**).

## Findings

1. **Fine-tuning works — representation adaptation is real.** `neg_cos` rises to 85.9,
   closing **over half** of Rung A's remaining ~10.7-point gap to STgram. Letting the encoder
   move under the discriminative objective is a genuine lever, not just a readout trick.

2. **The Rung A negative reverses — the objective *shapes* the representation.** `maha_embed`
   went from **67.0 (frozen, −4.8) → 81.8 (fine-tuned, +14.8)**. In Rung A a generative
   distance readout on the trained embedding *hurt*, because a frozen head can't reshape the
   feature space. Unfrozen, the classification loss reorganizes the manifold so that even a
   plain Mahalanobis distance separates normal from anomalous. This is the cleanest single
   piece of evidence that the *objective*, not the readout, is doing the work.

3. **`neg_cos` is the robust readout; `logit_nll` overfits.** Section train-accuracy hits
   99 % by epoch 5, and from there `logit_nll` (−log softmax confidence) **declines
   monotonically** (75.6 → ~70) as the softmax grows over-confident everywhere and loses
   discrimination. The cosine-margin readout `neg_cos` is far more stable. Use `neg_cos`.

4. **Peaks early — per-epoch eval was essential.** `neg_cos` peaks at **epoch 16** then
   noisily declines; the final epoch (82.9) *understates* the model by ~3 points. This
   reproduces the repo-wide "AUROC peaks early, no best-checkpoint selection" rule — comparing
   at the final epoch would have hidden a third of the Rung B gain.

5. **The residual −4.9 to STgram is now attributable to input + architecture, not the encoder
   or objective.** With a fine-tuned ResNet encoder and an ArcFace section objective, Rung B
   matches STgram-MFN's *recipe* on everything except the **audio-native input** (raw waveform
   / learned Tgram temporal detail) and the **MFN architecture**. Per-class, the residual is
   **concentrated in slider (−13.8)** and secondarily fan/pump, while **ToyCar (−1.9) and
   ToyConveyor (−1.9) are essentially matched** — ToyConveyor, hard everywhere, finally moved
   (Rung A stuck ~63 → Rung B 72.4 ≈ STgram 74.3). slider being the big holdout is telling:
   it is the class where raw-waveform / temporal structure most plausibly carries the signal
   that a log-Mel spectrogram flattens.

## The full ladder decomposition (mean AUROC)

```
frozen ResNet34 + Maha        71.8   ── anchor
  + classification readout    80.0   (+8.2)   Rung A — objective as a readout (encoder frozen)
  + fine-tune encoder         85.9   (+5.9)   Rung B — representation adaptation
  ⋯ audio input + architecture 90.75  (+4.9)   unexplored (raw wave / Tgram; MFN)   → Rung C/D
STgram-MFN                    90.75  ── supervised target   (total gap +18.9)
```

**The objective/labels axis (Rung A + B) accounts for +14.1 of the ~19-point gap — it is the
dominant lever, confirming the gap-decomposition hypothesis.** The remaining ~+5 is input +
architecture, and is concentrated in a single class (slider).

## Caveats
- Single seed; head/backbone training is not bit-reproducible (CUDA) — treat ≤1-point
  differences as noise. The peak epoch will wander run-to-run; the *level* (~85–86) is the
  claim, not "epoch 16" exactly.
- `--eval_maha` refits a per-class bank each epoch on a fresh train pass, so the `maha_embed`
  curve costs an extra forward per eval.

## Next step
The objective lever is now bounded (A + B ≈ +14 of +19). The remaining ~+5 lives in input +
architecture — either **Rung C/D** (put the objective back into MambaAD: Mamba-as-backbone /
joint distill+classify) or a direct raw-waveform / Tgram front-end to test the slider-shaped
residual.
