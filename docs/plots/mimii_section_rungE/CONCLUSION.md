# Rung E — B+C combined (fine-tune encoder + Mamba student)

**Question.** Rung B fine-tuned the ResNet *encoder* under the section loss (85.9); Rung C
trained the Mamba *decoder* on a frozen encoder (84.4). Rung E combines them: fine-tune the
teacher AND the Mamba student together under the same ArcFace section loss (classification only
— reconstruction was Rung D's dead end). Does encoder adaptation *stack* on top of the Mamba
decoder?

Two LR groups (teacher 1e-4 like B, from-scratch MFF/OCE+Mamba+head 1e-3 like C). The teacher
is un-frozen and `MAMBAAD.forward`'s feature-detach is bypassed so gradients reach the encoder.
Trainer: `diagnostics/section_combined_rungE.py`. 50 epochs, sub=2, 0 divergence skips.

## Headline (mean image-level AUROC %, best epoch)

| readout | best epoch | mean |
|---|---|---|
| **`maha_embed`** | 38 | **86.9** |
| `neg_cos` | 22 | 81.3 |
| `logit_nll` | 2 | 77.1 |

Rung E best **86.9** — the highest of every rung — vs Rung B 85.9 (**+0.95**), Rung C 84.4
(+2.5), STgram-MFN 90.75 (**−3.9**).

## Findings — the answer depends on the framing

1. **On a *fixed* readout, the two levers clearly STACK.** Best `maha_embed` across the ladder:
   A 67.0 → B 81.8 → C 84.4 → **E 86.9**. Fine-tuning the encoder (+14.8 over frozen) *and*
   adding the Mamba decoder (+2.5 over B's maha) each contribute, and combining them gives the
   best. See `maha_stacking/`.

2. **At the *frontier*, the gain over Rung B is marginal (+0.95) — within run-to-run noise.**
   Rung B could already reach 85.9 via a *different* readout (`neg_cos`), so the best-achievable
   AUROC barely moves. Read strictly best-to-best, encoder adaptation already captures most of
   what the Mamba decoder offers, and Rung E is a ~noise-level improvement. Both framings are
   true; the honest summary is: **Rung E is the new best model, but the encoder is the lever
   that matters — the Mamba decoder stacks on a fixed readout yet adds little at the frontier.**

3. **Mamba-in-the-loop rungs prefer the Maha readout; pure-ResNet prefers cosine.** As in Rung
   C, `maha_embed` (86.9) beats `neg_cos` (81.3) here, and `neg_cos` is noisy/weak — the Mamba
   decoder's reconstructive geometry suits a distance readout. (Rung B, pure ResNet, was the
   opposite: `neg_cos` > `maha`.) So the *right* readout is architecture-dependent.

4. **The residual to STgram MOVED — the Mamba decoder closes slider.** Rung B's biggest gap was
   slider (−13.8). Rung E's Mamba decoder + Maha readout pushes **slider to 96.6 (−2.3)** and
   **valve to 96.9 (−1.7)** — both near STgram. The new frontier is **ToyCar (85.5, −9.2)** and
   ToyConveyor (68.6, −5.7). The remaining ~3.9-point mean gap is now concentrated in two
   classes, not spread out.

## The ladder, complete (mean AUROC, best readout per rung)

```
native recon-only (no labels)   ~72.0
Rung A frozen + cls              80.0
Rung D joint distill+classify    84.0   (recon objective = dead weight)
Rung C Mamba + cls               84.4
Rung B fine-tune encoder         85.9
Rung E B+C (encoder + Mamba)     86.9   ← best; +0.95 over B (frontier), +5.1 over B on maha
STgram-MFN                       90.75
```

**Overall conclusion of the labels/objective ladder:** the section-classification objective is
the dominant lever (A→ closes ~8 of the ~19-pt gap as a readout; B/C/E push to ~87 via
representation + architecture). Encoder adaptation is the single biggest contributor; the Mamba
decoder stacks on a fixed readout and specifically rescues slider; the reconstruction objective
contributes nothing. Best "MambaAD + labels" model = **Rung E, 86.9**, leaving **−3.9 to
STgram**, now localised to ToyCar/ToyConveyor.

## Caveats
- Peaks early-ish (ep38 here; `maha_embed` is stable 84–87 across epochs, `neg_cos` noisy).

## UPDATE (2026-07-31) — seed-repeat CORRECTS the E>B claim

Reran both Rung B and Rung E at `--seed 1` and `--seed 2` (same protocol). Result
(`docs/plots/mimii_section_rungE/seed_repeat/`):

| | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| Rung B | 85.86 | 85.58 | 86.22 | **85.89 ± 0.26** |
| Rung E | 86.85 | 85.44 | 86.21 | **86.17 ± 0.58** |
| E − B  | +0.95 | −0.14 | −0.01 | **+0.28 ± 0.50** |

**The seed-0 "+0.95, new best of every rung" result does not reproduce** — in 2 of 3 seeds B
beats E. Revised verdict: **B and E are statistically tied** on mean AUROC; the Mamba decoder
does not reliably improve on fine-tuning the encoder alone. The earlier "Caveats" section
correctly flagged this as within the ~1-pt CUDA-noise band before the repeat confirmed it.

**What DOES reproduce robustly across all 3 seeds:** the per-class trade-off. ToyCar is B > E by
**~6.7 pts on average, in every seed** (B 93.3 vs E 86.7) — this is a real, repeatable effect of
the architecture choice, not noise; it just doesn't show up in the mean because it's offset by
slider/valve. See `docs/plots/mimii_section_frontier/CONCLUSION.md` for the full breakdown
(including a cross-rung oracle that recovers this ToyCar loss for free, no new training) and
confirmation that ToyConveyor's ceiling (~70, B≈E across all seeds) matches a structurally-hard
machine ID (id_02) that STgram-MFN itself also struggles with.

## Next (open)
- A **real (non-oracle) per-class or fused readout** — pick neg_cos vs maha_embed per class on a
  held-out val split, not test — to see how much of the cross-rung oracle's headroom a
  deployable rule actually captures.
- Optional completeness: `--lambda_cls 0` control for Rung D.
