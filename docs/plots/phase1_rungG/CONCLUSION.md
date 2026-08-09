# Phase 1 — Rung G: AST encoder, and what actually turned out to be the lever

> **Figures** (regenerate with `python docs/plot_phase1_rungG.py`, from the run CSVs — no
> retraining): `input_lever.png` (+ `input_lever.csv`, `attribution.csv`) for the attribution
> ladder in Headline 2; `encoder_ab.png` (+ `encoder_per_class.csv`,
> `frozen_vs_finetuned.csv`) for the encoder A/B, the frozen→fine-tuned reversal, and the lr
> probe. Both quote the ladder's optimistic convention: best epoch of the best readout, chosen
> on test.

Branch `aug-ast-phases`. Rung G = Rung B with the encoder swapped: trainable AST (AudioSet)
→ mean patch tokens → Linear → BN → ArcFace over the same 23 machine sections. Same readouts,
same per-epoch eval, same CSV schema as Rung B. Augmentation = **mixup only**, per the Phase 0
verdict. Script: `diagnostics/section_ast_rungG.py`.

| model | input | mean AUROC (`maha_embed`) | seeds | converged |
|---|---|---|---|---|
| Rung B | 8-bit PNG 256×256 | 85.89 ± 0.26 | 3 | yes |
| Rung F (previous best deployable) | PNG | 86.62 | 1 | — |
| **Rung G — AST** | wav → fbank 1024×128 | **88.28 ± 0.36** | 3 | yes (peaks ep23/18/11) |
| **control — ResNet34** | wav → fbank 1024×128 | **88.65** @ ep49 | 1 | yes (end-slope −0.08) |
| A–E cross-rung oracle (test-peeking) | PNG | 88.80 | — | — |
| STgram-MFN | raw wave + log-Mel | 90.75 | — | — |

## Headline 1 — the encoder is a non-lever end-to-end

The hypothesis was that AST is the last feature-level lever with real headroom, on the strength
of `diagnostics/audio_backbone_probe.py` measuring AST at 76.8 vs ResNet34's 71.8 (+5.0) under an
identical **frozen** Maha readout. To test it fairly, the ResNet34 control is fed the **same
cached AST fbanks**, with the same objective, mixup, optimizer and AMP — so only the encoder
differs.

> **AST − ResNet34 = +0.01** (88.28 vs 88.27 at matched 30 epochs), and with the control run to
> convergence at 70 epochs, **ResNet34 wins by +0.37** (88.65 vs 88.28) — inside AST's seed
> spread, so the honest verdict is *tied*.

**The +5 frozen-feature advantage does not survive fine-tuning.** Once ResNet34 is allowed to
adapt to the same input, it catches up entirely, at roughly one tenth the compute per step. This
mirrors, in the opposite direction, the Rung A→B result where a frozen-regime *negative*
(`maha_embed` 67.0) reversed to +14.8 the moment the encoder was unfrozen. **Frozen-feature
rankings do not predict fine-tuned outcomes in this campaign — in either direction.**

Corrected framing to carry forward: AST is a **+5 frozen-feature lever and a ~0 end-to-end
lever**. Both are true; quote whichever matches the regime under discussion.

## Headline 2 — the input pipeline is the real lever, and it is the largest one measured

Every rung A–F, and every scan-curve / scorer / schedule / decoder ablation before them, ran on
8-bit PNGs resized 313×128 → 256×256. Feeding **Rung B's own encoder** raw wavs as native
1024×128 kaldi fbanks instead gives **88.65 vs 85.89 = +2.76** — larger than any encoder,
decoder, scan curve, scorer, or schedule effect ever measured in this project.

That makes **88.28–88.65 the best deployable result on the ladder**: +1.66 to +2.03 over Rung F,
level with the five-rung A–E cross-rung oracle (88.80) that peeks at the test set, and
**−2.10 from STgram-MFN**, down from Rung F's −4.13. It also posts the best **ToyConveyor**
figures in the campaign (69.0–71.4) — the class that resisted every previous lever and that drags
STgram-MFN itself down via id_02.

### Attribution — RESOLVED by the PNG counterpart

The control differs from Rung B in more than input (lr 1e-4 → 5e-5, Adam → AdamW, no AMP →
bf16, batch 64 → 32, 50 → 30/70 epochs), so +2.76 was only an *upper bound*. The clean PNG
counterpart — Rung B's script, mixup-only, matched lr 5e-5 and batch 32, 30 epochs, 3 seeds
(`runs/phase1_pngctl/`) — splits it:

| step | change | Δ |
|---|---|---|
| Rung B → PNG+mixup | recipe (mixup + lr + batch), input held at PNG | **+1.00** (85.89 → 86.89 ± 0.10) |
| PNG+mixup → fbank ctl | **input** (PNG → fbank), matched 30 epochs | **+1.38** (86.89 → 88.27) |
| 30ep → 70ep | schedule on fbank | +0.38 (88.27 → 88.65) |
| | **total** | **+2.76** |

**So the input pipeline is worth ≈ +1.4, not +2.8** — still the largest single lever in the
campaign, and still larger than the recipe change, but half the headline figure. The residual
+1.38 also carries the optimizer (Adam → AdamW) and AMP (none → bf16) differences, so it remains
a modest upper bound on input alone; those are second-order next to a representation change, but
they are not zero.

Note also that PNG+mixup at **86.89 ± 0.10** already beats Rung F (86.62) — i.e. even without
touching the input, mixup at a matched recipe is worth a new best. That is a cheaper, more
robustly-measured result (3 seeds, σ 0.10) than several single-seed ladder claims.

## Per-class: the two encoders are complementary, not one-dominates

AST − ResNet34 control, at each one's best epoch:

| fan | pump | slider | valve | ToyCar | ToyConveyor |
|---|---|---|---|---|---|
| −4.23 | +1.33 | +4.23 | −2.78 | +1.81 | −0.27 |

AST owns slider and ToyCar; ResNet34 owns fan and valve. This is the same two-model
complementarity Phase 0 found between augmented and un-augmented runs, and it feeds Phase 2's
fusion directly — now with two encoders over one input rather than two recipes over one encoder.

`maha_embed` beats `neg_cos` by 5–15 points at every setting tried, consistent with Phase 0
(mixup favours Maha) and with Rung C/E (the right readout is architecture-dependent).

## Secondary findings

- **lr is not a sensitive knob for AST.** An 8-epoch probe over flat 1e-5 / 5e-5+llrd 0.85 /
  1e-4+llrd 0.75 spanned only 0.55 (87.09 / 87.21 / 86.66), so selecting it on test costs little.
  The 3 seeds ran at flat 1e-5 — the simplest, with no LLRD hyperparameter at all.
- **LLRD arithmetic is a trap.** `lr_ast` is the *deepest* block's lr and block *i* gets
  `lr_ast * llrd**(n-i)`. At 1e-5/0.75 block 0 sits at ~3e-7 — effectively frozen, which would
  have tested a different question than the one asked, since Rung B's whole finding is that
  *unfreezing* the encoder is the lever.

## Carry-over into Phase 2

1. **Move the whole track off PNGs.** This is the single highest-value change available and it is
   orthogonal to everything already tried. Pending the clean attribution run.
2. **Drop AST** unless Phase 2 wants it purely as a *diversity* member — it is not worth 10×
   compute for a tie, but its per-class profile differs enough to fuse usefully.
3. **The fusion candidates are now AST and ResNet34 over one input**, which is cheaper and
   cleaner than the two-model oracle Phase 0 identified.
4. ToyConveyor finally moved (≈63–70 across the ladder → 71.4). Worth rechecking whether the
   id_02 "structural floor" was partly a PNG artifact.

Reproduce: `python docs/plot_phase0_aug.py --arms "ast=runs/section_rungG/ast_seed*"
"rn34fbank=runs/section_rungG/rn34fbank70_seed*" --min_epochs 30`.
