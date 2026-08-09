# Phase 2, step 1 — per-section Mahalanobis banks and AS-norm

> **Figures** (regenerate with `python docs/plot_phase2_asnorm.py`, from the run CSVs — no
> retraining): `bank_lever.png` (+ `bank_lever.csv`, `bank_lever_per_section.csv`) for the
> bank/AS-norm result below; `vs_stgram.png` (+ `stgram_epoch_curve.csv`,
> `vs_stgram_per_section.csv`) for the epoch-for-epoch STgram-MFN comparison;
> `rung_h.png` (+ `rung_h.csv`) for `RUNG_H.md`.

Branch `aug-ast-phases`, built on Phase 1's fbank substrate. Code: `diagnostics/asnorm.py`,
wired into `diagnostics/section_ast_rungG.py`. Runs: `runs/section_rungG/rn34as_seed{0,1,2}`
(ResNet34, 70 epochs) and `astas_seed0` (AST, 30 epochs).

Two orthogonal knobs on the Mahalanobis readout, measured separately:

| knob | old (every rung A–G) | new |
|---|---|---|
| **bank** | one Gaussian per *class* | one per *section* (type×id, 23 units) |
| **norm** | raw distance | AS-norm: z-score by the clip's own section's train-score stats |

## Result: the bank is the lever; AS-norm is worth zero

ResNet34 on fbank, 3 seeds, **mean-of-per-ID AUROC** (STgram-MFN's own convention):

| readout | best epoch (test-selected) | final epoch (no selection) |
|---|---|---|
| `class_raw` (the existing readout) | 89.34 ± 0.27 | 88.99 ± 0.06 |
| `section_asnorm` | 91.38 ± 0.21 | 90.70 ± 0.19 |
| **bank lever** | **+2.04** | **+1.72** |

**AS-norm contributes exactly 0** to this metric (88.96 → 88.96 and 91.17 → 91.17 on seed 0),
and provably must: AUROC *within* a section is invariant under any strictly increasing
per-section transform, so z-scoring cannot reorder anything that the per-ID metric measures. It
moves only the *pooled* metric (+0.3–0.5), by making sections commensurable before they are
pooled. It remains a prerequisite for score fusion — a weighted sum of two models' raw distances
is otherwise dominated by whichever has the larger scale.

Why the bank matters so much: a class pools 3–4 machine units whose normal sound genuinely
differs, so a per-class Gaussian models the *mixture* rather than any real machine. Fitting per
unit removes that. **ToyConveyor 69.0 → 76.2** — the "structural floor" attributed to a dataset
property (id_02 being hard for STgram-MFN too) was substantially a pooled-bank artifact. The unit
is still the weakest, but ~7 points of it was the readout.

## Two traps that decide whether this is a SOTA claim

**1. STgram-MFN reports the mean of per-ID AUROCs, not the pooled-clip AUROC this campaign
computes.** (`STgram-MFN/results/STgram-MFN(m=0.7,s=30)/result.csv`: ToyCar
.8375/.9566/.9947/.99999 → Average .9472.) Per-section scoring specifically flatters the pooled
metric, so the mismatch bites exactly where it is most tempting. Checked here: the two metrics
agree closely (88.65 vs 88.96; 91.16 vs 91.17), so earlier ladder comparisons were not materially
wrong — but recompute mean-of-ID before any comparison.

**2. "Best epoch" is selected on the test set, and that is worth ~+0.7 of pure optimism.**

| selection rule | mean-of-ID | vs STgram 90.75 |
|---|---|---|
| best epoch, chosen on **test** (optimistic) | 91.38 ± 0.21 | +0.63 |
| **held-out 2-fold (HONEST)** | **91.15 ± 0.18** | **+0.40** |
| final epoch, no selection (floor) | 90.70 ± 0.19 | −0.05 |

> **The defensible claim is +0.40 over STgram-MFN, from `runs/section_rungG/rn34fold_seed*`.**
> Measured with `docs/select_heldout_epoch.py`: the epoch is chosen on one half of a
> section×label stratified split and scored on the disjoint half, both directions averaged, so
> no reported number took part in choosing itself. All three seeds are individually above 90.75
> (90.89 / 91.31 / 91.24). Selection optimism is **+0.23**, not the ~0.7 first inferred from the
> best-vs-final gap — the final epoch is the no-selection FLOOR, not an honest estimate, and an
> earlier revision of this file wrongly quoted it as the verdict. **Quote 91.15, not 91.38 and
> not 90.70.** Under the same honest rule the per-section bank is worth +1.90 (`class_raw`
> held-out 89.25 -> `section_asnorm` 91.15).

Using machine ID at test time is legitimate: DCASE-2020 task 2 provides it, STgram-MFN uses it
(its classifier is over machine IDs), and this ladder's `neg_cos` already depended on it.

## Encoder verdict unchanged

With the same per-section bank, AST reaches 90.44 mean-of-ID vs ResNet34's 91.38 — consistent
with Phase 1, where AST tied or lost to ResNet34 on identical input at ~10× the compute.

## Scope flag: there is no Mamba in any of this

The project goal is to match or beat STgram-MFN **with a MambaAD-adjacent architecture**. The
90.70 model is ResNet34 + ArcFace + per-section Maha — a baseline, not a MambaAD contribution.
The ladder walked away from Mamba one defensible step at a time (Rung C 84.4 < B 85.9; E − B =
+0.28 ± 0.50, a tie; D a clean negative).

But **every Mamba result in the campaign was measured on the PNG pipeline**, which Phase 1 showed
was handicapped and whose internal comparisons already produced one wrong conclusion ("input is
not a lever"). So "the Mamba decoder is a non-lever" is a PNG-era verdict that has never been
re-tested on fbank input, with mixup, with per-section banks — and Rung E's one robust effect was
per-class redistribution, which interacts directly with the bank change.

**Proposed Rung H:** Rung E/F's architecture (fine-tuned encoder + Mamba student) on the fbank
substrate, with held-out epoch selection built in, against the 90.70 / 91.38 baseline. Note the
fbank input is 1024×128 versus the PNG path's 256×256, and the Mamba scan curves and MFF/OCE
pyramid were tuned for square-ish feature maps — check the feature-map shapes before assuming
this is a flag flip.
