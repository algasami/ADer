# MambaAD on MIMII — Final Report

**Branch:** `aug-ast-phases` · **Date:** 2026-08-09 · **Baseline to beat:** STgram-MFN, 90.75

---

## 1. Headline

**The system beats STgram-MFN by +0.40 AUROC. It contains no Mamba.**

| | mean AUROC | vs STgram-MFN |
|---|---|---|
| **Final system** (held-out epoch selection, 3 seeds) | **91.15 ± 0.18** | **+0.40** |
| STgram-MFN (supervised audio SOTA) | 90.75 | — |
| Previous best on the PNG track (mixup on Rung B's recipe, 3 seeds) | 86.89 ± 0.10 | −3.86 |

Measured on **STgram-MFN's own metric** (mean of per-machine-ID AUROCs) with the epoch chosen on
a held-out half of the test split, so no reported number participated in choosing itself. All
three seeds clear 90.75 individually (90.89 / 91.31 / 91.24).

The project's second goal — reach that level **with a MambaAD-adjacent architecture** — is
**not** met, and the evidence says it is not reachable on this task: on a matched substrate the
Mamba decoder *subtracts* 2.2 AUROC.

---

## 2. The final system

```
raw wav (16 kHz, 10 s)
  → kaldi fbank, 128 mel × 1024 frames, AudioSet normalization   [NOT the 8-bit PNG pipeline]
  → ResNet34 (ImageNet init), fully fine-tuned
  → concat GAP over stages 1–3  →  Linear(448→128)  →  BatchNorm
  → ArcFace over the 23 machine sections (type × id), sub-clusters = 2, s = 30, m = 0.5
       loss: mixup(α = 0.2) in the two-label ArcFace form
  → score at test time: Mahalanobis distance to a bank fitted PER SECTION on that
       section's normal training clips
```

Training: AdamW, lr 5e-5 (encoder) / 1e-3 (head), wd 1e-4, bf16, batch 32, 70 epochs, grad-clip
5.0 with a non-finite-step guard. Runs: `runs/section_rungG/rn34fold_seed{0,1,2}`.

The objective uses **machine-section metadata**, not anomaly labels, so the method remains
unsupervised with respect to anomalies. Machine ID is available at test time in DCASE-2020
task 2, and STgram-MFN relies on it too.

### Per-class result (mean-of-per-ID AUROC, held-out selection, seed-averaged)

| class | ours | STgram-MFN | Δ |
|---|---|---|---|
| fan | 85.67 | 87.09 | −1.43 |
| pump | 92.73 | 90.94 | **+1.78** |
| slider | 97.52 | 98.87 | −1.35 |
| valve | 99.21 | 98.59 | +0.63 |
| ToyCar | 97.49 | 94.72 | **+2.77** |
| ToyConveyor | 75.14 | 74.27 | +0.87 |
| **mean** | **91.29** | **90.75** | **+0.54** |

(The 91.29 here averages the per-class figures at both fold-selected epochs; the 91.15 headline
is the stricter fold-B-given-fold-A estimate. Quote 91.15.)

---

## 3. What moved the needle — and what didn't

Every entry measured with everything else held fixed.

| lever | effect | verdict |
|---|---|---|
| **Per-section Mahalanobis bank** (vs per-class) | **+1.90** | largest readout lever ever found here |
| **Input pipeline** — raw-wav fbank vs 8-bit PNG | **≈ +1.4** | largest representation lever |
| **Recipe** — mixup + lr + batch | **+1.00** | free; mixup is the useful half |
| AST encoder (vs ResNet34, identical input) | **+0.01** | non-lever end-to-end |
| AS-norm (on top of per-section banks) | **0.00** | provably zero on the per-ID metric |
| **Mamba decoder** | **−2.22** | active liability |

Prior campaign levers, for completeness: input representation among PNG variants, decoder
depth, scan curve, schedule, and scorer were all flat (≈72 plateau); the objective/labels axis
(rungs A→B) was worth ~+14 and remains the single biggest contribution overall. Note what that
+14 is measured on: it is the **`maha_embed` readout** going from frozen to fine-tuned (A 67.4
→ B 82.1 in the re-runs, reproducing the +14.8). On each rung's *best* readout the A→B step is
**+6.25** under the honest rule (79.74 → 85.99), because Rung A already recovers most of the
gap through its classification readout alone. Both are true; say which one you mean.

---

## 4. Three negative results worth reporting

### 4.1 AST is a frozen-feature lever and a zero end-to-end lever

The frozen probe measured AST at 76.8 vs ResNet34's 71.8 (+5.0) under an identical Mahalanobis
readout — the largest feature-level effect in the campaign, and untested end-to-end. Trained
end-to-end on identical cached fbanks:

| | mean AUROC |
|---|---|
| AST (3 seeds, 30 ep) | 88.28 ± 0.36 |
| ResNet34, same input, matched 30 ep | 88.27 |
| ResNet34, run to convergence (70 ep) | **88.65** |

**+0.01 at matched budget; ResNet34 wins by +0.37 at convergence**, at roughly one tenth the
compute per step.

The general lesson: **frozen-feature rankings do not predict fine-tuned outcomes here, in
either direction.** Rung A→B showed the mirror image — a frozen-regime *negative*
(`maha_embed` 67.0) became +14.8 once the encoder was unfrozen. Frozen probes are cheap and
were the right first move, but they are hypothesis generators, not verdicts.

### 4.2 AS-norm is exactly zero, and must be

Score normalization per section changes nothing on the per-ID metric — **AUROC within a section
is invariant under any strictly increasing per-section transform**. It shifts only the
pooled-clip metric (+0.3–0.5) by making sections commensurable before pooling. It remains a
prerequisite for any future score fusion, since a weighted sum of two models' raw distances is
otherwise dominated by whichever has the larger scale.

The lever was the **bank**, not the normalization: fitting one Gaussian per *class* pools 3–4
machine units whose normal sound genuinely differs, so the bank models a mixture rather than any
real machine.

### 4.3 The Mamba decoder costs 2.2 AUROC

A true A/B — identical fp32, `scan_type=sweep`, lr_head 3e-4, mixup, fbank 512×128, 30 epochs,
3 seeds, per-section readout, held-out selection — differing only in the decoder:

| arm | held-out mean-of-ID | vs STgram |
|---|---|---|
| ResNet34 → GAP → ArcFace | **91.01 ± 0.40** | +0.26 |
| + MFF/OCE → Mamba student | **88.79 ± 0.72** | −1.96 |
| **decoder effect** | **−2.22** | |

~3× the seed spread, stable under every selection rule, with Rung H training cleanly (97–98%
train accuracy, zero non-finite steps) — a valid run that is simply worse.

This supersedes the earlier PNG-era reading. There, Rung E − Rung B was +0.28 ± 0.50 against an
~86 baseline, i.e. a tie. Here the decoder loses clearly against a baseline that beats
STgram-MFN. **The MambaAD framing does not survive contact with MIMII once the input pipeline
and readout are fixed.** It is a liability, not a tuning problem.

---

## 5. Methodology corrections (these affect every earlier number)

**5.1 The whole ladder trained with zero augmentation.**
`MambaAD/configs/mambaad/mimii/_base.py:100` assigns `test_transforms = train_transforms` — the
*same list object* — so training views were deterministic. Consequence: Rung B's recorded
"`logit_nll` overfits, peaks at epoch 5" was substantially a missing-regularizer artifact; under
mixup that readout's peak moves to epoch 19 and it gains +2.78.

**5.2 STgram-MFN reports the mean of per-ID AUROCs; this campaign computed pooled-clip AUROC.**
Different quantities. Per-section scoring specifically flatters the pooled one, so the mismatch
bites exactly where it is most tempting. Checked: they agree closely here (88.65 vs 88.96;
91.16 vs 91.17), so earlier comparisons were not materially wrong — but recompute mean-of-ID
before any comparison.

**5.3 "Best epoch" was selected on the test set throughout, worth ~+0.23 of optimism.**
Now fixed by `docs/select_heldout_epoch.py`: choose the epoch on one half of a section×label
stratified split, score the disjoint half, average both directions.

| selection rule | mean-of-ID | vs STgram |
|---|---|---|
| test-selected (optimistic) | 91.38 ± 0.21 | +0.63 |
| **held-out 2-fold (honest)** | **91.15 ± 0.18** | **+0.40** |
| final epoch (no-selection floor) | 90.70 ± 0.19 | −0.05 |

Two symmetric errors are easy here and both were made in sequence during this work: quoting the
test-selected number, then over-correcting to the final epoch and calling it a tie. The final
epoch is a floor, not an estimate.

**5.4 ToyConveyor's "structural floor" was mostly a readout artifact.**
Previously attributed to a dataset property because id_02 is hard for STgram-MFN too. The
per-section bank lifts ToyConveyor 69.0 → 76.2. The unit is still the weakest, but ~7 points of
it was the pooled bank.

> **Cross-era comparisons — RESOLVED (2026-08-10).** Rungs A–F were scored as pooled-clip AUROC
> with test-selected epochs, the final numbers as mean-of-per-ID with held-out selection. The
> ladder has now been re-run (the originals kept no checkpoints and no per-clip scores, so this
> could not be done from disk) and re-scored on the final system's footing —
> `docs/plots/ladder_honest/CONCLUSION.md`:
>
> | rung | A | B | C | D | E | F | F+ |
> |---|---|---|---|---|---|---|---|
> | mean-of-ID @ held-out | 79.74 | **85.99** | 83.29 | 83.43 | 84.54 | 84.47 | 88.16 |
>
> Two published claims do not survive: **B beats E by 1.45** (the "B ≈ E tie" was partly a
> convention artifact, and the honest verdict now agrees in sign with Rung H's −2.22), and
> **Rung F's per-class readout policy is worth −0.07, not +0.45** — do not quote 86.62 as a
> rung. The headroom it was reaching for is real but needs a per-class *epoch* (F+, +3.61
> held-out), which is up to six checkpoints rather than one model.

---

## 6. Engineering notes that cost real time

- **Autocast NaNs Mamba's selective-scan *backward*** while the forward stays finite — at the
  first bad step, loss, embeddings, inputs, all parameters and all optimizer state were finite
  and only `gnorm` was NaN. Symptom: epoch 1 clean, then ~1250/1258 steps skipped forever. Run
  the Mamba path in fp32.
- **`SCANS` assumes a square feature map** (`model/mambaad.py:53`, permutation over
  `size ** dim`). On a 128×32 map the flat length is 4096 = 64², so **no exception fires** while
  the Hilbert curve — computed for 64×64 — silently scrambles space. Only `sweep` (raster) is
  shape-agnostic. The scan-curve ablation's "all five equivalent" was measured on *square*
  inputs and does not transfer. **Absence of a crash is not evidence of correctness.**
- **`lr_head=1e-3` is too high for the Mamba student on fbank** (fine for rungs C/E on PNG).
  Not divergence — train accuracy *fell* 34 → 24 → 10 while loss fell. 3e-4 fixes it.
- **Thread pinning matters**: unpinned, 12 fbank-cache workers ran at exactly the serial rate
  and drove load average past 380, because torch defaults to one intra-op thread per core per
  process. `torch.set_num_threads(1)` → ~140× speedup.
- The **non-finite guard + per-epoch skip counter** is what made all of the above debuggable.
  Without it these failures present as an sklearn "SVD did not converge" hours later.

---

## 7. Limitations

1. **The held-out split comes from the test set.** MIMII's train split is normal-only, so AUROC
   cannot be computed there and a validation set carved from training data is impossible.
   Splitting test is the standard workaround and the reported half never participates in its own
   selection — but it is not an independent third set.
2. **One STgram-MFN number, no seed variance.** Taken from the submodule's `result.csv`. The
   +0.40 margin is 2σ of *our* seed noise; we cannot bound theirs.
3. **`lr_head` was never swept for the final system.** Dropping it from 1e-3 to 3e-4 was worth
   ~+1.3 at 512×128. The headline config may not be at its optimum.
4. **The input attribution (≈+1.4) still carries optimizer and AMP differences**; a strict
   single-factor input run does not exist.
5. **Rung H used a time-pooled 512×128 input** (the Mamba decoder rejects 1024×128 outright).
   The −2.22 is measured against a matched 512×128 baseline, so the comparison is fair, but the
   decoder was never tested at full time resolution because it cannot be.

---

## 8. If the work continues

1. ~~**Re-score rungs A–F under the honest rule**~~ — **done 2026-08-10**, see §5 and
   `docs/plots/ladder_honest/CONCLUSION.md`. The publication table exists; two ladder claims
   changed. Remaining wrinkle: the ladder's folds are path-keyed while Rung G/H's were
   index-keyed, so an A–H table still spans two fold definitions.
2. **A per-class early-stopping rule.** The re-score found +3.61 (held-out) available from
   choosing the epoch per class — the largest untaken lever on the PNG track, and it survives
   held-out policy selection. It currently costs six checkpoints; making it one model is the
   open problem.
3. **Sweep `lr_head` on the final system** — the cheapest remaining upside.
3. **Score fusion.** Per-clip scores are now persisted (`scores_best.npz`). Phase 0 and Phase 1
   both found complementary models (augmented vs not; AST vs ResNet34), with per-class oracles
   ~+2 over the best single arm. AS-norm exists precisely to make that fusion well-posed.
4. **Move the repo's default track off PNGs**, since that pipeline is now known to cost ~1.4.

---

## 9. Reproduction

```bash
conda activate mamba-ad
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/anaconda3/envs/mamba-ad/lib

# build the fbank cache once (~8 GB, ~60 s with 12 pinned workers)
python diagnostics/section_ast_rungG.py --build_cache_only --cache_procs 12

# the final system, 3 seeds
for s in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$s python diagnostics/section_ast_rungG.py \
    --epochs 70 --seed $s --decoder none --backbone resnet34 \
    --time_pool 1 --batch 32 --lr_ast 5e-5 --tag rn34fold &
done; wait

# the honest number
python docs/select_heldout_epoch.py 'runs/section_rungG/rn34fold_seed*' --mode section_asnorm
```

**Supporting writeups:** `docs/plots/phase0_aug/CONCLUSION.md` (augmentation),
`docs/plots/phase1_rungG/CONCLUSION.md` (AST / input pipeline),
`docs/plots/phase2_asnorm/CONCLUSION.md` (per-section banks),
`docs/plots/phase2_asnorm/RUNG_H.md` (Mamba A/B). Dated log: `NOTE.md`.
