# Rungs A–F re-scored under the honest rule

Branch `aug-ast-phases`. Closes the outstanding item in `docs/FINAL_REPORT.md` §8.1 and the §5
"caveat on cross-era comparisons": rungs A–F were quoted as **pooled-clip AUROC at a
test-selected epoch**, the final system as **mean-of-per-ID AUROC at a held-out-selected
epoch**, and no table could mix them. All six rungs are now on the final system's footing.

Code: `docs/rescore_ladder.py` (+ `plot_rescore_ladder.py`, driver `run_rescore_ladder.sh`,
self-checks `test_rescore_ladder.py`). Runs: 13, listed in `per_run.csv`.

## It required re-running the ladder

The original runs kept **no checkpoints and no per-clip scores** — only per-class AUROC
aggregates — and no fold split. Mean-of-per-ID needs per-section scores; held-out selection
needs the fold. Neither is recoverable from what was on disk, so each rung was repeated at its
original hyperparameters (recovered from the run logs) with per-clip score dumping on
(`diagnostics/heldout_eval.py`). Seed coverage matches the ladder's own history — A×3, B×3,
E×3, C×2, D×2 — so the comparison is like-for-like.

**Rung F has no script and no run directory**; it was always a post-hoc per-class readout
policy over Rung E, so it is reconstructed here from Rung E's dumped scores rather than
retrained.

### The re-runs are faithful
Rung A's `maha_concat_raw` reproduces **bit-identically** (71.80 on every class — it is the
frozen-encoder path), which pins the data pipeline, weights and split to July. Headline
readouts land within 0.1 (C), 0.2 (A), 0.7 (D), 0.8 (E) and 0.8 (B) of their published values.
One caveat that is itself a finding: `maha_embed` is stable run-to-run (±0.6) while `neg_cos`
is not (±3.6, with its peak epoch wandering from 9 to 40). Test-selected maxima of a noisy
readout are not reproducible quantities.

## Result

| rung | pooled-clip @ test (published convention) | mean-of-ID @ test | **mean-of-ID @ held-out** | vs STgram 90.75 |
|---|---|---|---|---|
| A  frozen enc + head | 79.94 ± 0.12 | 79.73 | **79.74 ± 0.36** | −11.01 |
| B  fine-tuned encoder | 86.66 ± 0.64 | 86.02 | **85.99 ± 0.63** | −4.76 |
| C  Mamba, frozen enc | 84.40 ± 0.04 | 83.58 | **83.29 ± 0.10** | −7.46 |
| D  joint distill+classify | 84.66 ± 0.43 | 83.91 | **83.43 ± 0.01** | −7.32 |
| E  B+C combined | 86.01 ± 0.17 | 84.85 | **84.54 ± 0.16** | −6.21 |
| F  E + per-class readout | — | 84.87 | **84.47 ± 0.14** | −6.28 |
| F+ per-class epoch *and* readout | — | 88.80 | **88.16 ± 1.22** | −2.59 |

Figure: `plot.png`; table: `data.csv`; per-run: `per_run.csv`.

The honest rule selects **both** the epoch and the readout on fold A and reports fold B, both
directions averaged, over 20 independent stratified fold draws (one draw is noisy: the reported
half is ~5400 clips and the selection ranges over 50 epochs × 3–7 readouts).

## Three things change

### 1. B beats E — the "B ≈ E tie" was partly a convention artifact

| convention | E − B |
|---|---|
| pooled-clip @ test (published) | −0.65 (July's own 3-seed figure: **+0.28**, "a tie") |
| mean-of-per-ID @ test | −1.17 |
| **mean-of-per-ID @ held-out (honest)** | **−1.45** |

Both corrections push the same way, and the honest gap is ~9× E's seed spread. The 7/31 seed
repeat had already downgraded "E > B by +0.95" to a tie; under the honest rule it becomes a
clear **loss for the Mamba decoder**, in the same direction and roughly half the size of Rung
H's −2.22 on the fbank substrate. The PNG-era and fbank-era verdicts now agree instead of
conflicting, which removes the main reason `phase2_asnorm/CONCLUSION.md` gave for re-testing
the decoder at all.

### 2. Rung F's mechanism is worth nothing (−0.07), not +0.45

Rung F was published at 86.62 against E's 86.17 — a +0.45 gain from choosing the readout per
class on a held-out half. Reconstructed and scored under the honest rule, **F = 84.47 vs E =
84.54: −0.07.** The per-class readout policy, at a single shared epoch, adds nothing.

The reason is visible in `per_run.csv`: the honest rule picks the *same* readout as the
test-selected rule in all 13 runs (`neg_cos` for A/B, `maha_embed` for C/D/E). There was no
per-class readout disagreement left to exploit.

### 3. The real headroom needs a different epoch per class, and it survives held-out selection

Letting the epoch vary per class as well (**F+**) is worth **+3.61 held-out** — 88.16, only
−2.59 from STgram-MFN, from Rung E alone. This is the honest version of the July "cross-rung
oracle = 88.8" headroom, and it is not oracle-inflated: 88.80 is the test-side number and
88.16 survives choosing the policy on a disjoint half.

But it is **up to six checkpoints, not one model** — per class, a different epoch means a
different set of weights. Quote it as an upper bound on the per-class-policy family, or as
motivation for a real per-class early-stopping rule, never as a deployable single system.
It also carries the largest seed spread on the table (±1.22).

## The two corrections are not the same size, and not uniform

| rung | §5.2 metric (pooled → mean-of-ID) | §5.3 selection (test → held-out) | total |
|---|---|---|---|
| A | −0.21 | +0.01 | −0.20 |
| B | −0.64 | −0.03 | −0.66 |
| C | −0.83 | −0.29 | −1.12 |
| D | −0.75 | −0.48 | −1.22 |
| E | −1.16 | −0.31 | −1.46 |

**The metric correction dominates and always costs something** (−0.2 to −1.2): pooled-clip
AUROC flatters every rung, most where scores are least commensurable across machine units.

**Selection optimism is ~0 for A and B but +0.3 to +0.5 for the Mamba rungs C/D/E.** That is
not a quirk of the estimator — the planted-winners-curse self-check in
`test_rescore_ladder.py` shows it detects +1.15 where inflation exists by construction. It is a
property of the curves: A and B win on `neg_cos`, which peaks early and sharply on real signal,
whereas C/D/E win on `maha_embed`, which drifts up over 50 epochs to a flat noisy plateau
(winning epochs 15–48) where the maximum is partly luck. **The selection correction
specifically penalizes the Mamba rungs**, which is the second reason the B-vs-E verdict moves.

Phase 2's fbank runs measured +0.23 of optimism, so this ladder's Mamba rungs are, if anything,
slightly worse-behaved than the final system.

## Caveats

1. **Not a bit-reproduction.** CUDA is not bit-exact, and `neg_cos`'s test-selected maximum is
   genuinely unstable (B seed0 drew 87.51 against July's 85.86). Read the table as one
   internally consistent re-measurement under one code version, not as a correction of each
   individual July number.
2. **The folds are not the ones Rung G/H used.** These are path-keyed (so every rung holds out
   exactly the same clips, whatever order the loader returns them in); G/H used index-keyed
   draws from `asnorm.make_folds`. Both are valid stratified splits and the estimate is
   averaged over 20 draws, but an A–H table is comparing across two fold definitions.
3. **C and D have 2 seeds, A/B/E have 3.** D's ±0.01 is two runs agreeing, not a tight
   distribution.
4. The held-out halves come from the test split — MIMII's train split is normal-only, so a
   third independent set is impossible. Same limitation as `FINAL_REPORT.md` §7.1.
