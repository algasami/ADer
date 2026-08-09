- AB test: Layer norm in HSS block swap for AdaLN (one-hot label)
- Preprocess mel-frequency spectrogram sliding window (arrival rate - congestion)
- Anomalib : PatchCore, ... legacy AD, baseline

## 5/6, 2026

Also need to export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/anaconda3/env/lib for modern library (Server outdated)


- `CUDA_VISIBLE_DEVICES=0 python run.py -c MambaAD/configs/mambaad/mambaad_mimii_toy.py -m train`
- `python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=12315 --use_env run.py -c MambaAD/configs/mambaad/mambaad_mimii_toy.py -m train`

## 5/13, 2026

- MVTec baseline
- fix MIMII issue
- AdaLN (backlog)

## 5/20, 2026

- PaDiM backbone, resnet change observe baseline
- swap layer normalization to AdaLN, for both PaDiM and mambaAD
- all changes: resnet, adaLN or not, FiLM or not
- copy PaDiM modify AdaLN, FiLM

## 6/17, 2026

- Fixed ratio issue (introduced anomalous images into training)
- Fixed testing set influenced by ratio (shouldn't)
- Look into STgram-MFN (SOTA?)
- pauc (dcase-2020-spectrogram)

## 6/30, 2026

- STgram-MFN for audio AD baseline

## 7/8, 2026

- ArcFace replaced by KNN - physical properties
- TgramNet to construct STgram for MambaAD?
- STgram-MFN baseline classifier replaced by MambaAD

## 7/13, 2026

- noticed training divergence in all three runs toy / stgram / stgram_delta:
  student weights (`mff_oce` convs, `A_logs`) grew steadily under constant lr=0.005 (step decay
  only at epoch 800) until overflow -> NaN (toy @ ~ep499, stgram @ ~ep599, stgram_delta @ ~ep399).
  surfaced later as sklearn "Input contains NaN" during test.
- root causes fixed:
  - loss never logged - configs logged term `cos` but trainer updates `pixel`
    (`update_log_term` silently no-ops on missing terms). renamed to `pixel` in the three configs.
  - added non-finite loss/grad guard so divergence can no longer silently contaminate weights.
- added `make_resume_ckpt.py`: rebuilds a resume ckpt from a finite `net_<epoch>.pth` snapshot.

## 7/22, 2026

- **lr/wd verdict (lowlr_lowwd_20260721 runs, lr 1e-3 / wd 1e-4, 200 ep):** stable (0 non-finite
  guard hits, flat curves) but underfit. Divergence was a long-schedule (1000-ep, decay@800) problem.
  The 200-ep 5e-3 runs never diverge. Also lowering wd 100x is the wrong lever for divergence.
  Best trained config so far = **5e-3 / 0.01 / 200 ep** + best-epoch selection (peak ~ep10-20).
- **Frozen-encoder probe now run on all 3 representations — encoder EXONERATED across the board.**
  Best frozen scorer (maha_concat, no decoder training): toy 71.8 / stgram 68.4 / delta 66.6 —
  each BEATS its trained MambaAD peak by +6.2 / +5.6 / +8.3. The reconstruction-distillation
  decoder + sp readout is the ceiling, not the ResNet34 encoder or the input representation.
  - signal is global: pooled maha/knn beat PatchCore-style `patch` by ~8-11 pts in every
    rep; yet default readout is sp_max (per-pixel max). sp_mean already > sp_max in trained runs.
  - per-class: only `slider` separates well everywhere (82-90); STgram lifts `fan` at encoder level
    (knn_layer1 ~71) but the trained decoder loses it.
- **Next (decoder-gap investigation):** re-score the "trained student" features with the same
  Maha/kNN probe. If trained-student+Maha ~= frozen-teacher+Maha -> only the cosine-residual
  readout is bad (cheap fix: swap readout). If trained-student+Maha << frozen -> decoder distorts
  the manifold (architectural). This bisects the downstream ceiling.

- Decoder-gap BISECTED (student_feature_probe.py, 720 net_20.pth peak ckpts):
  Scoring the trained Mamba *student's* GAP features with the same Maha as the
  frozen probe lands within ~1 pt of the teacher (decoder preserves the manifold):
  | rep | teacher+Maha | student+Maha | native residual sp_max | residual sp_mean |
  |-----|-----|-----|-----|-----|
  | toy          | 71.8 | 70.6 | 64.4 | 68.7 |
  | stgram       | 68.4 | 67.2 | 61.1 | 63.2 |
  | stgram_delta | 66.6 | 66.1 | 56.4 | 61.4 |
  The `1-cos(ft,fs)` residual + sp_max readout is 6-10 pts below the student's own separability.
  Also: student+Maha ~= teacher+Maha means the trained decoder adds ~0 under a Maha readout. The
  simplest strong pipeline is frozen teacher + Maha (71.8), no MambaAD training at all.

## 7/23, 2026

- **Scorer ablation promoted from one-off probe to a config/registry knob and swept over ALL
  epochs.** The `mimii/scorer/{cos-residual,student-maha,student-knn,teacher-maha,teacher-knn}`
  readouts were re-scored on the single e50 x log-Mel run (`...toy_20260722-215044`, net_2..50)
  via `docs/run_scorer_ablation.sh` (reuses `reeval_sp_mean.py`: same trained decoder at every
  saved epoch). cos-residual reproduces the native `metric.txt` to |Δ|=5e-4. Avg AUROC (sp_max):
  | scorer | start(ep2) | peak | final(ep50) |
  |-----|-----|-----|-----|
  | cos-residual (native) | 59.6 | **64.7** @ep10 | 62.0 |
  | student-maha          | 68.3 | **72.5** @ep40 | 72.3 |
  | student-knn           | 67.5 | **70.7** @ep48 | 70.7 |
  | teacher-maha (frozen) | 71.8 | **71.8** flat  | 71.8 |
  | teacher-knn  (frozen) | 71.5 | **71.5** flat  | 71.5 |
  - **All three distance readouts beat cos-residual by ~6-8 pts** — confirms the decoder-gap
    finding (the `1-cos` residual + sp pooling is the bottleneck, not the features).
  - **Refinement of "frozen beats trained":** scanning all epochs, student-maha (72.5@ep40)
    *edges past* the flat teacher-maha ceiling (71.8) — the earlier net_20-only probe (70.6 < 71.8)
    undersold the decoder. Given enough epochs + a Maha readout, the trained decoder adds a hair
    rather than zero; the student curve rises monotonically while cos-residual peaks ~ep10 then
    decays (its AUROC-peaks-early behaviour is a readout artifact, not a feature one).
  - teacher scorers are checkpoint-independent (frozen encoder, `static_fit`) -> flat lines.
    Maha edges kNN everywhere; sp_max == sp_mean for all image-level (Maha/kNN) scorers.
  - Outputs: `docs/plots/mimii_scorer/` — one folder per figure holding BOTH `plot.png` and its
    `data.csv`, plus `scorer_summary.csv` (peak/own-peak/final AUROC + AP/F1, both families) and
    `_data/<scorer>.txt` (42-col epoch-aligned metric files). Both sp_max and sp_mean families.

## 7/25, 2026

- **Scan-geometry ablation swept over all epochs, re-scored with the winning student-Maha readout
  (`mimii/scan-type-maha/*.py`, `docs/run_scan_ablation.sh` + `docs/plot_scan_ablation.py`).**
  Trained one e50 x log-Mel run per scan curve (hilbert/zorder/scan/sweep/zigzag), then re-scored
  every `net_<E>.pth` with `MahaScorer/student`. Avg AUROC @ avg-peak (ep40-50):
  | scan | Avg AUROC | peak ep |
  |-----|-----|-----|
  | zorder | **72.60** | 40 |
  | zigzag | 72.59 | 44 |
  | hilbert (baseline) | 72.54 | 40 |
  | scan | 72.47 | 40 |
  | sweep | 72.07 | 50 |
  - **Scan geometry is a non-lever: 0.5-pt spread across all 5 curves.** Even under the good Maha
    readout the decoder's scan order does not matter. Per-class picture unchanged: slider strong
    (~90), pump/valve/ToyCar middling (~72-75), fan/ToyConveyor weak (~58/65). Peaks-early still
    holds (~ep40). Outputs: `docs/plots/mimii_scan/` (`scan_summary.csv` + per-figure folders).

- **Backbone-swap probe — the last untested lever (`diagnostics/audio_backbone_probe.py`).**
  All downstream fronts (schedule/scan/scorer/decoder) were flat or a fixed +6; the remaining
  variable was the *feature extractor*. Swapped the frozen ImageNet ResNet34 for an
  audio-pretrained backbone used **natively on the raw MIMII wavs** (`data/dcase-2020/data_<cls>/
  <cls>/{train,test}/{normal,anomaly}_*.wav` — split verified identical to the image `meta.json`),
  then fit the SAME Maha/kNN scorers (imported from `frozen_encoder_probe.py`) on the SAME
  train-normal/test split. Only variable vs the 71.8 baseline = "ImageNet CNN" -> "AudioSet net".
  Best-scorer (Maha) mean AUROC:
  | backbone (frozen) | fan | pump | slider | valve | ToyCar | ToyConv | mean | Δ vs RN34 |
  |-----|-----|-----|-----|-----|-----|-----|-----|-----|
  | ResNet34 (ImageNet, image) | 58.4 | 72.1 | 90.8 | 70.0 | 75.7 | 63.8 | **71.8** | — |
  | CNN14 (AudioSet, audio)    | 50.4 | 67.8 | 90.3 | 71.9 | 76.0 | 58.7 | **69.2** | **−2.6** |
  | AST (AudioSet, audio)      | 56.7 | 82.3 | 93.6 | 78.9 | 82.7 | 66.4 | **76.8** | **+5.0** |
  | STgram-MFN (supervised)    | 87.1 | 90.9 | 98.9 | 98.6 | 94.7 | 74.3 | **90.7** | +18.9 |
  - **AST beats ResNet34 by +5.0 at the frozen-feature level** — backbone IS a real lever
    (pump +10, valve +9, ToyCar +7). AST embedding = mean over patch tokens (beats pooler);
    native `ASTFeatureExtractor` 128-mel/16k.
  - **CNN14 is −2.6, WORSE than ImageNet ResNet34** — so "audio-pretrained" is NOT automatically
    better; it's specifically AST (transformer+AudioSet). An audio *CNN* underperforms an image CNN
    here. Kills the naive "just use audio features" framing. (CNN14 = 16k checkpoint, sr-matched.)
  - **Best frozen backbone (AST 76.8) still −14 below STgram-MFN (90.7).** Backbone swap recovers
    only ~5 of the ~19-pt gap. Gap decomposition: 71.8 (ImageNet+UAD) → 76.8 (AST+UAD, **+5
    features**) → 90.7 (STgram-MFN, **+14 objective**). The dominant remaining lever is the
    **task framing** (supervised machine-ID/ArcFace discrimination), NOT the encoder. Even with
    per-class oracle backbone+rep+tap selection the frozen ceiling averages ~79.2, still −11.5 below
    STgram-MFN. (`fan` caveat: stays 50–58 for the audio backbones + log-Mel RN34, but the RN34
    STgram-image *shallow tap* `knn_layer1` reaches ~71.4 — fragile/tap-dependent, still −16 vs 87.)
  - **Scope caveat — this front has NO per-epoch curves, unlike scan/scorer, because nothing is
    trained.** The probe runs each backbone `.eval()`/`requires_grad=False` under `no_grad`, then
    fits Maha/kNN on the train-normal bank: no optimizer, no checkpoints, so no epoch axis exists.
    An epoch axis would add nothing anyway — the scorer ablation showed frozen-teacher scorers are
    checkpoint-independent by construction (`static_fit` → flat lines over all 25 epochs), and that
    "peaks-early" is a *cos-residual readout* artifact, not a feature property. So:
    - **established:** AST features are more separable than RN34 features under an identical
      readout/split/scorer (single variable, and Maha is already the best known readout).
    - **NOT established:** that a *trained* MambaAD on AST hits 76.8. That extrapolation leans on
      the decoder-gap bisection (student+Maha 72.5 ≈ teacher+Maha 71.8, within ~1 pt), measured on
      RN34 and assumed to transfer. Quote the +5 as a frozen-feature result, not end-to-end.
    - **why it stayed a probe:** the MambaAD teacher is `features_only=True, out_indices=[1,2,3]`
      (3-level spatial pyramid); AST emits one global vector per clip and the winning tap
      (`ast_meanpatch`) discards the time–freq layout. Teacher-slot AST = reshaping patch tokens to
      a grid at 3 depths + rebuilding fusion channel dims — a port, not an ablation flag.
    - **the free axis here is the TAP, not epochs:** AST pooler 73.4 vs mean-patch 76.8 = 3.3-pt
      spread from tap choice alone, comparable to the +5.0 headline, and only 2 taps (AST) / 1
      (CNN14) were tried. Sweeping AST hidden states by layer is the cheap follow-up.
  - Outputs: `runs/audio_probe/{ast,cnn14}/auroc.csv` + `run.log`. Deps added: `panns_inference`,
    `torchlibrosa` (env `mamba-ad`); `transformers` already present for AST. Both run single-GPU.

- **Campaign consolidated into `docs/ABLATION_SUMMARY.md`** (all fronts in one place: lever table,
  gap decomposition 71.8 → +5 AST → +14 objective → 90.7 STgram-MFN, per-class ceilings). Bottom
  line: the input-representation question is answered as a **negative result** — the downstream
  pipeline plateaus at ~72; the two remaining levers with headroom are the backbone (banked: AST
  +5, frozen-probe only — never trained end-to-end) and, dominantly, the **learning objective**
  (supervised machine-ID discrimination, +14), a new training track rather than an ablation of the
  existing one.

## 7/31, 2026

- **Seed-repeat on the labels/objective ladder's Rung B vs Rung E — corrects the earlier E>B
  claim.** Added `--seed` to `diagnostics/section_{finetune_rungB,combined_rungE}.py` and reran
  both at seed 1/2 (same 50ep/sub=2 protocol as the original seed-0 runs from 7/25-7/30). Mean
  AUROC: Rung B 85.86/85.58/86.22 → 85.89±0.26; Rung E 86.85/85.44/86.21 → 86.17±0.58. **E's
  seed-0 "+0.95, new best of every rung" does not reproduce** — B beats E in 2 of 3 seeds. B and
  E are statistically tied; the Mamba decoder does not reliably beat fine-tuning the encoder
  alone. Figure: `docs/plots/mimii_section_rungE/seed_repeat/`.
- **What IS robust across all 3 seeds: ToyCar trades off, ToyConveyor plateaus.** B beats E on
  ToyCar by ~6.7 pts in every single seed (93.3 vs 86.7) — Rung E's architecture choice reliably
  redistributes score toward slider/valve and away from ToyCar; it just doesn't move the mean.
  ToyConveyor is ~70 for both B and E in every seed — no rung/seed pushes it higher.
- **ToyCar/ToyConveyor frontier analysis** (`docs/plot_section_frontier.py`, writeup
  `docs/plots/mimii_section_frontier/CONCLUSION.md`) — splits the ladder's −3.9-to-STgram residual
  into two different problems:
  - ToyCar: Rung B alone already reaches 92.8-93.9 (only −1 to −2 vs STgram 94.7) — the gap is
    recoverable, Rung E's decoder just trades it away for other classes.
  - ToyConveyor: STgram-MFN's own per-machine-ID results (`STgram-MFN/results/.../result.csv`)
    show its class average is dragged down by one hard unit, id_02 (62.4 vs 74.85/85.51 for the
    other two ids). The new `id_breakdown.csv` logging (added to both scripts) shows the
    MambaAD-track ladder has the **exact same weak point** — id_02 is the worst ToyConveyor id in
    every one of the 4 new seed runs (both rungs, both readouts). This isn't a MambaAD-track
    failure; it's a dataset-intrinsic hard machine unit that trips up the supervised SOTA too.
  - **Cross-rung oracle** (best AUROC per class across all 5 rungs × all readouts — a
    val-selectable choice over models that already exist, not a new one) = 88.8 mean, only −1.9
    to STgram — vs −3.9 for the single best model (Rung E). Free headroom, zero new training,
    just by using Rung B's readout for ToyCar/ToyConveyor instead of Rung E's.
- Memory updated: `labels-objective-ablation-ladder` (now reflects the corrected B≈E verdict and
  the frontier split). Next open item on this ladder: a real (non-oracle) per-class/fused readout
  model, chosen on val not test.

## 8/8, 2026 — correction: AST is an OPEN lever, not a dismissed one

Retracting an earlier framing (in chat and, by implication, in the docs) that grouped the AST
backbone with the non-levers because it was "frozen-probe only / unproven end-to-end". That was
wrong in emphasis and is now fixed in `CLAUDE.md`, `diagnostics/CLAUDE.md`, and
`docs/ABLATION_SUMMARY.md` (§1, §3 lever table, §5, §7):

- **The measurement stands and is positive:** AST +5.0 (71.8 → 76.8) under an identical frozen
  Maha readout — the **largest feature-level effect in the whole campaign**, bigger than every
  representation/scan/schedule/decoder result combined.
- **AST was never trained end-to-end because that was a deliberate time/cost decision** (the
  teacher-slot port — patch tokens → 3-level spatial grid + fusion channel rebuild — was judged
  not worth the hours), **not because it failed, underperformed, or was refuted**. There is no
  negative evidence about AST end-to-end; there is *no* evidence, which is a different thing.
- The genuine epistemic caveat is unchanged and still worth quoting: *established* = AST features
  are more separable than RN34 features under the same readout; *not established* = that a trained
  MambaAD on AST reaches 76.8. Both statements are about missing measurement, not about a ceiling.
- Consequence for the ladder writeups: Rung F's "AST swap explicitly NOT included" should be read
  as **out of scope by choice**, not as a validated exclusion.

## 8/8, 2026 — Phase 1 / Rung G: AST finally trained end-to-end. It ties. The lever is the INPUT.

Branch `aug-ast-phases`. This **resolves** the open item from the correction entry above — AST
was the open lever, so it got measured rather than argued about. Writeup:
`docs/plots/phase1_rungG/CONCLUSION.md`. Script: `diagnostics/section_ast_rungG.py`.

Rung G = Rung B with the encoder swapped (trainable AST → mean patch tokens → ArcFace over the
same 23 sections, same readouts, same eval, same CSV schema). Augmentation = mixup only, per the
Phase 0 verdict below.

- **AST end-to-end is a non-lever.** With input held identical — both encoders fed the *same*
  cached AST fbanks, same objective/mixup/optimizer/AMP — **AST 88.28 ± 0.36 (3 seeds) vs
  ResNet34 88.27 (matched 30ep) = +0.01**; run the control to convergence (70ep) and **ResNet34
  wins, 88.65 vs 88.28**. At ~10× the compute per step.
- **The +5.0 frozen-feature result was real but not predictive.** It does not survive fine-tuning.
  This mirrors Rung A→B *in reverse* (a frozen-regime negative, maha 67.0, became +14.8 once the
  encoder was unfrozen). **General lesson for this campaign: frozen-feature rankings do not
  predict fine-tuned outcomes, in either direction.** Correct current phrasing: AST is a **+5
  frozen lever and a ~0 end-to-end lever** — quote whichever matches the regime.
  The correction entry above was right *as of its date*: "untested" ≠ "refuted", and the way to
  settle it was to test it. It is now tested.
- **What actually moved: the INPUT PIPELINE.** Every rung A–F, and every scan/scorer/schedule/
  decoder ablation, ran on 8-bit PNGs resized 313×128 → 256×256. Native 1024×128 kaldi fbanks
  from the raw wavs, same encoder, is the largest single lever measured in this project.
  Decomposed against a clean PNG counterpart (3 seeds, `runs/phase1_pngctl/`):

  | step | change | Δ |
  |---|---|---|
  | Rung B → PNG+mixup | recipe (mixup + lr + batch), PNG held | +1.00 → 86.89 ± 0.10 |
  | PNG+mixup → fbank | **input** (PNG → fbank), 30ep matched | **+1.38** → 88.27 |
  | 30ep → 70ep | schedule | +0.38 → 88.65 |

  So **input ≈ +1.4, not the +2.8** the raw Rung-B comparison first suggested (that figure was
  quoted as an upper bound and the counterpart run brought it down). The +1.38 residual still
  carries Adam→AdamW and bf16, so it remains a modest upper bound on input alone.
- **New best deployable: 88.65** (fbank, ResNet34, converged), vs Rung F 86.62. Gap to
  STgram-MFN: **−2.10**, down from −4.13. Best ToyConveyor in the campaign (69.0–71.4) — worth
  rechecking whether the id_02 "structural floor" was partly a PNG artifact.
- **PNG+mixup alone = 86.89 ± 0.10 (3 seeds) already beats Rung F (86.62)** without touching the
  input — cheap, and better-measured than several single-seed ladder claims.
- Per-class, AST and ResNet34 are **complementary** (AST − RN34: slider +4.2, ToyCar +1.8,
  pump +1.3; fan −4.2, valve −2.8) → Phase 2 fusion is now two encoders over one input.

### Phase 0 (same branch, prerequisite): augmentation
The whole ladder A–F trained with **zero augmentation** — `mimii/_base.py:100` assigns
`test_transforms = train_transforms`, the same list object. Adding it is readout-dependent:
maha +2.83, logit_nll +2.78, **neg_cos −4.15** (3 seeds each), i.e. **net −1.2 on
best-of-readout** — a headline negative that flips the readout ordering. Component split:
crop+masking alone is maha +0.32 / neg_cos −2.70 (harm, no benefit); **mixup** supplies
essentially the whole gain. Partly corrects a ladder claim: Rung B's "`logit_nll` overfits,
peaks at ep5" is substantially a missing-regularizer artifact (peak moves ep1.7 → ep19.0 under
mixup). Writeup: `docs/plots/phase0_aug/CONCLUSION.md`.

### Next
1. Move the track off PNGs (highest value, orthogonal to everything already tried).
2. Drop AST except as a fusion *diversity* member — not worth 10× compute for a tie.
3. Phase 2 fusion: AST + ResNet34 over one input, held-out per-class readout (Rung F mechanism).
4. Open: 3 seeds for the fbank control (currently 1); a clean input-only run that also holds
   optimizer/AMP fixed.

## 8/8, 2026 — Phase 2 step 1: per-section Maha banks. We MATCH STgram-MFN. Still no Mamba.

Writeup: `docs/plots/phase2_asnorm/CONCLUSION.md`. Code: `diagnostics/asnorm.py`.

Every readout A–G fit its Maha bank **per class**, but a class pools 3–4 machine units whose
normal sound genuinely differs, so the bank models a mixture rather than any real machine.
Fitting **per section** (type×id) instead, ResNet34 on fbank, 3 seeds, mean-of-per-ID AUROC:

| readout | best-ep (test-selected) | final-ep (no selection) |
|---|---|---|
| `class_raw` (existing) | 89.34 ± 0.27 | 88.99 ± 0.06 |
| `section_asnorm` | 91.38 ± 0.21 | **90.70 ± 0.19** |
| bank lever | +2.04 | **+1.72** |

- **AS-norm itself is worth 0** on the per-ID metric and provably must be — AUROC within a
  section is invariant under a strictly increasing per-section transform. It only moves the
  *pooled* metric (+0.3–0.5). Still needed as a prerequisite for score fusion.
- **ToyConveyor 69.0 → 76.2.** The "structural floor" (id_02 hard for STgram-MFN too) was
  substantially a **pooled-bank artifact**, not a data property. Revises the frontier writeup.
- **TRAP 1 — metric mismatch.** STgram-MFN reports the **mean of per-ID AUROCs**; this campaign
  computes the **pooled-clip AUROC**. Per-section scoring flatters the pooled one specifically.
  Checked: they agree closely here (88.65 vs 88.96; 91.16 vs 91.17), so earlier comparisons were
  not materially wrong — but always recompute mean-of-ID before comparing.
- **TRAP 2 — "best epoch" is selected on TEST, worth ~+0.7 of pure optimism.** best-ep 91.38
  (+0.63 vs STgram) but final-ep 90.70 (−0.05), last-20 mean 90.71, last-20 worst 90.26.
  **So the defensible claim is MATCH (90.70 vs 90.75), not beat. Do not quote 91.38.** A real
  held-out epoch selection should land in 90.70–91.38; that measurement does not exist yet.
- AST with the same bank: 90.44 — encoder verdict from Phase 1 unchanged.

### Scope flag (raised by the user): none of this contains Mamba
Goal is to match/beat STgram-MFN **with a MambaAD-adjacent architecture**; the 90.70 model is
ResNet34 + ArcFace + per-section Maha, i.e. a baseline, not a MambaAD contribution. The ladder
walked away from Mamba one defensible step at a time (C 84.4 < B 85.9; E−B = +0.28 ± 0.50 tie;
D negative). **But every Mamba result (C/D/E/F) was measured on the PNG pipeline**, which Phase 1
showed was handicapped and whose internal comparisons already yielded one wrong conclusion
("input is not a lever"). Proposed **Rung H** = Rung E/F architecture on fbank + mixup +
per-section banks, with held-out epoch selection built in, vs the 90.70/91.38 baseline. Watch:
fbank is 1024×128 vs PNG 256×256, and the scan curves / MFF-OCE pyramid were tuned for
square-ish feature maps.

## 8/9, 2026 — Rung H: the Mamba decoder COSTS 2.2 AUROC. Campaign verdict.

Writeup: `docs/plots/phase2_asnorm/RUNG_H.md`. True A/B, both arms identical except the decoder
(fp32, `scan_type=sweep`, lr_head 3e-4, mixup 0.2, fbank 512×128, 30ep, 3 seeds, per-section
Maha readout, held-out 2-fold epoch selection):

| arm | held-out mean-of-ID | vs STgram 90.75 |
|---|---|---|
| baseline (ResNet34 → GAP → ArcFace) | **91.01 ± 0.40** | +0.26 |
| Rung H (+ MFF/OCE → Mamba student) | **88.79 ± 0.72** | −1.96 |
| **decoder effect** | **−2.22** | |

~3× the seed spread, holds under every selection rule. Rung H trained cleanly (97–98% train acc,
ZERO non-finite steps) — a valid run that is simply worse. **Much stronger than the PNG-era tie**
(E−B = +0.28 ± 0.50 vs a ~86 baseline): here the decoder loses clearly against a baseline that
beats STgram-MFN.

Three silent defects had to be fixed before it would train at all (none raise an exception):
1. **autocast NaNs the selective-scan BACKWARD** while the forward stays finite (post-mortem:
   loss/emb/params/opt-state all finite, `gnorm=nan`) → epoch 1 clean, then ~1250/1258 steps
   skipped forever. Fix: fp32.
2. **`SCANS` assumes a square feature map** (`model/mambaad.py:53`, permutation over
   `size**dim`). On 128×32 the flat length is 4096 = 64² so no exception fires while the curve —
   computed for 64×64 — silently scrambles space. Only `sweep` is shape-agnostic. The scan-curve
   ablation's "all five equivalent" was measured on SQUARE inputs and does not transfer.
3. **`lr_head=1e-3` too high** on fbank (fine for C/E on PNG): train acc FELL 34→24→10 while
   loss fell. 3e-4 fixes it. The plain encoder trained first time at both lrs and in bf16.

### Campaign bottom line
**Best deployable: 91.15 ± 0.18** (ResNet34 + fbank 1024×128 + per-section Maha, honest held-out
epoch selection, 3 seeds) **vs STgram-MFN 90.75 → +0.40.** Lever sizes: per-section bank +1.90,
input pipeline ~+1.4, recipe (mixup/lr/batch) +1.00, AST encoder 0.00, AS-norm 0.00, Mamba
decoder **−2.22**. The goal "beat STgram-MFN" is met; the goal "with a MambaAD-adjacent
architecture" is **not**, and the evidence now says that combination is not reachable on this
task — the decoder is a liability, not a missing-tuning problem.

## 8/10, 2026 — Rungs A–F re-scored under the honest rule. B beats E; Rung F was worth nothing.

Closes `FINAL_REPORT.md` §8.1 and the §5 cross-era caveat — the whole ladder is now on the
final system's footing (mean-of-per-ID, epoch AND readout chosen on a held-out half).
Writeup: `docs/plots/ladder_honest/CONCLUSION.md`. Code: `docs/rescore_ladder.py`,
`docs/plot_rescore_ladder.py`, `docs/run_rescore_ladder.sh`, `diagnostics/heldout_eval.py`,
self-checks `docs/test_rescore_ladder.py`.

**It needed re-runs.** The July runs kept no checkpoints, no per-clip scores and no fold split,
so neither correction was recoverable from disk. All five rungs were repeated at their original
hyperparameters with per-clip score dumping (13 runs; seed coverage A×3 B×3 E×3 C×2 D×2, i.e.
the ladder's own history plus a free second seed for C/D). Rung F has no script and never did —
it is a post-hoc per-class readout policy over E, reconstructed from E's dumped scores.

| rung | pooled/test (published) | mean-of-ID/test | **mean-of-ID/held-out** |
|---|---|---|---|
| A | 79.94 ± 0.12 | 79.73 | **79.74 ± 0.36** |
| B | 86.66 ± 0.64 | 86.02 | **85.99 ± 0.63** |
| C | 84.40 ± 0.04 | 83.58 | **83.29 ± 0.10** |
| D | 84.66 ± 0.43 | 83.91 | **83.43 ± 0.01** |
| E | 86.01 ± 0.17 | 84.85 | **84.54 ± 0.16** |
| F  (E + per-class readout) | — | 84.87 | **84.47 ± 0.14** |
| F+ (per-class epoch too) | — | 88.80 | **88.16 ± 1.22** |

- **B > E by 1.45 under the honest rule** (published convention on the same re-runs: −0.65;
  July's 3-seed figure: +0.28, "a tie"). Both corrections push the same way and the gap is ~9x
  E's seed spread. Same direction as Rung H's −2.22 on fbank, so the PNG-era and fbank-era
  decoder verdicts now AGREE — the "every Mamba result was measured on the handicapped PNG
  pipeline" worry that motivated Rung H is resolved in Rung H's favour.
- **Rung F's +0.45 does not survive: F − E = −0.07.** The honest rule picks the same readout as
  the test-selected rule in all 13 runs (`neg_cos` for A/B, `maha_embed` for C/D/E), so there
  was no per-class readout disagreement left to exploit. **Do not quote 86.62 as a rung.**
- **The real headroom is a per-class EPOCH: F+ = +3.61 held-out (88.16, only −2.59 to STgram).**
  Survives held-out policy selection, so it is not oracle inflation — but it is up to six
  checkpoints, not one model. Upper bound / motivation for a real per-class early-stopping rule.
- **The metric correction dominates** (−0.2 to −1.2, always negative). **Selection optimism is
  ~0 for A/B but +0.29..+0.48 for the Mamba rungs**: A/B win on `neg_cos` (early sharp peak on
  real signal), C/D/E on `maha_embed` (late flat noisy plateau, winning epochs 15–48). Verified
  with a planted winners curse (`test_rescore_ladder.py` detects +1.15 by construction), so the
  ~0 is a property of the curves, not a peeking estimator.
- Re-run fidelity: Rung A's frozen `maha_concat_raw` is **bit-identical** to July (71.80); the
  headline readouts land within 0.1–0.8. But `neg_cos` is unstable run-to-run (±3.6, peak epoch
  9→40) while `maha_embed` is not (±0.6) — a test-selected max over a noisy readout is not a
  reproducible quantity.

### Damage report (my error, 2026-08-09)
The re-runs were first pointed at the ladder's existing run dirs; the rung scripts truncate
their CSVs at startup, so **Rung E seed1/seed2 lost their July `metric_curve.csv`,
`train_log.csv` and `id_breakdown.csv`** before I redirected them. `runs/` is gitignored. What
survives: `best_summary.csv` (written only at the end) for both, the salvaged mean-only curves
in `runs/section_rungE/log-Mel_seed{1,2}_july2026/metric_curve_from_log.csv` (recovered peaks
85.44 / 86.21 match the published seed-repeat exactly), and full backups of every other July
dir under `*_july2026/`. Permanently lost: per-epoch per-class AUROC and `logit_nll` for those
two runs. Gotcha now recorded in `docs/CLAUDE.md`: **always pass an explicit `--out_dir` when
repeating a rung.**
