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
