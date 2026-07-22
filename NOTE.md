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
