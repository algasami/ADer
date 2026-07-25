# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this fork is

A research fork of **ADer** (multi-class unsupervised anomaly detection) whose goal is to run
**MambaAD** on the **MIMII** audio dataset (DCASE-2020) by treating **spectrograms as images**:
MIMII recordings are pre-converted to 3-channel PNGs which flow through the otherwise-unchanged
image AD engine. Baselines for comparison: PaDiM (anomalib track) and STgram-MFN (submodule).

The active research question is which input representation works best. Three ablation fronts:
- **log-Mel toy**: (log-Mel, delta, delta-delta) → `data/dcase-2020-spectrogram`
- **STgram-Sgram**: (Sgram, learned Tgram, Sgram) → `data/dcase-2020-stgram`
- **STgram-delta**: (Sgram, learned Tgram, delta) → `data/dcase-2020-stgram-delta`

The Tgram channel comes from a **frozen TgramNet** (weights from the STgram-MFN baseline
checkpoint). Full design rationale: `docs/stgram-mambaad/PLAN.md`.

> **Status (2026-07-25): the input-representation question is answered, and it's a negative
> result.** No representation, decoder, scan curve, or schedule moves the needle — every
> downstream lever plateaus at ~72 AUROC. log-Mel actually *beats* both STgram variants. The
> only downstream lever that helps is the test-time readout (Maha, +6–8); the only feature lever
> is the backbone (AST, +5); the dominant remaining gap to STgram-MFN (−14) is the *learning
> objective*, not the encoder or input. Full write-up: **`docs/ABLATION_SUMMARY.md`**.

### Three code surfaces coexist — keep them straight
- **Root ADer engine** (`run.py`, `configs/`, `model/`, `trainer/`, `data/`, `loss/`, `optim/`,
  `util/`) — the registry-driven training/eval framework. This is what `run.py` drives.
- **`MambaAD/`** — the forked MambaAD codebase. Holds the MIMII configs under
  `MambaAD/configs/mambaad/` that the root `run.py` loads.
- **`src/`** — a *separate* anomalib-based experiment track (PaDiM baselines, AdaLN/FiLM variants,
  anomalib datamodules) **plus the spectrogram-image generators**. NOT wired into `run.py`.
- (`STgram-MFN/` is a git submodule — the audio-AD baseline and the source of TgramNet weights.)

## Environment

```bash
conda env create -f environment.yml   # creates env "mamba-ad" (Python 3.10)
conda activate mamba-ad
# On this server (outdated libs), export before running or Mamba CUDA kernels fail to load:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/anaconda3/envs/mamba-ad/lib
```

## Common commands

```bash
# Train / test MambaAD on MIMII (single GPU); swap the leaf for stgram / stgram-delta
CUDA_VISIBLE_DEVICES=0 python run.py -c MambaAD/configs/mambaad/mimii/e50/log-Mel.py -m train
CUDA_VISIBLE_DEVICES=0 python run.py -c MambaAD/configs/mambaad/mimii/e50/stgram.py -m test

# Config overrides = trailing path.key=value args (parsed by run.py's REMAINDER opts)
python run.py -c <cfg> -m train data.cls_names=pump trainer.checkpoint=runs/my_run

# Multi-GPU (DDP)
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 \
  --master_addr=127.0.0.1 --master_port=12315 --use_env run.py -c <cfg> -m train
```

- MIMII configs live under `MambaAD/configs/mambaad/mimii/<ablation-type>/<input-type>.py`.
  All leaves are thin subclasses of `mimii/_base.py` (`cfg_mimii_base`) that only flip the
  ablation knobs (`ABL_EPOCH`/`ABL_LR_BASE`/`ABL_WD`/`ABL_METRICS`), the input representation
  (`INPUT_ROOT`), and/or the scan curve (`SCAN_TYPE`).
  - ablation-type: `e50` (super-short, lr 5e-3/wd 0.01 — current default), `e200`
    (lr 5e-3/wd 0.01), `e1000` (long schedule, decay@800, sp_max-only 21-col metrics),
    `lowlr-lowwd` (lr 1e-3/wd 1e-4, 200 ep — stable but underfit).
  - scan-type ablation (log-Mel + e50, varying scan curve): `mimii/scan-type/{hilbert,scan,
    sweep,zigzag,zorder}.py` (`hilbert` == the `e50/log-Mel` baseline). Verdict: **non-lever**
    (all 5 curves land 72.07–72.60 avg AUROC, 0.5-pt spread). Companion set `mimii/scan-type-maha/
    {hilbert,scan,sweep,zigzag,zorder}.py` re-scores each scan run with the winning student-Maha
    readout (`ABL_SCORER=MahaScorer/student`) — used by `docs/run_scan_ablation.sh`.
  - scorer-type ablation (log-Mel + e50, varying the *test-time score readout*):
    `mimii/scorer/{cos-residual,student-maha,student-knn,teacher-maha,teacher-knn}.py`
    (`cos-residual` == the native `e50/log-Mel` readout). The scorer is orthogonal to
    training — the loss/checkpoint are unchanged; only how features are turned into a per-clip
    score differs. So the cleanest use is `-m test` with `model.kwargs.checkpoint_path=<net.pth>`
    re-scoring one trained checkpoint several ways. Motivated by the student/frozen probes:
    Maha/kNN on GAP features beat the cosine-residual sp_max/sp_mean readout (see the decoder-gap
    finding). Engine: `util/scorer.py` (`SCORER` registry); knob: `ABL_SCORER` in `mimii/_base.py`.
    Maha/kNN fit a per-class bank on the normal train split (one extra forward pass per eval) —
    **run single-GPU** (the bank is not gathered across DDP ranks); image-level scorers make
    sp_max == sp_mean by construction (both metric columns report the same number).
- Visualization: add `vis=True vis_dir=<dir>` to a test run.
- Single-class sweep helper: `runs_single_class.py` (`-d <dataset> -c <cfg> -n <num_procs> -m <mode> -g <gpu>`).
- PaDiM baseline (anomalib track, *not* `run.py`): `cd src && python padim_baseline.py`
  (MVTec) or `python padim_baseline_mimii.py` → writes `padim_performance.csv`.

## Data

MIMII classes: `fan, pump, slider, valve, ToyCar, ToyConveyor`.

Datasets are **pre-generated PNGs plus a `meta.json` per root** — both steps are required before
training. To (re)build them, use the `mimii-data-pipeline` skill.

## Architecture notes (the non-obvious parts)

**Config = Python class inheritance, not YAML.** Configs subclass bases in `configs/__base__/`
(`cfg_common`, `cfg_dataset_default`, `cfg_model_mambaad`).

**Trainer:** `trainer/_base_trainer.py` holds the DDP/AMP/logging/checkpoint loop; MambaAD uses
`MAMBAADTrainer` (`trainer/mambaad_trainer.py`) — frozen teacher (resnet34) → Mamba-decoder
student, trained with **`CosLoss` under log-term name `cos`**. The base trainer has a
**non-finite loss/grad guard** that skips bad steps so divergence can't silently corrupt weights.

> Gotcha: `update_log_term` silently no-ops when a config's logged term name doesn't match what
> the trainer computes — a mismatched name means the loss is never logged, with no error. Keep
> config `loss_terms`/`log_terms` names and the trainer in sync (currently both `cos`).

**Metrics:** current MIMII configs record both the `*_sp_max` and `*_sp_mean` pooling families
(6 metrics, 42-column `metric.txt`); **older runs recorded sp_max only (21 columns)**, so any
tool reading `metric.txt` must handle both layouts.

**Scorer:** `CosResidualScorer` is the default whenever a config sets no `cfg.scorer`, so
non-MIMII configs are unaffected by the scorer registry.

## Audio-specific customizations vs. stock ADer

- Dataset roots `dcase-2020-spectrogram` / `dcase-2020-three-channel` / `dcase-2020-stgram` /
  `dcase-2020-stgram-delta` are special-cased in `data/ad_dataset.py:69` (`dcase-2020-three-channel`
  is a **legacy alias**, still tolerated by the code but no longer populated — the log-Mel root is
  now `dcase-2020-spectrogram`; raw wavs for the audio-backbone probe live under `data/dcase-2020/`).
  Layout is the standard
  `<cls>/{train,test}/{normal,abnormal}/*.png` split (all machine IDs — `id_00/id_02/id_04/id_06`
  — pooled into both splits; the ID is only in the filename). The special-case exists because the
  `train` split may still contain `abnormal` samples, which are **filtered out at load time** so
  training stays normal-only (UAD). (Legacy note: an older layout used machine-ID phases `id_00`
  train / `id_02` test — that is no longer the case.)
- Encoder is `resnet34` (stock image MambaAD uses `wide_resnet50_2`); local weights at
  `model/pretrain/resnet34-43635321.pth`.
- For MIMII, `meta.json`'s `mask_path` is empty — image-level labels only, no pixel masks.

## Training stability & resume (hard-won, July 2026)

- **Divergence history:** at lr 5e-3 with only a late step decay, student weights (`mff_oce`
  convs, `A_logs`) grew until overflow → NaN (surfaced later as sklearn "Input contains NaN" at
  test). The non-finite guard now catches this, but **`ckpt.pth` from a diverged run is
  unusable** — resume from a finite periodic `net_<epoch>.pth` snapshot instead:
  ```bash
  python make_resume_ckpt.py --run_dir runs/<run> --epoch <E> --lr 5e-4
  python run.py -c <cfg> -m train trainer.resume_dir=<run> \
      model.kwargs.checkpoint_path=ckpt_resume<E>.pth
  ```
- **AUROC peaks early:** stable runs peak around epoch 50 then decline; there is **no
  best-checkpoint selection** — final-epoch numbers understate the model. Compare runs at their
  peaks (plot first), not at the final epoch.

## Notes

- `NOTE.md` is the running, dated research log — check the latest entries for current direction
  and known issues, and append there when direction changes.
- `docs/ABLATION_SUMMARY.md` is the **consolidated results doc** (all ablation fronts in one place:
  lever table, gap decomposition, per-class ceilings, conclusions). Read this first for the big
  picture; `NOTE.md` for the dated blow-by-blow.
- `docs/stgram-mambaad/PLAN.md` is the design doc for the STgram→MambaAD pipeline.
- Tooling guidance is lazy-loaded: `docs/CLAUDE.md` (plotters, re-evaluation) and
  `diagnostics/CLAUDE.md` (frozen-feature probes and their verdicts) load when you work in
  those directories.
- **Keep this file current:** when the loss, configs, dataset roots, or workflow change, update
  the matching section here in the same commit.
