# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this fork is

A research fork of **ADer** (multi-class unsupervised anomaly detection) whose goal is to run
**MambaAD** on the **MIMII** audio dataset (DCASE-2020) by treating **spectrograms as images**:
MIMII recordings are pre-converted to 3-channel PNGs which flow through the otherwise-unchanged
image AD engine. Baselines for comparison: PaDiM (anomalib track) and STgram-MFN (submodule).

The active research question is which input representation works best. Three ablation fronts:
- **log-Mel toy**: (log-Mel, delta, delta-delta) → `data/dcase-2020-three-channel`
- **STgram-Sgram**: (Sgram, learned Tgram, Sgram) → `data/dcase-2020-stgram`
- **STgram-delta**: (Sgram, learned Tgram, delta) → `data/dcase-2020-stgram-delta`

The Tgram channel comes from a **frozen TgramNet** (weights from the STgram-MFN baseline
checkpoint). Full design rationale: `docs/stgram-mambaad/PLAN.md`.

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

Mamba deps: `mamba_ssm`, `causal_conv1d`, `triton`, `numpy-hilbert-curve`, `pyzorder`.
Audio: `librosa`. Baseline track: `anomalib`, `faiss-gpu`.

## Common commands

```bash
# Train / test MambaAD on MIMII (single GPU); swap the config for stgram / stgram_delta
CUDA_VISIBLE_DEVICES=0 python run.py -c MambaAD/configs/mambaad/mambaad_mimii_toy.py -m train
CUDA_VISIBLE_DEVICES=0 python run.py -c MambaAD/configs/mambaad/mambaad_mimii_stgram.py -m test

# Config overrides = trailing path.key=value args (parsed by run.py's REMAINDER opts)
python run.py -c <cfg> -m train data.cls_names=pump trainer.checkpoint=runs/my_run

# Multi-GPU (DDP)
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 \
  --master_addr=127.0.0.1 --master_port=12315 --use_env run.py -c <cfg> -m train
```

- MIMII configs: `mambaad_mimii_toy.py` (log-Mel), `mambaad_mimii_stgram.py`,
  `mambaad_mimii_stgram_delta.py`; scan-type ablations: `..._toy_{scan,zigzag,zorder,sweep}.py`.
- Visualization: add `vis=True vis_dir=<dir>` to a test run.
- Single-class sweep helper: `runs_single_class.py` (`-d <dataset> -c <cfg> -n <num_procs> -m <mode> -g <gpu>`).
- PaDiM baseline (anomalib track, *not* `run.py`): `cd src && python padim_baseline.py`
  (MVTec) or `python padim_baseline_mimii.py` → writes `padim_performance.csv`.

## Data pipeline (two steps, both required before training)

**Step 1 — generate spectrogram PNGs** from raw wavs (`src/` scripts; clips force-cropped to
10 s / 313 frames so all generators match):
- `src/gen_mel_images.py` — single-channel log-Mel images.
- `src/gen_stgram_images.py` — 3-channel STgram images (Sgram + frozen-TgramNet Tgram + third
  channel selected by flag: `sgram` or `delta`). Loads TgramNet weights from the STgram-MFN
  checkpoint. Scaling is min-max per channel — see the script for the current global-vs-local
  convention (per-audio local scaling caused training issues and was fixed).
- Legacy notebooks live in `src/notebooks/` (`MIMII_spectrogram_converter.ipynb`,
  `MIMII_Toy_spectrogram_converter.ipynb`).

**Step 2 — generate `meta.json`** at each dataset root (`data/gen_benchmark/`):
```bash
python data/gen_benchmark/mimii-toy.py      # -> data/dcase-2020-three-channel/meta.json
python data/gen_benchmark/mimii-stgram.py   # -> data/dcase-2020-stgram*/meta.json
python data/gen_benchmark/mvtec.py          # -> data/mvtec/meta.json  (visa.py, coco.py likewise)
```

MIMII classes: `fan, pump, slider, valve, ToyCar, ToyConveyor`.

## Architecture (the cross-file picture)

**Entry flow:** `run.py` → `get_cfg()` (`configs/__init__.py`) → `init_training` →
`init_checkpoint` → `get_trainer(cfg).run()`.

**Registry pattern** (`util/registry.py`): MODEL / DATA / TRAINER / LOSS / OPTIM / TRANSFORMS
registries populated by `@*.register_module` decorators, consumed by `get_*` factories in each
package's `__init__.py`. timm/torchvision models auto-register as `timm_<name>` / `tv_<name>`.

**Config = Python class inheritance, not YAML.** Configs subclass bases in `configs/__base__/`
(`cfg_common`, `cfg_dataset_default`, `cfg_model_mambaad`) and set `data.*`, `model_t`/`model_s`,
`loss.loss_terms`, `trainer.name`, `metrics`, etc.

**Trainer:** `trainer/_base_trainer.py` holds the DDP/AMP/logging/checkpoint loop; MambaAD uses
`MAMBAADTrainer` (`trainer/mambaad_trainer.py`) — frozen teacher (resnet34) → Mamba-decoder
student, trained with **`CosLoss` under log-term name `cos`**. The base trainer has a
**non-finite loss/grad guard** that skips bad steps so divergence can't silently corrupt weights.

> Gotcha: `update_log_term` silently no-ops when a config's logged term name doesn't match what
> the trainer computes — a mismatched name means the loss is never logged, with no error. Keep
> config `loss_terms`/`log_terms` names and the trainer in sync (currently both `cos`).

**Model:** `model/mambaad.py` — Mamba decoder with Hilbert-curve scanning
(`scan_type='hilbert'`, `num_direction=8`).

**Dataset:** `data/ad_dataset.py` `DefaultAD` reads `meta.json` and yields
`{img, img_mask, cls_name, anomaly, img_path}`.

**Metrics:** two pooling families over the anomaly map (`util/metric.py`) — `*_sp_max` (max over
pixels) and `*_sp_mean` (mean). Current MIMII configs record both (6 metrics, 42-column
`metric.txt`); older runs recorded sp_max only (21 columns).

## Audio-specific customizations vs. stock ADer

- Dataset roots `dcase-2020-spectrogram` / `dcase-2020-three-channel` / `dcase-2020-stgram` /
  `dcase-2020-stgram-delta` are special-cased in `data/ad_dataset.py:69` — splits are
  **machine-ID phases** `id_00` (train) / `id_02` (test), not `train`/`test`.
- Encoder is `resnet34` (stock image MambaAD uses `wide_resnet50_2`); local weights at
  `model/pretrain/resnet34-43635321.pth`.
- `meta.json` schema is shared across datasets: per-split → per-class list of
  `{img_path, mask_path, cls_name, specie_name, anomaly (0/1)}`. For MIMII `mask_path` is empty
  (image-level labels only — no pixel masks).
- Baseline/ablation track in `src/`: PaDiM + AdaLN/FiLM (`src/padim_adaln/`), patched anomalib
  datamodules (`src/mvtecad_patched_*`, `src/mimii_anomalib_*`).

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

## Analysis & diagnostic tooling

- `docs/plot_mimii_val_metrics.py` — plots AUROC/AP/F1 vs. epoch from a run's `metric.txt`
  (presets per run group; `--family sp_max|sp_mean`). Auto-detects the 21- vs 42-column layout.
- `docs/reeval_sp_mean.py` (+ `docs/reeval_and_plot_sp_mean.sh`) — re-evaluates saved
  `net_<E>.pth` checkpoints to recover sp_mean metrics for runs that only recorded sp_max;
  writes `<run-dir>/metric_reeval.txt` in the native 42-column layout.
- `diagnostics/frozen_encoder_probe.py` — tests whether frozen ResNet34 teacher features alone
  separate normal/anomalous MIMII clips (Mahalanobis / kNN / PatchCore-style scorers), to decide
  whether the encoder or the downstream decoder is the bottleneck.

## Notes

- `NOTE.md` is the running, dated research log — check the latest entries for current direction
  and known issues, and append there when direction changes.
- `docs/stgram-mambaad/PLAN.md` is the design doc for the STgram→MambaAD pipeline.
- `runs/` holds checkpoints, logs, and outputs; plots land in `docs/plots/` and
  `docs/short-epoch-plots/`.
- **Keep this file current:** when the loss, configs, dataset roots, or workflow change, update
  the matching section here in the same commit.
