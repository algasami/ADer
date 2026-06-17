# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this fork is

This is a research fork of **ADer** (a multi-class unsupervised anomaly-detection toolbox). The primary
goal is to run **MambaAD** on the **MIMII** audio dataset (DCASE-2020) by reconstructing **Mel
spectrograms**, and to establish baselines (e.g. PaDiM on MVTec-AD) for comparison.

The central idea: **audio is treated as images.** MIMII recordings are pre-converted to 1- or 3-channel
spectrogram PNGs, which then flow through the otherwise-unchanged image anomaly-detection engine. The
active research concern is how the audio/temporal vs. visual/spatial mismatch affects model behaviour, so
audio-specific deltas (below) matter more than they look.

### Three code surfaces coexist — keep them straight
- **Root ADer engine** (`run.py`, `configs/`, `model/`, `trainer/`, `data/`, `loss/`, `optim/`, `util/`) —
  the registry-driven training/eval framework. This is what `run.py` drives.
- **`MambaAD/`** — the forked MambaAD codebase. Notably it holds the MIMII configs under
  `MambaAD/configs/mambaad/` that the root `run.py` loads.
- **`src/`** — a *separate* anomalib-based experiment track (PaDiM baselines, AdaLN/FiLM variants,
  MIMII/MVTec anomalib datamodules, spectrogram-generation notebooks). **This is NOT wired into `run.py`**
  and uses anomalib/Lightning, not the ADer trainer.

## Environment

```bash
conda env create -f environment.yml   # creates env "mamba-ad" (Python 3.10)
conda activate mamba-ad
```

Mamba-specific deps: `mamba_ssm`, `causal_conv1d`, `triton`, `numpy-hilbert-curve`, `pyzorder`
(Hilbert-curve scan). Audio: `librosa`. Baseline track: `anomalib`, `faiss-gpu`.

Gotcha (from `NOTE.md`): on older servers, export the env's lib path before running, or Mamba CUDA
kernels fail to load:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/anaconda3/envs/mamba-ad/lib
```

## Common commands

```bash
# Train MambaAD on MIMII (single GPU)
CUDA_VISIBLE_DEVICES=0 python run.py -c MambaAD/configs/mambaad/mambaad_mimii_toy.py -m train

# Test (same config, -m test)
CUDA_VISIBLE_DEVICES=0 python run.py -c MambaAD/configs/mambaad/mambaad_mimii_toy.py -m test

# Multi-GPU (DDP)
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 \
  --master_addr=127.0.0.1 --master_port=12315 --use_env run.py -c <cfg> -m train
```

**Config overrides** are passed as trailing `path.key=value` args (parsed by `run.py`'s REMAINDER `opts`):

```bash
python run.py -c <cfg> -m train data.cls_names=grid trainer.checkpoint=runs/my_run
```

- Visualization: add `vis=True vis_dir=<dir>` to a test run.
- Single-class sweep helper: `runs_single_class.py` (`-d <dataset> -c <cfg> -n <num_procs> -m <mode> -g <gpu>`).
- PaDiM baseline (independent anomalib track, *not* `run.py`): `cd src && python padim_baseline.py`
  → writes `padim_performance.csv`.

## Generating dataset metadata (required before training)

Training reads a `meta.json` at each dataset root; generators live in `data/gen_benchmark/`. Run the
matching generator before training a new dataset:

```bash
python data/gen_benchmark/mimii-toy.py   # -> data/dcase-2020-three-channel/meta.json  (MIMII/DCASE)
python data/gen_benchmark/mvtec.py       # -> data/mvtec/meta.json
python data/gen_benchmark/visa.py        # -> data/visa/meta.json
python data/gen_benchmark/coco.py        # -> data/coco/meta_20_*.json
```

The spectrogram PNGs themselves are **pre-computed** via the notebooks in `src/`
(`MIMII_spectrogram_converter.ipynb`, `MIMII_Toy_spectrogram_converter.ipynb`) — they are not generated
at training time.

## Architecture (the cross-file picture)

**Entry flow:** `run.py` → `get_cfg()` (`configs/__init__.py`) → `init_training` → `init_checkpoint`
→ `get_trainer(cfg).run()`.

**Registry pattern** (`util/registry.py`): MODEL / DATA / TRAINER / LOSS / OPTIM / TRANSFORMS registries
are populated by `@*.register_module` decorators and consumed by `get_*` factories in each package's
`__init__.py`. timm/torchvision models auto-register as `timm_<name>` / `tv_<name>`.

**Config = Python class inheritance, not YAML.** A config such as
`MambaAD/configs/mambaad/mambaad_mimii_toy.py` subclasses bases in `configs/__base__/` (`cfg_common`,
`cfg_dataset_default`, `cfg_model_mambaad`) and sets `data.type`, `data.root`, `data.cls_names`,
`model_t`/`model_s`, `loss.loss_terms`, `trainer.name`, etc.

**Trainer:** `trainer/_base_trainer.py` holds the DDP/AMP/logging/checkpoint loop; per-method subclasses
specialize it. MambaAD uses `MAMBAADTrainer` (`trainer/mambaad_trainer.py`) — a teacher (resnet) → student
(Mamba decoder) reconstruction with pixel-level `L2Loss`.

**Model:** `model/mambaad.py` — Mamba decoder with Hilbert-curve scanning (`scan_type='hilbert'`,
`num_direction=8`).

**Dataset:** `data/ad_dataset.py` `DefaultAD` reads `meta.json` and yields dicts
`{img, img_mask, cls_name, anomaly, img_path}`.

## Audio-specific customizations vs. stock ADer (the research deltas)

- New dataset roots `dcase-2020-spectrogram` / `dcase-2020-three-channel` are special-cased in
  `data/ad_dataset.py`: splits are **machine-ID phases** `id_00` (train) / `id_02` (test), not
  `train`/`test`.
- Input is a **3-channel spectrogram** (three spectrogram representations stacked) rather than RGB; the
  encoder is swapped to `resnet34` (stock image MambaAD uses `wide_resnet50_2`).
- The `meta.json` schema is shared across datasets: per-split → per-class list of
  `{img_path, mask_path, cls_name, specie_name, anomaly (0/1)}`. For MIMII, `mask_path` is empty
  (image-level labels only — no pixel masks).
- Baseline/ablation track in `src/`: PaDiM + AdaLN/FiLM (`src/padim_adaln/`), plus patched anomalib
  datamodules that load multiple categories at once (`src/mvtecad_patched_*`, `src/mimii_anomalib_*`).

## Notes

- `NOTE.md` is the running, dated research log (e.g. AdaLN/FiLM swaps, PaDiM backbone sweeps) — check it
  for current direction and known issues.
- `runs/` holds checkpoints, logs, and outputs.
