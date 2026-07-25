---
name: mimii-data-pipeline
description: Regenerate the MIMII spectrogram-image datasets (log-Mel / STgram / STgram-delta PNGs) and their meta.json files. Use when creating a new input representation, re-running the generators after a scaling or crop change, or when a dataset root is missing/stale.
---

# MIMII data pipeline

Two steps, **both required** before training. Skipping step 2 leaves a dataset root the
`DefaultAD` loader cannot read.

## Step 1 — generate spectrogram PNGs from raw wavs

`src/` scripts. Clips are force-cropped to 10 s / 313 frames so all generators match.

- `src/gen_mel_images.py` — single-channel log-Mel images.
- `src/gen_stgram_images.py` — 3-channel STgram images (Sgram + frozen-TgramNet Tgram + third
  channel selected by flag: `sgram` or `delta`). Loads TgramNet weights from the STgram-MFN
  checkpoint. Scaling is min-max per channel — see the script for the current global-vs-local
  convention (**per-audio local scaling caused training issues and was fixed**; the PNGs on disk
  were made by the fixed global-percentile generator).

## Step 2 — generate `meta.json` at each dataset root

```bash
python data/gen_benchmark/mimii-toy.py      # -> data/dcase-2020-spectrogram/meta.json
python data/gen_benchmark/mimii-stgram.py   # -> data/dcase-2020-stgram*/meta.json
python data/gen_benchmark/mvtec.py          # -> data/mvtec/meta.json  (visa.py, coco.py likewise)
```

`meta.json` schema (shared across datasets): per-split → per-class list of
`{img_path, mask_path, cls_name, specie_name, anomaly (0/1)}`. For MIMII `mask_path` is empty —
image-level labels only, no pixel masks.

## Layout the generators must produce

`<cls>/{train,test}/{normal,abnormal}/*.png`, with all machine IDs (`id_00/id_02/id_04/id_06`)
pooled into both splits — the ID appears only in the filename. The `train` split may still
contain `abnormal` samples; they are filtered out at load time (`data/ad_dataset.py`) so training
stays normal-only (UAD).
