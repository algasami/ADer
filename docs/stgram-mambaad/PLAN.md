# Plan: STgram → image → MambaAD reconstruction

This document is the working plan for future agents. It describes how to feed **STgram** (from the
STgram-MFN baseline) into MambaAD as a reconstruction target, replacing the current plain log-Mel
spectrogram front-end.

## Context

This research fork runs MambaAD on MIMII audio by treating spectrograms as images. Today the input is a
plain log-Mel spectrogram PNG (`data/dcase-2020-spectrogram/`). The goal of this work is to swap that
front-end for **STgram** — the two-channel representation from the STgram-MFN baseline — so MambaAD
reconstructs a richer input that pairs the hand-crafted log-Mel spectrogram (**Sgram**) with a **learned**
temporal-gram (**Tgram**). This tests whether the learned temporal branch improves the anomaly-detection
signal over the current spectrogram-only pipeline, and gives a direct comparison against the STgram-MFN
and PaDiM baselines already in the repo.

### What STgram actually is (key facts that shape the plan)
- STgram is a **2-channel** tensor `(2, 128, 313)` assembled in `STgram-MFN/net.py:183-190`:
  `torch.cat((Sgram, Tgram), dim=1)`.
  - **Sgram** = log-Mel spectrogram from `Wave2Mel` (`STgram-MFN/utils.py:117`, torchaudio
    `MelSpectrogram` + `AmplitudeToDB`). Shape `(128, 313)`.
  - **Tgram** = output of `TgramNet` (`STgram-MFN/net.py:147`), a **learned** `Conv1d(1→128, k=1024,
    s=512)` over the raw waveform followed by 3 conv blocks. Shape `(128, 313)`. Because it is learned,
    it is only meaningful with **trained weights**.
- Trained weights already exist:
  `STgram-MFN/runs/STgram-MFN(m=0.7,s=30)/model/best_checkpoint.pth.tar`. The state dict is under key
  `['model']` (see `STgram-MFN/run.py:63-65`); `TgramNet` params are the `tgramnet.*` keys.
- Both grams are `(128, 313)` for a 10 s / 16 kHz clip (160000 samples, hop 512) — **the same dimensions
  as the current spectrogram PNGs**, so downstream image sizing is unchanged.
- Audio hyperparameters (must match the checkpoint): `sr=16000, n_fft=1024, win_length=1024,
  hop_length=512, n_mels=128, power=2.0, secs=10` (`STgram-MFN/config.yaml:33-40`).

### Chosen encoding
- **3-channel PNG**, channels = `[Sgram, Tgram, <third>]`, each independently min-max scaled to 0–255.
- The **third channel is a config/CLI flag** for ablation: `sgram` (duplicate Sgram, default) or `delta`
  (1st temporal derivative of Sgram, reusing `librosa.feature.delta` as in the toy notebook).
- This keeps the **entire existing MambaAD engine unchanged** (ResNet34 3-ch teacher, ImageNet
  normalization, PIL-RGB loader, meta-gen); only a new data root + config are introduced.

## Approach (pipeline)

```
raw wav ──► Wave2Mel ─────► Sgram (128×313) ─┐
        └─► TgramNet(ckpt) ► Tgram (128×313) ─┼─► stack+minmax+flip ─► 3-ch PNG
                                    third ────┘        │
                                                       ▼
                          data/dcase-2020-stgram/<machine>/<split>/<label>/<name>.png
                                                       │
                          mimii-stgram.py ► meta.json ─┤
                                                       ▼
                    run.py -c mambaad_mimii_stgram.py  (unchanged engine)
```

## Steps

### 1. STgram image generator — `src/gen_stgram_images.py` (new)
A standalone script (mirrors the `src/` experiment track). Inserts the submodule on `sys.path` and
imports its front-end so Sgram/Tgram come from the **same** model the checkpoint was trained with.

- `sys.path.insert(0, 'STgram-MFN')`; `from net import TgramNet`; `from utils import Wave2Mel`.
- Load Tgram weights by filtering the checkpoint:
  ```python
  state = torch.load(CKPT, map_location='cpu')['model']
  tg_state = {k[len('tgramnet.'):]: v for k, v in state.items() if k.startswith('tgramnet.')}
  tgramnet = TgramNet(mel_bins=128, win_len=1024, hop_len=512)
  tgramnet.load_state_dict(tg_state); tgramnet.eval().cuda()
  ```
  (Filtering the `tgramnet.*` subset avoids needing `num_classes` / the classifier head.)
- Per clip: `y = librosa.load(path, sr=16000, mono=True)[0]`; **force-crop to exactly 160000 samples**,
  matching STgram-MFN's dataloader verbatim (`STgram-MFN/dataset.py:28`: `x = x[: sr*secs]`). All
  DCASE-2020 clips are ≥ 10 s, so this yields the 313 frames that `TgramNet`'s hardcoded
  `nn.LayerNorm(313)` (`net.py:155`) requires. Then:
  - `sgram = wav2mel(torch.from_numpy(y).float()).cpu().numpy()` → `(128, 313)`
  - `tg = tgramnet(x.view(1,1,-1).cuda())[0].cpu().numpy()` → `(128, 313)` under `torch.no_grad()`
  - `ch2 = sgram` (flag `sgram`) or `librosa.feature.delta(sgram)` (flag `delta`)
  - `img = np.flip(np.stack([scale_minmax(sgram), scale_minmax(tg), scale_minmax(ch2)], -1).astype(uint8), axis=0)`
  - `cv2.imwrite(out_path, img)`
- Reuse `scale_minmax` and the vertical `np.flip(axis=0)` convention from
  `src/notebooks/MIMII_Toy_spectrogram_converter.ipynb` so orientation matches the existing pipeline.
- Walk the **raw wav tree** and mirror the toy notebook's mapping into the new output root
  (`<machine> ∈ {fan,pump,slider,valve}`, `<split> ∈ {train,test}`, label `anomaly→abnormal`), using the
  same filename regex `^{label}_(.*)\.wav$` to derive the output name. First **verify the raw wav layout**
  under `data/dcase-2020/` (the toy notebook referenced `tmp/dcase-2020/data_<m>/<m>/<split>/{normal,anomaly}`)
  and set `SRC_ROOT` accordingly.
- Output root: **`data/dcase-2020-stgram/`**, structure identical to `data/dcase-2020-spectrogram/`:
  `<machine>/<split>/<normal|abnormal>/<name>.png`.
- Expose `--third {sgram,delta}` and `--out-root` so the two ablations write to separate roots
  (e.g. `data/dcase-2020-stgram` vs `data/dcase-2020-stgram-delta`).

### 2. Metadata generator — `data/gen_benchmark/mimii-stgram.py` (new)
Copy `data/gen_benchmark/mimii-toy.py` verbatim and change only the `__main__` root to
`data/dcase-2020-stgram`. Schema is unchanged (`{train,test}[cls] = [{img_path, mask_path:'',
cls_name, specie_name, anomaly}]`; `CLSNAMES_2D = ['fan','pump','slider','valve']`). Produces
`data/dcase-2020-stgram/meta.json`.

### 3. Register the new root — `data/ad_dataset.py:69`
Add `'dcase-2020-stgram'` (and `'dcase-2020-stgram-delta'` if ablating) to the existing dcase `elif`
list. No other reader change: the branch already selects `train`/`test` from meta and applies the
train-time normal-only filter. `name` is derived as `self.root.split('/')[-1]` (line 57), so matching the
root's basename is all that's required.

### 4. New MambaAD config — `MambaAD/configs/mambaad/mambaad_mimii_stgram.py` (new)
Copy `mambaad_mimii_toy.py` and change **only** `self.data.root = 'data/dcase-2020-stgram'`. Everything
else stays: `size=256`, `fvcore_c=3`, teacher `timm_resnet34` (`out_indices=[1,2,3]`), student
`mambaad` (hilbert, 8 dir), single `L2Loss` pixel term (`lam=5.0`), `MAMBAADTrainer`, ImageNet
normalize, sample-level metrics. (For the delta ablation, a second config pointing at the delta root.)

### 5. Generate data, train, test
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/anaconda3/envs/mamba-ad/lib   # Mamba CUDA (NOTE.md)
conda activate mamba-ad
python src/gen_stgram_images.py --third sgram --out-root data/dcase-2020-stgram
python data/gen_benchmark/mimii-stgram.py
CUDA_VISIBLE_DEVICES=0 python run.py -c MambaAD/configs/mambaad/mambaad_mimii_stgram.py -m train
CUDA_VISIBLE_DEVICES=0 python run.py -c MambaAD/configs/mambaad/mambaad_mimii_stgram.py -m test
```

## Critical files
- **New:** `src/gen_stgram_images.py`, `data/gen_benchmark/mimii-stgram.py`,
  `MambaAD/configs/mambaad/mambaad_mimii_stgram.py`, this document.
- **Edit (1 line):** `data/ad_dataset.py:69` (add root name to the dcase `elif`).
- **Read/reuse:** `STgram-MFN/net.py` (`TgramNet`, `get_tgram`), `STgram-MFN/utils.py` (`Wave2Mel`),
  `STgram-MFN/config.yaml` (hyperparams), `src/notebooks/MIMII_Toy_spectrogram_converter.ipynb`
  (`scale_minmax`, flip, dir-walk/regex), `data/gen_benchmark/mimii-toy.py`,
  `MambaAD/configs/mambaad/mambaad_mimii_toy.py`.

## Verification
1. **Weight load:** filtered `tgramnet.*` load must be non-strict-clean (all keys matched). Print a
   parameter norm to confirm non-random weights.
2. **Image sanity:** dump a handful of PNGs; assert shape `(128, 313, 3)`, `uint8`, and that the **Tgram
   channel is non-degenerate** (has structure, not flat) — the main failure mode is a mis-loaded or
   random TgramNet.
3. **Counts match baseline:** per-class train/test PNG counts should equal the existing
   `data/dcase-2020-spectrogram/` (e.g. fan train=3675 normal, test=1875) since it's the same wav set.
4. **Meta:** `meta.json` has `train`/`test` → `fan/pump/slider/valve` keys; spot-check an entry.
5. **Dataloader smoke test:** instantiate `DefaultAD` on the new root, pull one batch, confirm tensor
   `(B,3,256,256)` and that train split is normal-only.
6. **End-to-end train/test** on one class first (`data.cls_names=fan`) to fail fast, then full multi-class.
   Compare `mAUROC_sp_max / mAP_sp_max / mF1_max_sp_max` against (a) the spectrogram MambaAD baseline and
   (b) the STgram-MFN and PaDiM baselines; log results in `NOTE.md`.

## Caveats / risks
- **Hardcoded 10 s:** `TgramNet`'s `LayerNorm(313)` forces exactly 160000 samples → 313 frames. Force-
  crop every clip (`y[:160000]`) exactly as STgram-MFN does (`dataset.py:28`); all DCASE-2020 clips are
  ≥ 10 s (ToyCar is 11 s, cropped down; the rest are 10 s). **ToyCar/ToyConveyor are excluded for
  baseline comparability, NOT a TgramNet limitation** — the ADer meta-gen fixes
  `CLSNAMES_2D=['fan','pump','slider','valve']` and the existing spectrogram baseline already dropped
  them. They would run fine; add their `CATS` entries + point meta-gen at them to include them.
- **Normalization mismatch (pre-existing):** ImageNet RGB mean/std are applied to non-image channels
  (same caveat as today's pipeline). Acceptable for a first pass; a spectrogram-tuned norm is a possible
  follow-up.
- **cv2 vs PIL channel order:** `cv2.imwrite` stores array channel 0 as B and PIL loads as RGB, so the
  R/B channels swap on read. This is consistent across all images (cosmetic to the network), but note it
  when visualizing.
- **Ablation is first-class:** the `--third {sgram,delta}` flag + parallel roots/configs let steps 1–5 be
  re-run for the delta variant without touching the engine.
