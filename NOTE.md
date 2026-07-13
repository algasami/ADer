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
