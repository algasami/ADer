# diagnostics/ — frozen-feature probes

Each probe bypasses the trained decoder and scores features directly (Mahalanobis / kNN /
PatchCore-style), to locate the bottleneck. Their verdicts drive the whole ablation story —
see `docs/ABLATION_SUMMARY.md`.

- `frozen_encoder_probe.py` — frozen ResNet34 teacher features alone. **Verdict: the encoder is
  exonerated** (frozen beats the trained MambaAD peak on all three representations).
- `student_feature_probe.py` — same probe on the *trained Mamba student's* GAP features, to
  bisect readout vs. decoder. **Verdict: it's the readout** — student+Maha ≈ teacher+Maha, so the
  decoder preserves the manifold and the cosine-residual/sp_max head is what loses 6–10 pts.
- `audio_backbone_probe.py` — swaps the frozen ImageNet ResNet34 for an audio-pretrained backbone
  (**AST**, **CNN14**) run natively on the raw MIMII wavs under `data/dcase-2020/`.
  **Verdict: AST +5, CNN14 −2.6** — "audio-pretrained" is not automatically better; AST is the
  best frozen encoder found. AST was never trained end-to-end **by choice** (the teacher-slot port
  was judged not worth the time), so its ceiling is untested, not disproven.
  Outputs `runs/audio_probe/{ast,cnn14}/`. Extra deps in env `mamba-ad`: `panns_inference`,
  `torchlibrosa`, `transformers`.
