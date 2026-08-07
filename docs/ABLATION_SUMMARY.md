# MambaAD-on-MIMII — Ablation Campaign Summary

_As of 2026-07-25. Consolidates every ablation front run to date. Companion to `NOTE.md`
(dated log) and `docs/stgram-mambaad/PLAN.md` (design). Numbers are image-level AUROC (%),
averaged over the 6 MIMII classes unless noted._

---

## 1. TL;DR

The fork's original question — *"which spectrogram input representation makes MambaAD work best
on MIMII?"* — is answered, and the answer reframes the project:

- **No representation, decoder, scan curve, or training schedule closes the gap.** Every
  downstream lever converges on **~72 AUROC**. The differences between them are ≤1 pt (noise).
- **The one downstream lever that moves anything is the test-time readout:** swapping the native
  cosine-residual head for a Mahalanobis memory-bank readout on GAP features is a fixed **+6–8**
  (64.7 → 72.5). Everything else — scan geometry, schedule, distillation training itself — is flat.
- **The trained Mamba decoder adds ~nothing.** Frozen ResNet34 + Maha (71.8, *no training*)
  matches the fully-trained student (72.5). The decoder is effectively a no-op under a distance
  readout.
- **The feature extractor is a real but partial lever — measured at the frozen-feature level.**
  Swapping ImageNet ResNet34 for the **AST** audio transformer buys **+5** (76.8) *under an
  identical frozen Maha readout*. But an audio *CNN* (CNN14) is **−2.6** — so "audio-pretrained"
  is not automatically better; the AST architecture/pretraining is doing it. **Not yet
  demonstrated end-to-end:** no MambaAD run has been trained with AST as the teacher — that port
  was **deliberately skipped as a time/cost decision**, not because AST underperformed. This is
  the one *open* lever, not a settled one (see §5).
- **The dominant remaining gap is the learning objective, not the features.** Even the best frozen
  backbone (AST, 76.8) is **−14** below the audio-native STgram-MFN baseline (90.7). That 14 pts
  is the supervised machine-ID/ArcFace discrimination that no frozen-feature + unsupervised-Maha
  pipeline can reproduce.

**Gap decomposition to the SOTA baseline:**

```
 71.8  ImageNet ResNet34 + unsupervised Maha      (the MambaAD-track ceiling)
+ 5.0  → AST audio features                        (backbone lever, frozen-probe only)
+13.9  → STgram-MFN                                (supervised discriminative objective)
 90.7  STgram-MFN (SOTA audio baseline)
```

All four rows are **frozen-feature + distance-readout** numbers except the last. The +5.0 backbone
step is a probe result, not a trained-pipeline result — it says AST features are more separable
than ResNet34 features under the same scorer, not that a MambaAD run on AST would land at 76.8.

---

## 2. What was run

| Front | Variants | Config / tooling |
|---|---|---|
| Training schedule | e50 / e200 / e1000 / lowlr-lowwd | `mimii/{e50,e200,e1000,lowlr-lowwd}/*.py` |
| Input representation | log-Mel / STgram-Sgram / STgram-delta | `mimii/*/{log-Mel,stgram,stgram-delta}.py` |
| Decoder scan geometry | hilbert / zorder / scan / sweep / zigzag | `mimii/scan-type/*.py`, `docs/run_scan_ablation.sh` |
| Test-time scorer | cos-residual / {student,teacher}×{maha,knn} | `mimii/scorer/*.py`, `util/scorer.py`, `docs/run_scorer_ablation.sh` |
| Encoder / backbone | ResNet34(ImageNet) / CNN14 / AST | `diagnostics/{frozen_encoder,student_feature,audio_backbone}_probe.py` |

Baselines: **STgram-MFN** (audio SOTA, `STgram-MFN/`), **PaDiM** (anomalib track, `src/`).

---

## 3. The lever table — effect size of every ablation

_How many AUROC points does each lever move? This is the whole campaign in one table._

| Lever | Range explored | Avg AUROC effect | Verdict |
|---|---|---|---|
| **Training schedule** | e50 / e200 / e1000 / lowlr | e1000 peaks 62@ep50 then **decays** to 56; e200≈e50; lowlr underfits | **Non-lever.** Longer hurts (identical-shortcut); short + early-stop is all you get |
| **Scan geometry** | hilbert/zorder/scan/sweep/zigzag | 72.07 – 72.60, **spread 0.5 pt** | **Non-lever.** Decoder geometry is irrelevant |
| **Representation** | log-Mel / STgram-Sgram / STgram-delta | 71.8 / 68.4 / 66.6 (frozen maha_concat) | log-Mel **wins**; the STgram gambit *hurts* on average |
| **Decoder training itself** | frozen teacher vs trained student (Maha) | 71.8 → 72.5 (**+0.7**) | **~No-op.** Distillation buys almost nothing |
| **Scorer / readout** | cos-residual → maha/knn | 64.7 → **72.5** (**+7.8**) | **The one real downstream lever** |
| **Backbone (features)** | ImageNet RN34 → AST | 71.8 → **76.8** (**+5.0**, frozen probe) | **Real lever — the largest feature-level effect measured**; CNN14 shows it's AST-specific, not "audio" per se. End-to-end training not attempted (skipped by choice, §5) → **open, not refuted** |
| **Learning objective** | UAD distance vs STgram-MFN supervised | 76.8 → **90.7** (**+13.9**) | **The dominant untapped lever** |

---

## 4. Downstream bisection — it was the readout, not the decoder

The e50 × log-Mel run re-scored at every saved epoch with each readout (`docs/run_scorer_ablation.sh`):

| scorer | start (ep2) | peak | final (ep50) |
|---|---|---|---|
| cos-residual (native) | 59.6 | **64.7** @ep10 | 62.0 |
| student-maha | 68.3 | **72.5** @ep40 | 72.3 |
| student-knn | 67.5 | **70.7** @ep48 | 70.7 |
| teacher-maha (frozen) | 71.8 | flat | 71.8 |
| teacher-knn (frozen) | 71.5 | flat | 71.5 |

- All three distance readouts beat cos-residual by **+6–8 pts** — the `1−cos` residual + sp_max
  pooling was the bottleneck, not the features.
- cos-residual's "AUROC peaks early then decays" is a **readout artifact** — the student-maha curve
  rises monotonically; only the residual head degrades as the student learns to reconstruct anomalies.
- student-maha (72.5) ≈ teacher-maha (71.8): the decoder adds a hair, not zero, but effectively the
  frozen teacher + Maha is the whole story.

---

## 5. The backbone probe (2026-07-25)

Audio-native probe (`diagnostics/audio_backbone_probe.py`): raw MIMII wavs → each backbone's own
front-end → **the same Maha/kNN scorers on the same split** as the ResNet34 probe. Best scorer (Maha):

| backbone (frozen) | fan | pump | slider | valve | ToyCar | ToyConv | **mean** | Δ vs RN34 |
|---|---|---|---|---|---|---|---|---|
| ResNet34 · ImageNet · image | 58.4 | 72.1 | 90.8 | 70.0 | 75.7 | 63.8 | **71.8** | — |
| CNN14 · AudioSet · audio | 50.4 | 67.8 | 90.3 | 71.9 | 76.0 | 58.7 | **69.2** | **−2.6** |
| **AST · AudioSet · audio** | 56.7 | 82.3 | 93.6 | 78.9 | 82.7 | 66.4 | **76.8** | **+5.0** |
| STgram-MFN (supervised) | 87.1 | 90.9 | 98.9 | 98.6 | 94.7 | 74.3 | **90.7** | +18.9 |

- **AST +5.0** → the ImageNet backbone *was* a real ceiling. Biggest wins: pump +10, valve +9, ToyCar +7.
- **CNN14 −2.6** → an audio CNN is *worse* than an image CNN here. "Audio-pretrained" is not the
  magic; the transformer + AudioSet regime is. This rules out the naive "swap to any audio net" plan.
- **AST still −14 below STgram-MFN** → features recover only ~5 of the ~19-pt gap.

### Scope of this result — read before quoting the +5

Unlike the scan and scorer ablations, this front has **no per-epoch curves, because nothing is
trained**: `audio_backbone_probe.py` runs each backbone `.eval()` / `requires_grad=False` under
`no_grad`, then fits Maha/kNN on the train-normal bank. There are no checkpoints to sweep. An epoch
axis would also have been uninformative — the scorer ablation showed frozen-teacher scorers are
checkpoint-independent by construction (`static_fit`, flat lines across all 25 epochs), and that
"AUROC peaks early" is a *cos-residual readout artifact*, not a feature property. The epoch axis
measures the decoder and the readout, both of which this probe deliberately holds out.

Consequences for what the +5 does and does not establish:

- **Established:** AST features are more separable than ResNet34 features under an identical
  readout, split, and scorer code (imported verbatim from `frozen_encoder_probe.py`), so the
  backbone is a genuinely single variable. Maha was already the best known readout, so no backbone
  is being scored through a handicapped lens.
- **Not established:** that a *trained* MambaAD on AST features reaches 76.8. The extrapolation
  rests on the decoder-gap bisection (trained student + Maha 72.5 lands within ~1 pt of frozen
  teacher + Maha 71.8), which was measured on ResNet34 and assumed to transfer.
- **Why it stayed a probe — a scheduling decision, not a verdict.** Making AST a teacher is a port,
  not an ablation flag: MambaAD's teacher is `features_only=True, out_indices=[1,2,3]` (a three-level
  *spatial* pyramid the student distills), while AST emits one global vector per clip and the winning
  tap (`ast_meanpatch`) discards the time–frequency layout entirely, so it means reshaping patch
  tokens back to a grid at three depths and rebuilding the fusion channel dims. That cost was judged
  not worth the time at the point the campaign wound down. **Nothing here is evidence against AST
  end-to-end** — it is simply unmeasured, and remains the most promising untried encoder.
- **The under-explored axis here is the tap, not epochs.** Within AST alone: pooler 73.4 vs
  mean-patch 76.8 — a 3.3-pt spread from tap choice, comparable to the +5.0 headline. Only 2 taps
  were tried for AST and 1 for CNN14. Sweeping AST hidden states by layer is the cheap, informative
  follow-up if this front is revisited.

---

## 6. Per-class ceiling vs STgram-MFN — the transient story

Best frozen number per class across **all** probes (RN34 on log-Mel/STgram/delta images incl.
shallow taps, plus AST/CNN14 on raw audio) vs the supervised baseline:

| class | best frozen-track | source | STgram-MFN | gap | character |
|---|---|---|---|---|---|
| slider | 93.6 | AST | 98.9 | −5 | sustained slide — closest |
| ToyConveyor | 66.4 | AST | 74.3 | −8 | (baseline also weak here) |
| pump | 82.3 | AST | 90.9 | −9 | quasi-stationary |
| ToyCar | 82.7 | AST | 94.7 | −12 | mixed |
| fan | 71.4 | RN34 / STgram-img, knn_layer1 | 87.1 | −16 | stationary broadband |
| valve | 78.9 | AST | 98.6 | −20 | impulsive clicks — biggest gap |

Even with **per-class oracle** selection of backbone + representation + tap, the frozen ceiling
averages **79.2**, still **−11.5** below STgram-MFN (90.7). Notes:

- `fan` only reaches its 71.4 with a *specific* pairing — RN34 on the STgram image at a **shallow
  tap** (`knn_layer1`); the deep/concat and audio embeddings collapse it to 50–58, and the trained
  decoder loses it entirely. So fan's signal survives in frozen features but is fragile and
  tap-dependent — no single global readout captures it.
- `valve` (impulsive) is the **largest** gap (−20). AST's temporal attention helps (+9 over RN34)
  but can't match the supervised model; consistent with a delta (transient) channel specifically
  lifting valve. The two extremes — valve (transient) and fan (stationary broadband) — are exactly
  where frozen features + a single distance readout fall furthest short of the supervised objective.

---

## 7. Conclusions & open directions

**Settled (do not re-run):** schedule, scan geometry, representation, decoder architecture, and the
distillation objective are all flat. The MambaAD reconstruction-distillation pipeline on
spectrogram-images plateaus at ~72 (Maha readout on a trained run); the best *frozen-feature*
number on this track is ~77 (AST + Maha, no MambaAD training involved).

**The two levers with headroom, ranked:**

1. **Learning objective (+14 of the remaining gap).** The frozen-feature + unsupervised-Maha framing
   is the ceiling. The path to the baseline is a *discriminative* objective on MIMII — e.g.
   self-supervised machine-ID classification (the STgram-MFN recipe) or SSL fine-tuning of the AST
   backbone — rather than distance-from-normal on frozen features. This is a new training track, not
   an ablation of the existing one.
2. **Backbone (+5 at the frozen-feature level, open).** AST is the best frozen encoder found and the
   largest feature-level effect in the campaign; use it as the default feature source for any future
   probe/readout. CNN14 is a dead end. Converting the +5 into an end-to-end trained number requires
   porting AST into the MambaAD teacher slot (§5); that port was **not attempted — a deliberate
   time/cost call, not a result**. Treat this lever as untested with real headroom, not as settled.

**Clean negative result, ready to write up:** *reconstruction-distillation image-AD (MambaAD) on
spectrograms is bottlenecked first by its cosine-residual readout (a fixed +6–8 from a Maha head)
and then by the unsupervised frozen-feature framing; an audio-pretrained transformer backbone
recovers a further +5 in frozen-feature separability, but the pipeline still sits ~14 pts below an
audio-native supervised model, and that residual is the learning objective, not the encoder or the
input representation.*

### Provenance
- Lever tables: `docs/plots/mimii_scan/scan_summary.csv`, `docs/plots/mimii_scorer/scorer_summary.csv`.
- Backbone probe: `runs/audio_probe/{ast,cnn14}/auroc.csv`; frozen/student probes:
  `runs/{frozen,student}_probe/*/auroc.csv`.
- Baseline: `STgram-MFN/results/STgram-MFN(m=0.7,s=30)/result.csv`.
