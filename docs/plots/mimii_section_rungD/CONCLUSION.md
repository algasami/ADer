# Rung D — joint distill + classify ("MambaAD + labels")

**Question.** Rung C trained the Mamba student to classify sections *instead of* reconstructing
teacher features. Rung D does BOTH at once — the faithful "MambaAD + labels":

    loss = CosLoss(feats_t, feats_s)  +  lambda_cls * CE(ArcFace(GAP(feats_s)), section)

so the student must simultaneously reconstruct the frozen teacher (the native UAD objective)
and discriminate the 23 sections. At test time three score families are available and fusible:
the native reconstruction residual (`recon_spmax/spmean`, `Evaluator.cal_anomaly_map`), the
classification readouts (`neg_cos`, `logit_nll`, `maha_embed`), and `fusion_*` = z-normed
recon_spmean + z-normed classification score.

Trainer: `diagnostics/section_joint_rungD.py`. Run: 50 epochs, sub=2, **lambda_cls=1.0**, lr
1e-3, 0 divergence skips. Figures: `auroc_vs_epoch/`, `readout_bars/`.

## Headline (mean image-level AUROC %, best epoch)

| family | readout | best | vs |
|---|---|---|---|
| classification | **`maha_embed`** | **84.0** | ≈ Rung C 84.4 (**−0.4**, noise) |
| classification | `neg_cos` | 80.3 | |
| **reconstruction** | `recon_spmax` | **57.3** | native recon-only ~72 (**−15**) |
| reconstruction | `recon_spmean` | 56.5 | |
| **fusion** | `fusion_maha` | 74.1 | best single 84.0 (**−10**) |
| fusion | `fusion_negcos` | 73.2 | |

## Findings — a clean negative result

1. **Joint training buys nothing over labels-only.** Rung D's best (`maha_embed` 84.0) is
   within noise of Rung C's 84.4. Adding the reconstruction objective to the section classifier
   did not improve the discriminative score at all.

2. **The classification objective *degrades* the reconstruction anomaly score.** `recon_spmax/
   spmean` collapse to ~57 — **below** the native recon-only baseline (~72) *and* below the
   frozen-teacher Maha anchor (71.8). And this is not a failure to reconstruct: the cos-loss
   trains down fine (0.15). The student learns to reconstruct the teacher for *both* normal and
   anomalous clips, so the residual stops discriminating. The discriminative objective is
   **antagonistic** to the reconstruction-residual UAD premise ("anomalies reconstruct badly").

3. **Fusion *hurts* — the objectives are not complementary.** `fusion_*` (73–74) sits far below
   the classification readout alone (84): the recon score is near-chance, so z-fusing it only
   injects noise. Adding a weak generative score to a strong discriminative one is strictly bad
   here.

The `readout_bars` figure shows all three families at a glance: reconstruction ≈ chance,
classification carries everything, fusion lands *between* them — dragged down, not lifted.

## Placement on the ladder (mean AUROC, best readout)

```
native recon-only (no labels)   ~72.0   ── the project's original MambaAD score
Rung A frozen + cls              80.0
Rung D joint (labels+recon)      84.0    ← = Rung C; recon objective is dead weight
Rung C Mamba + cls (labels only) 84.4
Rung B fine-tune ResNet          85.9    ← best so far
STgram-MFN                       90.75
```

**Conclusion: "MambaAD + labels" as a joint multi-task is NOT the payoff.** Once the
classification objective is present, the reconstruction distillation is dead weight (at best
redundant, at worst — via fusion — harmful). The clean takeaway across C+D: it is the
*classification objective on the Mamba student* that helps; the reconstruction half of MambaAD
contributes nothing to it and its residual score is actively spoiled by joint training.

## Caveats
- **lambda_cls=1.0 only.** With CE ≈ 5 vs cos ≈ 0.3 early, the classifier dominates the joint
  loss, so "recon collapses" is partly "recon is under-weighted." A `lambda_cls` sweep (or a
  recon-weighted schedule) might preserve the recon score — but since labels-only already
  matches Rung D, a better recon score would only help if fusion then turned positive.
- **No in-harness lambda_cls=0 control.** The ~72 recon-only baseline is the project's separate
  full MambaAD run, not this harness/recon_size=64 setup. The rigorous control — run this exact
  script with `--lambda_cls 0` — would pin the "joint training hurt the recon score" claim
  precisely. Recommended before publishing the magnitude of the recon collapse.
- Single seed; from-scratch student.

## Next
The joint multi-task is answered (negative). The remaining ladder move is **Rung E — B+C
combined**: fine-tune the teacher *and* the Mamba student jointly under the section loss (does
encoder adaptation, the biggest lever, stack with the Mamba decoder?). Also open: a `lambda_cls`
sweep here for completeness, and a cross-rung fused readout (C wins slider, B wins the rest).
