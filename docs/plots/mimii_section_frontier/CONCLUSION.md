# ToyCar / ToyConveyor frontier analysis

**Question.** Rung E closed the labels/objective ladder at 86.9 mean AUROC (-3.9 to STgram-MFN
90.75), with the residual now concentrated in two classes: ToyCar (-9.2) and ToyConveyor (-5.7)
— fan/pump/slider/valve all land within ~1-2 pts of STgram. Why these two, and is there cheap
headroom left? Pure data analysis over already-completed rung results (`diagnostics/plot_section_frontier.py`)
— no new training.

## 1. The ladder isn't monotonic for ToyCar — Rung E actively trades it away

`ladder_by_class/plot.png`. Tracking each rung's own class-level score under its own winning
readout:

| rung | ToyCar | ToyConveyor |
|---|---|---|
| native (recon, no labels) | 61.6 | 61.5 |
| A (frozen + cls) | 83.2 | 63.0 |
| D (joint) | 82.9 | 68.8 |
| C (Mamba + cls) | 88.4 | 67.8 |
| **B (FT encoder)** | **92.8** | 72.4 |
| E (FT encoder + Mamba) | 85.5 | 68.6 |
| STgram-MFN | 94.7 | 74.3 |

ToyCar peaks at **Rung B (92.8, only -1.9 to STgram)** and then **drops 7.3 points at Rung E**
even though Rung E has the highest *overall* mean (86.9 > B's 85.9). This is the same trade the
Rung E CONCLUSION already flagged for slider/valve in reverse: the Mamba decoder's maha-friendly
geometry rescues slider (+11.5 over B) and valve (+2.3 over B) but actively hurts ToyCar (-7.3).
ToyConveyor never breaks 73 anywhere in the ladder — no rung variant helps it much past Rung B's
72.4, including the ones with the highest overall mean.

**So "the residual moved to ToyCar/ToyConveyor" is not one story — it's two:**
- **ToyCar is a *trade-off*, not a ceiling.** The representation already reaches 92.8 (Rung B);
  Rung E's architecture choice re-loses most of that gain in exchange for slider/valve. A
  per-class readout choice (see §3) recovers it for free.
- **ToyConveyor is a genuine ceiling.** No rung, readout, or architecture in the ladder pushes it
  past ~73 — closer to a representation/objective limit than a trade-off.

## 2. Part of the ToyConveyor ceiling is dataset-intrinsic — even STgram-MFN struggles with one ID

`stgram_per_id/plot.png`, from STgram-MFN's own per-machine-ID eval
(`STgram-MFN/results/STgram-MFN(m=0.7,s=30)/result.csv`) — **not** MambaAD-track output, the
supervised SOTA baseline's own numbers:

| class | id_01 | id_02 | id_03 | id_04 | class avg |
|---|---|---|---|---|---|
| ToyCar | 83.8 | 95.7 | 99.5 | 100.0 | 94.7 |
| ToyConveyor | 85.5 | 62.4 | 74.8 | — | 74.3 |

Even the audio-native, fully-supervised baseline is **not uniformly strong**: ToyConveyor's
class average is dragged down almost entirely by **id_02 (62.4)** — a single machine unit that's
hard for everyone, consistent with known DCASE-2020 domain-shift characteristics of this
particular ToyConveyor unit. ToyCar shows the same pattern in miniature: id_01 (83.8) is the
outlier, ids 02-04 are 95.7-100.

**Implication:** part of ToyConveyor's ~73 ceiling across the whole ladder is likely a hard
floor set by id_02, not something a better objective/encoder/decoder on the MambaAD track can
close — the MambaAD-track class average was never going to comfortably clear ~85 while pooling
in a unit STgram-MFN itself gets 62.4 on.

**CONFIRMED (2026-07-31):** the seed-repeat Rung B/E reruns (seeds 1+2, launched alongside this
analysis) now log a per-machine-ID breakdown (`runs/section_rung{B,E}/*_seed*/id_breakdown.csv`).
MambaAD-track shows the **same** id_02-specific weakness, in every single run:

| run | ToyConveyor id_01 | id_02 | id_03 |
|---|---|---|---|
| B seed1 (maha, ep8)  | 0.735 | **0.558** | 0.667 |
| B seed2 (maha, ep9)  | 0.724 | **0.579** | 0.700 |
| E seed1 (maha, ep27) | 0.800 | **0.609** | 0.679 |
| E seed2 (maha, ep21) | 0.790 | **0.570** | 0.791 |

id_02 is the worst ToyConveyor id in all 4 runs (both readouts, both rungs, both seeds) —
directly matching STgram-MFN's own weak point (id_02 = 62.4, its worst id too). This is not a
MambaAD-track-specific failure; it's the same hard machine unit tripping up every model tried on
it, supervised or not. The ToyConveyor ceiling is best read as substantially dataset-intrinsic.

## 3. Cross-rung oracle: a per-class-best selection (not a new model) gets to -1.9

`cross_rung_oracle/plot.png`. Taking, for each class, the best AUROC achieved by **any** rung
(A-E) under **any** of its readouts (a val-selectable choice, not a jointly-optimal model):

| class | Rung E (single model) | cross-rung oracle | source | STgram-MFN |
|---|---|---|---|---|
| fan | 84.5 | 84.5 | E maha | 87.1 |
| pump | 89.1 | 89.7 | E neg_cos | 90.9 |
| slider | 96.6 | 96.6 | E maha | 98.9 |
| valve | 96.9 | 96.9 | E maha | 98.6 |
| ToyCar | 85.5 | **92.8** | **B neg_cos** | 94.7 |
| ToyConveyor | 68.6 | **72.4** | **B neg_cos** | 74.3 |
| **mean** | **86.9** | **88.8** | | **90.75** |

The oracle is entirely driven by ToyCar/ToyConveyor — everywhere else Rung E is already the best
available. Swapping in Rung B's readout for just these two classes closes **half the remaining
gap to STgram** (-3.9 → -1.9) with **zero new training** — it's a selection rule over models
that already exist.

## Bottom line

The -3.9 residual is not one uniform gap. It splits into:
1. **ToyCar (-1.9 achievable today):** a pure trade-off from Rung E's architecture choice —
   Rung B already nearly matches STgram on this class; a per-class or fused readout recovers it.
2. **ToyConveyor (~-2 likely structural):** STgram-MFN's own per-ID numbers show one machine unit
   (id_02, 62.4) is hard for the supervised SOTA too — the MambaAD-track ceiling here may already
   be close to what's achievable while pooling that unit in.

## Update (2026-07-31) — seed-repeat confirms the ToyCar trade-off is real, not noise

3-seed repeat of Rung B/E (`docs/plots/mimii_section_rungE/CONCLUSION.md`): the *overall-mean*
E > B claim does NOT reproduce (B and E are statistically tied, 85.9 vs 86.2 ± noise), but the
**ToyCar gap reproduces in every seed** — B beats E by 4.8–7.9 pts (mean 6.7) every time. So the
trade-off described in §1 is a genuine, repeatable property of Rung E's architecture choice, not
a single-seed fluke — while the "E is the new best" framing it was originally read from is.

## Next (open, cheap)
- **A real fused/per-class-readout model** (not just an oracle) — e.g. pick neg_cos vs maha_embed
  per class on a held-out val split — to see how much of the 88.8 oracle a *deployable* rule
  actually captures. This is now the main open item on the whole ladder.
