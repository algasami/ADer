# docs/ — plotting and re-evaluation tooling

- `plot_mimii_val_metrics.py` — AUROC/AP/F1 vs. epoch from a run's `metric.txt`. Auto-detects the
  21- vs 42-column layout; `--family sp_max|sp_mean`.
- `reeval_sp_mean.py` (+ `reeval_and_plot_sp_mean.sh`) — re-evaluates saved `net_<E>.pth`
  checkpoints, writing `<run-dir>/metric_reeval.txt` in the native 42-column layout. Named for
  recovering sp_mean on sp_max-only runs, but it is **also the workhorse for the scorer and scan
  ablations** — it re-scores a trained decoder under a swapped `ABL_SCORER` without retraining.
- **Ablation drivers + plotters** — each writes one folder per figure under `docs/plots/mimii_*`
  holding both `plot.png` and its `data.csv`, plus a `*_summary.csv`:
  `run_scorer_ablation.sh` + `plot_scorer_ablation.py`, `run_scan_ablation.sh` +
  `plot_scan_ablation.py`, and `plot_backbone_ablation.py` (reads `runs/audio_probe/`).

- **Phase 1 / Phase 2 figures** — `plot_phase1_rungG.py` and `plot_phase2_asnorm.py` write the
  figures + CSVs into `docs/plots/phase1_rungG/` and `docs/plots/phase2_asnorm/` (those folders
  were prose-only until 2026-08-10). Both read only the per-run CSVs (`metric_curve.csv`,
  `meanid_folds.csv`, `asnorm_by_section.csv`, `train_log.csv`) plus STgram-MFN's own
  `runs/STgram-MFN(m=0.7,s=30)/running.log`, which evaluates every 10 epochs and is what makes
  the epoch-for-epoch baseline comparison possible — **nothing is retrained**. Note the two
  scripts use different scoring conventions on purpose: Phase 1 quotes the ladder's
  best-readout/best-epoch-on-test number, Phase 2 quotes the held-out rule.

- **Honest-rule scoring** — `select_heldout_epoch.py` does it for the Phase-2 fbank runs (reads
  `meanid_folds.csv`, which only the Rung H version of `section_ast_rungG.py` writes).
  `rescore_ladder.py` (+ `plot_rescore_ladder.py`, driver `run_rescore_ladder.sh`) does it for
  the PNG ladder A-F from `scores_by_epoch.npz`, and reports pooled-clip vs mean-of-per-ID
  crossed with test-selected vs held-out. Two things it does that the older script does not:
  it selects the **readout** on the held-out half as well as the epoch (every rung's headline
  maximized over both on test), and it averages the estimate over 20 independent fold draws so
  split noise is not mistaken for a rung difference. Rung F has no run of its own — it is
  reconstructed as Rung E under the per-class readout policy.

> **Gotcha: the rung scripts truncate their CSVs at startup.** Pointing a re-run at an existing
> run directory destroys that run's curves the moment it starts (`best_summary.csv` survives —
> it is written only at the end). `runs/` is gitignored, so there is no undo. Always pass an
> explicit `--out_dir` when repeating a rung. The July ladder runs are preserved under
> `runs/section_rung{B,E}/log-Mel*_july2026/`.

Remember the run-comparison rule from the root `CLAUDE.md`: AUROC peaks early and there is no
best-checkpoint selection, so compare runs at their peaks — plot first.
