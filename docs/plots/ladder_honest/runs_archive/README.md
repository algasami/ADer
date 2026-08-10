# Run archive — the honest re-score of rungs A–F

`runs/` is gitignored, and these runs cost ~10 GPU-hours, so the per-run CSVs are preserved
here (476 KB) ahead of a machine reset. One directory per run, named `<rung-dir>_<run>`.

* `*_july2026/` — the ORIGINAL July 2026 ladder runs (backups). Rung E seed1/seed2 hold only
  `best_summary.csv` plus `metric_curve_from_log.csv`, a mean-only curve salvaged from the run
  log after their per-epoch CSVs were destroyed on 2026-08-09 (see `NOTE.md`, 8/10 entry).
* everything else — the 2026-08-10 re-runs that produced `../per_run.csv`.

**Not preserved: `scores_by_epoch.npz`** (58 MB across the 13 re-runs). Those hold the per-clip
score of every test clip at every epoch, which is what lets any NEW selection rule be evaluated
post-hoc without retraining. Recreating them means re-running the ladder
(`docs/run_rescore_ladder.sh`). The CSVs here are enough to reproduce every number in
`CONCLUSION.md` except a re-derivation under a different fold definition or selection rule.
