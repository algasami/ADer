#!/bin/bash
# Detached driver: re-evaluate the three long MIMII runs to recover sp_mean, then plot.
# Safe to run under nohup and survive disconnect. Writes a clean status log + DONE/FAILED marker.
#
#   nohup bash docs/reeval_and_plot_sp_mean.sh > <log> 2>&1 &
#
# Per-run raw output (progress bars etc.) goes to <run-dir>/reeval_raw.log; this script's
# stdout is only high-level status lines.
cd /home/f74134118/ADer
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mamba-ad
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$CONDA_PREFIX/lib"
export CUDA_VISIBLE_DEVICES=3          # free GPU; GPUs 0-2 host another job
export MPLBACKEND=Agg                  # headless plotting

STAMP() { date '+%Y-%m-%d %H:%M:%S'; }
say() { echo "[$(STAMP)] $*"; }

declare -A CFG=(
  [stgram]=MambaAD/configs/mambaad/mambaad_mimii_stgram.py
  [delta]=MambaAD/configs/mambaad/mambaad_mimii_stgram_delta.py
  [toy]=MambaAD/configs/mambaad/mambaad_mimii_toy.py
)
declare -A DIR=(
  [stgram]=runs/MAMBAADTrainer_MambaAD_configs_mambaad_mambaad_mimii_stgram_20260716-121818
  [delta]=runs/MAMBAADTrainer_MambaAD_configs_mambaad_mambaad_mimii_stgram_delta_20260716-121820
  [toy]=runs/MAMBAADTrainer_MambaAD_configs_mambaad_mambaad_mimii_toy_20260716-121820
)

ok=1
for key in stgram delta toy; do
  d="${DIR[$key]}"
  say "===== re-eval $key -> $d"
  python -u docs/reeval_sp_mean.py --cfg "${CFG[$key]}" --run-dir "$d" > "$d/reeval_raw.log" 2>&1
  rc=$?
  if [[ $rc -ne 0 ]]; then
    say "FAILED: $key re-eval exited $rc (see $d/reeval_raw.log)"; ok=0; break
  fi
  if [[ ! -s "$d/metric_reeval.txt" ]]; then
    say "FAILED: $key produced no metric_reeval.txt"; ok=0; break
  fi
  # surface the sp_max reproduction check from the raw log
  grep -E "CHECK|wrote" "$d/reeval_raw.log" | sed 's/^/    /'
  say "done $key"
done

if [[ $ok -eq 1 ]]; then
  say "===== all re-evals done; plotting long sp_mean + sp_max"
  python docs/plot_mimii_val_metrics.py --preset long --family sp_mean
  python docs/plot_mimii_val_metrics.py --preset long --family sp_max
  say "DONE"
else
  say "ABORTED with failures"
fi
