#!/bin/bash
# =============================================================================
# Scorer-ablation driver: re-score ONE trained e50 x log-Mel MambaAD run with
# each test-time score readout in MambaAD/configs/mambaad/mimii/scorer/.
#
# Reuses docs/reeval_sp_mean.py, which builds the model + loaders ONCE and swaps
# each net_<E>.pth in, re-running trainer.test() under the config's `cfg.scorer`.
# So every scorer sees the *same* trained decoder at every saved epoch:
#   * cos-residual         -> native per-pixel readout (reproduces metric.txt)
#   * student-maha/knn      -> Maha/kNN on the DECODER's GAP features (refit each ep)
#   * teacher-maha/knn      -> Maha/kNN on the FROZEN encoder (checkpoint-independent;
#                              static_fit => fit once, flat across epochs)
#
#   nohup bash docs/run_scorer_ablation.sh > <log> 2>&1 &
#
# Writes one 42-column metric file per scorer to $OUT_DIR (native metric.txt layout,
# epoch-aligned), consumed by docs/plot_scorer_ablation.py.
# =============================================================================
cd /home/f74134118/ADer
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"   # conda activate.d refs this; must exist first
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mamba-ad
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:$CONDA_PREFIX/lib"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export MPLBACKEND=Agg

RUN_DIR=runs/MAMBAADTrainer_MambaAD_configs_mambaad_mambaad_mimii_toy_20260722-215044
OUT_DIR=docs/plots/mimii_scorer/_data
SCRATCH="${CLAUDE_JOB_DIR:-/tmp/claude}/tmp/scorer_ablation/scratch"
mkdir -p "$OUT_DIR" "$SCRATCH"

STAMP() { date '+%Y-%m-%d %H:%M:%S'; }
say() { echo "[$(STAMP)] $*"; }

declare -A CFG=(
  [cos-residual]=MambaAD/configs/mambaad/mimii/scorer/cos-residual.py
  [student-maha]=MambaAD/configs/mambaad/mimii/scorer/student-maha.py
  [student-knn]=MambaAD/configs/mambaad/mimii/scorer/student-knn.py
  [teacher-maha]=MambaAD/configs/mambaad/mimii/scorer/teacher-maha.py
  [teacher-knn]=MambaAD/configs/mambaad/mimii/scorer/teacher-knn.py
)
ORDER=(cos-residual student-maha student-knn teacher-maha teacher-knn)

say "===== scorer ablation on $RUN_DIR (GPU $CUDA_VISIBLE_DEVICES)"
ok=1
for key in "${ORDER[@]}"; do
  out="$OUT_DIR/${key}.txt"
  raw="$OUT_DIR/${key}.rawlog"
  say "----- scoring: $key -> $out"
  python -u docs/reeval_sp_mean.py \
      --cfg "${CFG[$key]}" \
      --run-dir "$RUN_DIR" \
      --out "$out" \
      --scratch "$SCRATCH" > "$raw" 2>&1
  rc=$?
  if [[ $rc -ne 0 ]]; then
    say "FAILED: $key exited $rc (see $raw)"; ok=0; break
  fi
  if [[ ! -s "$out" ]]; then
    say "FAILED: $key produced no output"; ok=0; break
  fi
  grep -E "CHECK|wrote|classes:|metrics:" "$raw" | sed 's/^/    /'
  say "done $key"
done

if [[ $ok -eq 1 ]]; then
  say "===== all scorers done; aggregating + plotting"
  python -u docs/plot_scorer_ablation.py --data-dir "$OUT_DIR" --run-dir "$RUN_DIR"
  say "DONE"
else
  say "ABORTED with failures"
fi
