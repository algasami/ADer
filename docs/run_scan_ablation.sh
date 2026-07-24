#!/bin/bash
# =============================================================================
# Scan-type-ablation driver: TRAIN one e50 x log-Mel MambaAD run per Mamba scan
# curve, then re-score every run with the SAME test-time readout (student-maha,
# the winning readout from the scorer/ front).
#
# Unlike the scorer ablation (docs/run_scorer_ablation.sh, which re-scores ONE
# trained checkpoint many ways), scan_type is a decoder-architecture choice baked
# in at training time, so each curve needs its OWN training run:
#   * hilbert -> REUSED: the existing e50 log-Mel baseline run ($HILBERT_RUN) is
#                already scan_type='hilbert'; its student-maha re-score was produced
#                by the scorer sweep, so we just copy that metric file.
#   * scan / sweep / zigzag / zorder -> trained here (scan-type/<x>.py, native
#                cos-residual readout — same recipe as the hilbert baseline), 2 GPUs
#                in 2 waves, then re-scored with student-maha via reeval_sp_mean.py.
#
# student-maha only changes the TEST-TIME readout, not the loss/backprop, so a run
# trained under cos-residual and re-scored with maha is identical to one trained
# under the maha config — which keeps every scan run directly comparable to the
# reused hilbert baseline. Re-scoring is single-GPU (the per-class bank is not
# gathered across DDP ranks).
#
#   nohup bash docs/run_scan_ablation.sh > <log> 2>&1 &
#
# Writes one 42-column metric file per scan to $OUT_DIR (native metric.txt layout,
# epoch-aligned), consumed by docs/plot_scan_ablation.py.
# =============================================================================
cd /home/f74134118/ADer
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"   # conda activate.d refs this; must exist first
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mamba-ad
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:$CONDA_PREFIX/lib"
export MPLBACKEND=Agg

TAG=scanabl                                     # deterministic logdir suffix
GPUS=(2 3)                                       # two free GPUs, one run each per wave
HILBERT_RUN=runs/MAMBAADTrainer_MambaAD_configs_mambaad_mambaad_mimii_toy_20260722-215044
HILBERT_MAHA=docs/plots/mimii_scorer/_data/student-maha.txt   # hilbert already re-scored here
OUT_DIR=docs/plots/mimii_scan/_data
SCRATCH="${CLAUDE_JOB_DIR:-/tmp/claude}/tmp/scan_ablation/scratch"
mkdir -p "$OUT_DIR" "$SCRATCH"

STAMP() { date '+%Y-%m-%d %H:%M:%S'; }
say() { echo "[$(STAMP)] $*"; }

CFG_DIR=MambaAD/configs/mambaad/mimii/scan-type
MAHA_DIR=MambaAD/configs/mambaad/mimii/scan-type-maha
TRAIN_SCANS=(scan sweep zigzag zorder)          # hilbert reused, not retrained

run_dir_of() { echo "runs/MAMBAADTrainer_MambaAD_configs_mambaad_mimii_scan-type_${1}_${TAG}"; }

# ---- train one scan curve on a given GPU (blocking; call with &) ----
train_one() {
  local scan="$1" gpu="$2"
  local rd; rd="$(run_dir_of "$scan")"
  if [[ -f "$rd/net_50.pth" ]]; then
    say "SKIP train $scan — $rd/net_50.pth already exists"; return 0
  fi
  say "----- TRAIN $scan on GPU $gpu -> $rd"
  CUDA_VISIBLE_DEVICES="$gpu" python -u run.py -c "$CFG_DIR/${scan}.py" -m train \
      trainer.logdir_sub="$TAG" > "$SCRATCH/train_${scan}.log" 2>&1
  local rc=$?
  say "train $scan exited $rc"
  return $rc
}

# ---- re-score one trained run with student-maha ----
reeval_one() {
  local scan="$1" gpu="$2"
  local rd; rd="$(run_dir_of "$scan")"
  local out="$OUT_DIR/${scan}.txt"
  local raw="$OUT_DIR/${scan}.rawlog"
  say "----- REEVAL(student-maha) $scan on GPU $gpu -> $out"
  CUDA_VISIBLE_DEVICES="$gpu" python -u docs/reeval_sp_mean.py \
      --cfg "$MAHA_DIR/${scan}.py" \
      --run-dir "$rd" \
      --out "$out" \
      --scratch "$SCRATCH" > "$raw" 2>&1
  local rc=$?
  if [[ $rc -ne 0 || ! -s "$out" ]]; then
    say "FAILED reeval $scan (rc=$rc); see $raw"; return 1
  fi
  grep -E "CHECK|wrote|classes:|metrics:" "$raw" | sed 's/^/    /'
  say "done reeval $scan"
}

# ===================== 1) TRAIN (2 waves of 2) =====================
say "===== scan-type ablation: training ${TRAIN_SCANS[*]} on GPUs ${GPUS[*]}"
ok=1
i=0
while [[ $i -lt ${#TRAIN_SCANS[@]} ]]; do
  a="${TRAIN_SCANS[$i]}";  b="${TRAIN_SCANS[$((i+1))]:-}"
  say "--- wave: $a (GPU ${GPUS[0]})${b:+ + $b (GPU ${GPUS[1]})}"
  train_one "$a" "${GPUS[0]}" & p1=$!
  p2=""
  [[ -n "$b" ]] && { train_one "$b" "${GPUS[1]}" & p2=$!; }
  wait $p1 || ok=0
  [[ -n "$p2" ]] && { wait $p2 || ok=0; }
  i=$((i+2))
done
[[ $ok -eq 1 ]] || { say "ABORTED: a training run failed (see $SCRATCH/train_*.log)"; exit 1; }
say "===== all trainings done"

# ===================== 2) RE-SCORE with student-maha =====================
# hilbert: reuse the scorer sweep's already-computed student-maha metric file
if [[ -f "$HILBERT_MAHA" ]]; then
  cp "$HILBERT_MAHA" "$OUT_DIR/hilbert.txt"
  say "copied hilbert student-maha metrics from $HILBERT_MAHA"
else
  say "hilbert student-maha not found at $HILBERT_MAHA; re-scoring $HILBERT_RUN"
  reeval_one_run() { :; }   # fallback: reeval the baseline directly
  CUDA_VISIBLE_DEVICES="${GPUS[0]}" python -u docs/reeval_sp_mean.py \
      --cfg "$MAHA_DIR/hilbert.py" --run-dir "$HILBERT_RUN" \
      --out "$OUT_DIR/hilbert.txt" --scratch "$SCRATCH" > "$OUT_DIR/hilbert.rawlog" 2>&1 \
      && say "re-scored hilbert" || { say "FAILED hilbert reeval"; ok=0; }
fi

for scan in "${TRAIN_SCANS[@]}"; do
  reeval_one "$scan" "${GPUS[0]}" || ok=0
done

# ===================== 3) AGGREGATE + PLOT =====================
if [[ $ok -eq 1 ]]; then
  say "===== all scans re-scored; aggregating + plotting"
  python -u docs/plot_scan_ablation.py --data-dir "$OUT_DIR"
  say "DONE"
else
  say "FINISHED WITH FAILURES — plotting whatever is present"
  python -u docs/plot_scan_ablation.py --data-dir "$OUT_DIR" || true
fi
