#!/usr/bin/env bash
# Re-run rungs A-E so the ladder can be re-scored under the honest rule.
# ---------------------------------------------------------------------
# The original runs kept no checkpoints and no per-clip scores -- only per-class AUROC
# aggregates -- so mean-of-per-ID AUROC and held-out epoch selection cannot be recovered from
# what is on disk. These runs repeat each rung at its ORIGINAL hyperparameters (recovered from
# the run logs in runs/section_rung*/) with per-clip score dumping on, at 3 seeds each.
#
# Rung F is not here: it never had a script. It was a post-hoc per-class readout policy over
# Rung E, and docs/rescore_ladder.py reconstructs it from Rung E's dumped scores.
#
#   bash docs/run_rescore_ladder.sh          # ~6-7 h wall on 4 GPUs
#
# One queue per GPU, balanced by cost (D > C ~ E > B >> A). Logs: runs/rescore/<job>.log
set -u
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/anaconda3/envs/mamba-ad/lib"
# 55 of 64 cores are busy with another user's job; keep the loaders modest and stop torch
# from spawning one intra-op thread per core per process.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
CFG=MambaAD/configs/mambaad/mimii/e50/log-Mel.py
W=4
mkdir -p runs/rescore

run() {  # run <gpu> <tag> <script> <args...>
  local gpu=$1 tag=$2; shift 2
  echo "[$(date +%H:%M:%S)] start $tag on gpu$gpu"
  CUDA_VISIBLE_DEVICES=$gpu python "$@" > "runs/rescore/$tag.log" 2>&1
  echo "[$(date +%H:%M:%S)] done  $tag (exit $?)"
}

queue0() { for s in 0 1 2; do
  run 0 "D_seed$s" diagnostics/section_joint_rungD.py -c $CFG --epochs 50 --sub 2 \
      --lambda_cls 1.0 --eval_maha --workers $W --seed $s
done; }
queue1() { for s in 0 1 2; do
  run 1 "C_seed$s" diagnostics/section_mamba_rungC.py -c $CFG --epochs 50 --sub 2 \
      --lr 1e-3 --eval_maha --workers $W --seed $s
done; }
queue2() { for s in 0 1 2; do
  run 2 "E_seed$s" diagnostics/section_combined_rungE.py -c $CFG --epochs 50 --sub 2 \
      --lr_teacher 1e-4 --lr_rest 1e-3 --eval_maha --workers $W --seed $s
done; }
queue3() { for s in 0 1 2; do
  run 3 "B_seed$s" diagnostics/section_finetune_rungB.py -c $CFG --epochs 50 --sub 2 \
      --eval_maha --workers $W --seed $s
done
for s in 0 1 2; do
  run 3 "A_seed$s" diagnostics/section_classifier_probe.py -c $CFG --head_epochs 100 --sub 2 \
      --workers $W --seed $s
done; }

queue0 & queue1 & queue2 & queue3 &
wait
echo "[$(date +%H:%M:%S)] all rungs finished"
