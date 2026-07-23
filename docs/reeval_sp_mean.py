"""Re-evaluate saved MambaAD checkpoints to recover metrics the original run didn't record.

The long MIMII runs (20260716-*) were trained with a config whose `metrics` list held only
the sp_max family, so their metric.txt has no sp_mean columns. But sp_max and sp_mean are
BOTH pooled from the same anomaly map (util/metric.py: .max vs .mean over pixels), so the
saved net_<E>.pth checkpoints already contain everything needed — we just re-run the eval
with the current 6-metric config and read sp_mean off.

This builds the model + test loader ONCE, then swaps each net_<E>.pth in and re-tests, so a
whole run's 20 checkpoints cost one data/model setup rather than 20 subprocess launches.

Output: writes `<run-dir>/metric_reeval.txt` in the exact 42-column layout a native 6-metric
training run produces (see trainer/_base_trainer.py:_finish), with each checkpoint's values
placed at its true epoch row (row E-1). plot_mimii_val_metrics.py reads it identically to a
native metric.txt.

Usage:
    python docs/reeval_sp_mean.py \
        --cfg MambaAD/configs/mambaad/mambaad_mimii_stgram.py \
        --run-dir runs/MAMBAADTrainer_..._stgram_20260716-121818

Validation: the re-computed sp_max columns should match the run's original metric.txt to
~1e-3; the script prints a check against the first checkpoint so you can confirm the pipeline
reproduces before trusting the sp_mean values.
"""
import argparse
import glob
import os
import re
import sys
from argparse import Namespace

import numpy as np
import torch

# import the ADer machinery exactly as run.py does
sys.path.insert(0, os.getcwd())
from configs import get_cfg
from util.net import init_training
from util.util import run_pre, init_checkpoint
from trainer import get_trainer


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfg", required=True, help="config .py (current 6-metric version)")
    ap.add_argument("--run-dir", required=True, help="run dir holding net_<E>.pth checkpoints")
    ap.add_argument("--out", default=None, help="output file (default <run-dir>/metric_reeval.txt)")
    ap.add_argument("--scratch", default="/tmp/claude-1007/reeval",
                    help="throwaway logdir base so runs/ isn't polluted")
    ap.add_argument("--glob", default="net_*.pth", help="checkpoint glob within run-dir")
    return ap.parse_args()


def find_checkpoints(run_dir, pattern):
    """Return [(epoch, path), ...] sorted by epoch, only files like net_<int>.pth."""
    out = []
    for p in glob.glob(os.path.join(run_dir, pattern)):
        m = re.fullmatch(r"net_(\d+)\.pth", os.path.basename(p))
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def build_trainer(cfg_path, scratch):
    """Mirror run.py's setup for a single-process test session."""
    term = Namespace(cfg_path=cfg_path, mode="test", sleep=-1, memory=-1,
                     dist_url="env://", logger_rank=0,
                     opts=[f"trainer.checkpoint={scratch}", "fvcore_is=False"])
    cfg = get_cfg(term)
    run_pre(cfg)
    init_training(cfg)          # no torchrun env -> cfg.dist=False, single GPU
    init_checkpoint(cfg)        # fresh throwaway logdir under scratch
    return cfg, get_trainer(cfg)


def load_net(net, path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "net" in sd:
        sd = sd["net"]
    net.load_state_dict(sd, strict=True)
    net.eval()


def finish_layout(cls_names, metrics):
    """Reproduce _finish's column order: (cls,metric) pairs, Avg interleaved after last cls."""
    keys = []
    for idx, cls in enumerate(cls_names):
        for metric in metrics:
            keys.append(f"{metric}_{cls}")
            if idx == len(cls_names) - 1 and len(cls_names) > 1:
                keys.append(f"{metric}_Avg")
    return keys


def main():
    args = parse_args()
    out = args.out or os.path.join(args.run_dir, "metric_reeval.txt")
    ckpts = find_checkpoints(args.run_dir, args.glob)
    if not ckpts:
        sys.exit(f"no checkpoints matching {args.glob} in {args.run_dir}")
    print(f"==> {len(ckpts)} checkpoints, epochs {ckpts[0][0]}..{ckpts[-1][0]}")

    cfg, trainer = build_trainer(args.cfg, args.scratch)
    cls_names, metrics = trainer.cls_names, trainer.metrics
    print(f"==> classes: {list(cls_names)}")
    print(f"==> metrics: {list(metrics)}")

    # metric_recorder accumulates one entry per test() call, in checkpoint order.
    for epoch, path in ckpts:
        load_net(trainer.net, path)
        trainer.test()
        auroc = trainer.metric_recorder[f"{metrics[0]}_Avg"][-1] if len(cls_names) > 1 else None
        print(f"   ep{epoch:>4}: {path}  ({metrics[0]}_Avg={auroc:.3f})" if auroc is not None
              else f"   ep{epoch:>4}: {path}")

    # ---- assemble epoch-aligned metric.txt clone ----
    keys = finish_layout(cls_names, metrics)
    max_epoch = ckpts[-1][0]
    table = np.zeros((max_epoch, len(keys)), dtype=float)
    for ci, (epoch, _) in enumerate(ckpts):
        for col, key in enumerate(keys):
            table[epoch - 1, col] = trainer.metric_recorder[key][ci]

    with open(out, "w") as f:
        for row in table:
            f.write("".join("{:3.5f}\t".format(v) for v in row) + "\n")
    print(f"==> wrote {out}  ({table.shape[0]} rows x {table.shape[1]} cols)")

    # ---- validation: re-computed sp_max vs the run's original metric.txt ----
    orig = os.path.join(args.run_dir, "metric.txt")
    if os.path.exists(orig) and "mAUROC_sp_max" in metrics:
        a = np.loadtxt(orig)
        # original layout: M_orig metrics/class; detect and compare AUROC_sp_max of first class
        C = len(cls_names)
        m_orig = a.shape[1] // (C + 1)
        e0 = ckpts[0][0]
        orig_fan = a[e0 - 1, 0]                       # first class, first metric (sp_max AUROC)
        new_fan = table[e0 - 1, keys.index(f"mAUROC_sp_max_{cls_names[0]}")]
        print(f"==> CHECK ep{e0} {cls_names[0]} AUROC_sp_max: original={orig_fan:.3f} "
              f"re-eval={new_fan:.3f}  |Δ|={abs(orig_fan - new_fan):.4f} "
              f"(orig file had {m_orig} metrics/class)")


if __name__ == "__main__":
    main()
