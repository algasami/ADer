"""Plot validation metrics (AUROC / AP / F1) vs. epoch for MambaAD-MIMII runs.

Reads each run's metric file (one row per epoch, nonzero on validation epochs only) and
writes two figures under the preset's output dir:
  - mimii_val_avg_metrics_<family>.png     : Avg over classes, one panel per metric.
  - mimii_val_auroc_per_class_<family>.png : per-class AUROC, one panel per class.

Presets select which run set to plot; --family selects the pixel-pooling variant
(sp_max = max-pool the anomaly map, sp_mean = mean-pool). Add a new preset entry for a
future run group instead of copying this file:

    python docs/plot_mimii_val_metrics.py --preset short --family sp_max
    python docs/plot_mimii_val_metrics.py --preset long  --family sp_mean

Column layout of the metric file (see trainer/_base_trainer.py:_finish): outer loop over
classes, inner loop over M metrics, with the class Avg interleaved right after the last
class (ToyConveyor). The metric order is fixed by config:
    0 mAUROC_sp_max  1 mAP_sp_max  2 mF1_max_sp_max      (sp_max family)
    3 mAUROC_sp_mean 4 AP_sp_mean  5 F1_max_sp_mean      (sp_mean family, if recorded)

M is NOT fixed and is auto-detected per file as ncols // (len(CLASSES)+1): runs that record
only sp_max have M=3 (21 cols); runs that also record sp_mean have M=6 (42 cols). Hardcoding
M silently misaligns every column past the first class on M=6 files.

The long (20260716) runs were trained with an sp_max-only config, so their native metric.txt
has M=3 and no sp_mean. `metric_reeval.txt` (produced by docs/reeval_sp_mean.py, which re-runs
the saved checkpoints through the current 6-metric evaluator) supplies M=6 for them; the long
preset points --family sp_mean at that file.
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

CLASSES = ["fan", "pump", "slider", "valve", "ToyCar", "ToyConveyor"]
COLORS = {"STgram+Sgram": "#1f77b4", "STgram+delta": "#d62728", "log-Mel x3": "#2ca02c"}

# family -> (within-class metric indices for AUROC/AP/F1, display labels)
FAMILIES = {
    "sp_max":  dict(idx=[0, 1, 2], names=["mAUROC_sp_max", "mAP_sp_max", "mF1_max_sp_max"]),
    "sp_mean": dict(idx=[3, 4, 5], names=["mAUROC_sp_mean", "AP_sp_mean", "F1_max_sp_mean"]),
}

_LONG = "runs/MAMBAADTrainer_MambaAD_configs_mambaad_mambaad_mimii_{}_20260716-{}"
_SHORT = "runs/MAMBAADTrainer_MambaAD_configs_mambaad_mambaad_mimii_{}_20260720-{}"
_LOWLR = "runs/MAMBAADTrainer_MambaAD_configs_mambaad_mambaad_mimii_{}_latest_lowlr_lowwd_20260721"
# super-short 50-epoch runs (20260722): each rep has its own launch timestamp.
_SUPERSHORT = "runs/MAMBAADTrainer_MambaAD_configs_mambaad_mambaad_mimii_{}_20260722-{}"

# Each preset fully describes one figure set. `runs` maps a legend label -> run directory.
# `file` maps a metric family -> the filename (within each run dir) that holds it.
PRESETS = {
    "long": dict(
        runs={
            "STgram+Sgram": _LONG.format("stgram", "121818"),
            "STgram+delta": _LONG.format("stgram_delta", "121820"),
            "log-Mel x3":   _LONG.format("toy", "121820"),
        },
        # native metric.txt is sp_max-only (M=3); sp_mean comes from the re-eval clone.
        file={"sp_max": "metric.txt", "sp_mean": "metric_reeval.txt"},
        out="docs/plots",
        xmax=1000,
        title="MambaAD / MIMII",
        avg_note="(open ring = peak; note decay after epoch 50)",
    ),
    "short": dict(
        runs={
            "STgram+Sgram": _SHORT.format("stgram", "020034"),
            "STgram+delta": _SHORT.format("stgram_delta", "020036"),
            "log-Mel x3":   _SHORT.format("toy", "020017"),
        },
        # short runs recorded all 6 metrics natively, so both families live in metric.txt.
        file={"sp_max": "metric.txt", "sp_mean": "metric.txt"},
        out="docs/short-epoch-plots",
        xmax=200,
        title="MambaAD / MIMII (short schedule, 200 ep)",
        avg_note="(open ring = peak)",
    ),
    "lowlr": dict(
        runs={
            "STgram+Sgram": _LOWLR.format("stgram_stgram_sgram"),
            "STgram+delta": _LOWLR.format("stgram_delta_stgram_delta"),
            "log-Mel x3":   _LOWLR.format("toy_logmel"),
        },
        # low-lr/low-wd runs record all 6 metrics natively, so both families live in metric.txt.
        file={"sp_max": "metric.txt", "sp_mean": "metric.txt"},
        out="docs/lowlr-lowwd-plots",
        xmax=200,
        title="MambaAD / MIMII (low lr, low wd, 200 ep)",
        avg_note="(open ring = peak)",
    ),
    "supershort": dict(
        runs={
            "STgram+Sgram": _SUPERSHORT.format("stgram", "215109"),
            "STgram+delta": _SUPERSHORT.format("stgram_delta", "215108"),
            "log-Mel x3":   _SUPERSHORT.format("toy", "215044"),
        },
        # super-short runs record all 6 metrics natively, so both families live in metric.txt.
        file={"sp_max": "metric.txt", "sp_mean": "metric.txt"},
        out="docs/supershort-epoch-plots",
        xmax=50,
        title="MambaAD / MIMII (super-short schedule, 50 ep)",
        avg_note="(open ring = peak)",
    ),
}


def build_col(ncols):
    """Column index of (class, metric_idx), auto-detecting metrics-per-class M from ncols.

    Layout: (C-1) regular classes as contiguous M-blocks, then ToyConveyor and Avg
    interleaved (ToyConveyor at even offsets, Avg at odd) for the tail. Total = (C+1)*M.
    """
    C = len(CLASSES)
    if ncols % (C + 1) != 0:
        raise ValueError(f"metric file has {ncols} cols, not divisible by {C + 1}; "
                         "unexpected layout — cannot infer metrics-per-class.")
    M = ncols // (C + 1)
    col = {}
    for ci, c in enumerate(CLASSES[:-1]):
        for mi in range(M):
            col[(c, mi)] = ci * M + mi
    base = (C - 1) * M
    for mi in range(M):
        col[("ToyConveyor", mi)] = base + 2 * mi
        col[("Avg", mi)] = base + 2 * mi + 1
    return col


def load(path, need_idx):
    a = np.loadtxt(path)
    if a.ndim == 1:                                   # single-row file -> keep 2D
        a = a[None, :]
    col = build_col(a.shape[1])
    if max(need_idx) >= a.shape[1] // len(CLASSES):   # family not present in this file
        raise ValueError(f"{path} has too few metrics/class for the requested family "
                         f"(need within-class index {max(need_idx)}); is this an sp_max-only "
                         f"file? Point the preset at a re-eval file.")
    rows = np.where(a[:, col[("Avg", 0)]] > 0)[0]     # validation epochs (Avg AUROC nonzero)
    return rows + 1, a[rows], col                     # epochs, values, column map


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=PRESETS, default="long",
                    help="which run set / output config to plot (default: long)")
    ap.add_argument("--family", choices=FAMILIES, default="sp_max",
                    help="pixel-pooling variant: sp_max (max) or sp_mean (mean) (default: sp_max)")
    ap.add_argument("--out", help="override output directory")
    ap.add_argument("--xmax", type=float, help="override x-axis (epoch) upper limit")
    args = ap.parse_args()

    cfg = PRESETS[args.preset]
    fam = FAMILIES[args.family]
    fam_idx, fam_names = fam["idx"], fam["names"]
    out = str(args.out or cfg["out"])
    xmax = float(args.xmax if args.xmax is not None else cfg["xmax"])
    title, avg_note = str(cfg["title"]), str(cfg["avg_note"])
    fname = cfg["file"][args.family]

    os.makedirs(out, exist_ok=True)
    data = {name: load(os.path.join(run_dir, fname), fam_idx)
            for name, run_dir in cfg["runs"].items()}
    sfx = args.family  # output-filename suffix keeps sp_max / sp_mean figures separate

    # ---------- Figure 1: Avg metrics ----------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    titles = [f"Image-level AUROC (Avg, {args.family})",
              f"Average Precision (Avg, {args.family})",
              f"F1-max (Avg, {args.family})"]
    for pos, (ax, ptitle) in enumerate(zip(axes, titles)):
        raw = fam_idx[pos]
        for name, (ep, vals, col) in data.items():
            y = vals[:, col[("Avg", raw)]]
            ax.plot(ep, y, "-o", ms=4, lw=1.8, color=COLORS[name], label=name)
            pk = int(np.argmax(y))
            ax.scatter([ep[pk]], [y[pk]], s=90, facecolors="none",
                       edgecolors=COLORS[name], linewidths=1.8, zorder=5)
        ax.set_title(ptitle)
        ax.set_xlabel("epoch")
        ax.set_ylabel(fam_names[pos] + " (%)")
        ax.set_xlim(0, xmax)
        ax.grid(alpha=0.3)
        if pos == 0:
            ax.axhline(50, ls="--", c="gray", lw=1, alpha=0.7)
            ax.text(0.02 * xmax, 50.4, "chance", color="gray", fontsize=8)
        ax.legend(fontsize=9)
    fig.suptitle(f"{title} — validation metrics ({args.family}) vs. epoch {avg_note}",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    f1 = os.path.join(out, f"mimii_val_avg_metrics_{sfx}.png")
    fig.savefig(f1, dpi=150)

    # ---------- Figure 2: per-class AUROC ----------
    auroc_raw = fam_idx[0]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    for ax, cls in zip(axes.ravel(), CLASSES):
        for name, (ep, vals, col) in data.items():
            ax.plot(ep, vals[:, col[(cls, auroc_raw)]], "-o", ms=3.5, lw=1.6,
                    color=COLORS[name], label=name)
        ax.axhline(50, ls="--", c="gray", lw=1, alpha=0.7)
        ax.set_title(f"{cls} — AUROC ({args.family})")
        ax.set_xlim(0, xmax)
        ax.set_ylabel(fam_names[0] + " (%)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("epoch")
    fig.suptitle(f"{title} — per-class image-level AUROC ({args.family}) vs. epoch", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    f2 = os.path.join(out, f"mimii_val_auroc_per_class_{sfx}.png")
    fig.savefig(f2, dpi=150)

    print("wrote", f1)
    print("wrote", f2)
    for name, (ep, vals, col) in data.items():
        y = vals[:, col[("Avg", auroc_raw)]]
        print(f"{name:14s} AUROC({args.family}): start(ep{ep[0]})={y[0]:.2f}  "
              f"peak={y.max():.2f}@ep{ep[int(np.argmax(y))]}  final={y[-1]:.2f}")


if __name__ == "__main__":
    main()
