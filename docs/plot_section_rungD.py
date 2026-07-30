"""Aggregate + plot Rung D of the "labels close the gap?" ladder
(diagnostics/section_joint_rungD.py).

Rung D = the faithful "MambaAD + labels": train the Mamba student with BOTH the native
reconstruction distillation (CosLoss) AND the section-classification head, then score with the
reconstruction residual, the classification readouts, and their fusion. The story is a READOUT
comparison (does adding recon help? does fusion beat the single best?), so the figures are
readout-centric:

    docs/plots/mimii_section_rungD/
        auroc_vs_epoch/   plot.png data.csv   # recon stays ~chance; classification carries; fusion dragged
        readout_bars/     plot.png data.csv   # best-epoch mean per readout, grouped by family
        rungD_summary.csv

Reads LIVE from runs/section_rungD/<cfg>/metric_curve.csv. Reference constants: native
recon-only ~72 (project baseline), Rung C labels-only 84.4, STgram-MFN 90.75.
"""
import argparse
import csv
import os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASSES = ["fan", "pump", "slider", "valve", "ToyCar", "ToyConveyor"]
OUT_ROOT = "docs/plots/mimii_section_rungD"
RECON_ONLY = 72.0     # native MambaAD recon-only baseline (project ~72)
RUNGC = 84.4          # labels-only Mamba student (Rung C best)
STGRAM = 90.75

# readout -> (family, color). recon=vermillion, classification=blue/green/orange, fusion=purple.
FAMILY = {
    "recon_spmax":   ("recon",   "#D55E00"),
    "recon_spmean":  ("recon",   "#E69F00"),
    "logit_nll":     ("classify", "#56B4E9"),
    "neg_cos":       ("classify", "#0072B2"),
    "maha_embed":    ("classify", "#009E73"),
    "fusion_negcos": ("fusion",  "#CC79A7"),
    "fusion_maha":   ("fusion",  "#7B3294"),
}
ORDER = ["recon_spmax", "recon_spmean", "logit_nll", "neg_cos", "maha_embed",
         "fusion_negcos", "fusion_maha"]
CURVE = ["recon_spmean", "neg_cos", "maha_embed", "fusion_maha"]
INK, MUTED, GRID = "#222222", "#666666", "#dddddd"


def read_curve(path):
    curve = defaultdict(dict)
    with open(path) as f:
        for r in csv.DictReader(f):
            ep = int(r["epoch"])
            curve[r["readout"]][ep] = {c: float(r[c]) * 100 for c in CLASSES}
            curve[r["readout"]][ep]["mean"] = float(r["mean"]) * 100
    return curve


def style_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)


def emit(name, fig, header, table):
    d = os.path.join(OUT_ROOT, name)
    os.makedirs(d, exist_ok=True)
    fig.savefig(os.path.join(d, "plot.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    with open(os.path.join(d, "data.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(table)
    print(f"[out] {d}/plot.png  +  data.csv")


def ref_lines(ax, xr):
    for lab, val, col in [(f"native recon-only ~{RECON_ONLY:.0f}", RECON_ONLY, "#D55E00"),
                          (f"Rung C labels-only {RUNGC:.1f}", RUNGC, "#009E73"),
                          (f"STgram-MFN {STGRAM:.1f}", STGRAM, "#666666")]:
        ax.axhline(val, color=col, lw=1.4, ls=":", zorder=2)
        ax.annotate(lab, (xr, val), xytext=(0, 3), textcoords="offset points",
                    ha="right", fontsize=8, color=col)


def fig_vs_epoch(curve):
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    style_axes(ax)
    for r in CURVE:
        if r not in curve:
            continue
        eps = sorted(curve[r])
        ax.plot(eps, [curve[r][e]["mean"] for e in eps], color=FAMILY[r][1],
                marker="o", ms=3, lw=2, zorder=3, label=r)
    xr = max(e for r in curve for e in curve[r])
    ref_lines(ax, xr)
    ax.set_xlabel("epoch", color=INK)
    ax.set_ylabel("mean image-level AUROC (%)", color=INK)
    ax.set_ylim(45, 92)
    ax.set_title("Rung D — joint distill+classify: the recon score collapses, fusion is dragged down",
                 fontsize=10.5, color=INK, pad=8)
    ax.legend(frameon=False, fontsize=9, loc="center right")
    readouts = [r for r in CURVE if r in curve]
    all_eps = sorted({e for r in readouts for e in curve[r]})
    header = ["epoch"] + [f"{r}_mean" for r in readouts]
    table = [[e] + [f"{curve[r][e]['mean']:.2f}" if e in curve[r] else "" for r in readouts]
             for e in all_eps]
    emit("auroc_vs_epoch", fig, header, table)


def fig_readout_bars(curve):
    """Best-epoch mean AUROC per readout, grouped by family."""
    best = {r: max(curve[r].values(), key=lambda d: d["mean"])["mean"] for r in ORDER if r in curve}
    bepoch = {r: max(curve[r], key=lambda e: curve[r][e]["mean"]) for r in ORDER if r in curve}
    labels = [r for r in ORDER if r in best]
    vals = [best[r] for r in labels]
    cols = [FAMILY[r][1] for r in labels]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 4.6))
    style_axes(ax)
    bars = ax.bar(x, vals, 0.62, color=cols, zorder=3)
    for b, r in zip(bars, labels):
        ax.annotate(f"{b.get_height():.1f}\n(ep{bepoch[r]})",
                    (b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8, color=INK)
    ref_lines(ax, len(labels) - 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("best-epoch mean AUROC (%)", color=INK)
    ax.set_ylim(45, 92)
    # family brackets in the title
    ax.set_title("Rung D readouts: classification carries it, reconstruction is ~chance, "
                 "fusion < classification alone", fontsize=10.5, color=INK, pad=8)
    emit("readout_bars", fig, ["readout", "best_epoch", "mean_auroc"],
         [[r, bepoch[r], f"{best[r]:.2f}"] for r in labels])


def write_summary(curve):
    os.makedirs(OUT_ROOT, exist_ok=True)
    path = os.path.join(OUT_ROOT, "rungD_summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["readout", "best_epoch"] + CLASSES + ["mean"])
        for r in ORDER:
            if r not in curve:
                continue
            be = max(curve[r], key=lambda e: curve[r][e]["mean"])
            d = curve[r][be]
            w.writerow([r, be] + [f"{d[c]:.2f}" for c in CLASSES] + [f"{d['mean']:.2f}"])
    print(f"[out] {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--cfg_name", default="log-Mel")
    args = ap.parse_args()
    curve_path = f"runs/section_rungD/{args.cfg_name}/metric_curve.csv"
    if not os.path.isfile(curve_path):
        raise SystemExit(f"missing {curve_path} — run Rung D first")
    curve = read_curve(curve_path)
    fig_vs_epoch(curve)
    fig_readout_bars(curve)
    write_summary(curve)


if __name__ == "__main__":
    main()
