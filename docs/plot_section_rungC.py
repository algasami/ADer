"""Aggregate + plot Rung C of the "labels close the gap?" ladder
(diagnostics/section_mamba_rungC.py).

Rung C = MambaAD's own student (frozen ResNet34 teacher -> MFF/OCE -> MambaUPNet decoder)
trained to CLASSIFY the 23 sections instead of to reconstruct teacher features. It is the
first rung that exercises the Mamba architecture. Same folder/figure convention as
docs/plot_section_rungB.py:

    docs/plots/mimii_section_rungC/
        auroc_vs_epoch/   plot.png data.csv   # peaks-early curve, with A/B/STgram ref lines
        per_class_best/   plot.png data.csv   # ladder A -> B -> C -> STgram, per machine
        rungC_summary.csv

Reads LIVE from runs/section_rungC/<cfg>/metric_curve.csv. Rung A/B and STgram are documented
constants matching their own CONCLUSION.md figures.
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
OUT_ROOT = "docs/plots/mimii_section_rungC"

# ---- documented constants (from the Rung A/B CONCLUSION.md figures) ----
RUNGA = {"fan": 76.10, "pump": 85.27, "slider": 81.46, "valve": 90.76,
         "ToyCar": 83.61, "ToyConveyor": 62.84, "mean": 80.01}   # frozen + classification
RUNGB = {"fan": 82.77, "pump": 87.53, "slider": 85.08, "valve": 94.58,
         "ToyCar": 92.82, "ToyConveyor": 72.41, "mean": 85.86}   # fine-tuned ResNet, ep16
STGRAM = {"fan": 87.09, "pump": 90.94, "slider": 98.87, "valve": 98.59,
          "ToyCar": 94.72, "ToyConveyor": 74.27, "mean": 90.75}

# Okabe-Ito by role: RungA=orange, RungB=vermillion, RungC=reddish-purple, STgram=green.
C_RUNGA, C_RUNGB, C_RUNGC, C_TARGET = "#E69F00", "#D55E00", "#CC79A7", "#009E73"
INK, MUTED, GRID = "#222222", "#666666", "#dddddd"
READOUT_STYLE = {
    "neg_cos":    dict(color=C_RUNGC,   marker="o", ls="-"),
    "logit_nll":  dict(color="#0072B2", marker="s", ls="-"),
    "maha_embed": dict(color="#56B4E9", marker="^", ls="--"),
}


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


def fig_vs_epoch(curve):
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    style_axes(ax)
    peak = {}
    for r, st in READOUT_STYLE.items():
        if r not in curve:
            continue
        eps = sorted(curve[r])
        ys = [curve[r][e]["mean"] for e in eps]
        ax.plot(eps, ys, color=st["color"], marker=st["marker"], ms=4, lw=2, ls=st["ls"],
                zorder=3, label=r)
        pe = max(eps, key=lambda e: curve[r][e]["mean"])
        peak[r] = (pe, curve[r][pe]["mean"])
        ax.scatter([pe], [curve[r][pe]["mean"]], s=90, facecolors="none",
                   edgecolors=st["color"], lw=2, zorder=4)
    xmax = max(e for r in curve for e in curve[r])
    for lab, val, col in [("Rung A frozen  {:.1f}".format(RUNGA["mean"]), RUNGA["mean"], C_RUNGA),
                          ("Rung B FT enc  {:.1f}".format(RUNGB["mean"]), RUNGB["mean"], C_RUNGB),
                          ("STgram-MFN  {:.1f}".format(STGRAM["mean"]), STGRAM["mean"], C_TARGET)]:
        ax.axhline(val, color=col, lw=1.5, ls=":", zorder=2)
        ax.annotate(lab, (xmax, val), xytext=(0, 3), textcoords="offset points",
                    ha="right", fontsize=8.5, color=col)
    ax.set_xlabel("epoch", color=INK)
    ax.set_ylabel("mean image-level AUROC (%)", color=INK)
    br = max(peak, key=lambda r: peak[r][1]) if peak else None
    ncstr = f"  (best {br} {peak[br][1]:.1f} @ ep{peak[br][0]})" if br else ""
    ax.set_title("Rung C — Mamba student trained to classify sections" + ncstr,
                 fontsize=11, color=INK, pad=8)
    ax.legend(frameon=False, fontsize=9, loc="lower right")

    readouts = [r for r in READOUT_STYLE if r in curve]
    all_eps = sorted({e for r in readouts for e in curve[r]})
    header = ["epoch"] + [f"{r}_mean" for r in readouts]
    table = [[e] + [f"{curve[r][e]['mean']:.2f}" if e in curve[r] else "" for r in readouts]
             for e in all_eps]
    emit("auroc_vs_epoch", fig, header, table)
    return peak


def fig_per_class_best(curve, peak, best_r):
    be = peak[best_r][0]
    rungc = curve[best_r][be]
    cats = CLASSES + ["mean"]
    x = np.arange(len(cats))
    w = 0.20
    fig, ax = plt.subplots(figsize=(10, 4.7))
    style_axes(ax)
    series = [("Rung A (frozen+cls)", RUNGA, C_RUNGA),
              ("Rung B (FT ResNet)", RUNGB, C_RUNGB),
              (f"Rung C (Mamba {best_r}, ep{be})", rungc, C_RUNGC),
              ("STgram-MFN", STGRAM, C_TARGET)]
    offs = np.linspace(-1.5, 1.5, len(series)) * w
    bars = []
    for (lab, d, col), off in zip(series, offs):
        bars.append(ax.bar(x + off, [d[c] for c in cats], w, color=col, zorder=3, label=lab))
    ax.axvline(len(CLASSES) - 0.5, color=MUTED, lw=0.8, ls=":", zorder=1)
    for b in bars:
        h = b[-1].get_height()
        ax.annotate(f"{h:.1f}", (b[-1].get_x() + b[-1].get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                    fontsize=8, color=INK, fontweight="bold")
    ax.set_ylim(50, 100)
    ax.set_ylabel("image-level AUROC (%)", color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=9)
    ax.get_xticklabels()[-1].set_fontweight("bold")
    ax.set_title("The ladder at Rung C's peak — does the Mamba decoder beat fine-tuning?",
                 fontsize=11, color=INK, pad=10)
    ax.legend(frameon=False, fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=4)

    header = ["class", "rungA", "rungB", "rungC", "stgram"]
    table = [[c, f"{RUNGA[c]:.2f}", f"{RUNGB[c]:.2f}", f"{rungc[c]:.2f}", f"{STGRAM[c]:.2f}"]
             for c in cats]
    emit("per_class_best", fig, header, table)


def write_summary(curve):
    os.makedirs(OUT_ROOT, exist_ok=True)
    path = os.path.join(OUT_ROOT, "rungC_summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["readout", "best_epoch"] + CLASSES + ["mean"])
        for r in curve:
            be = max(curve[r], key=lambda e: curve[r][e]["mean"])
            d = curve[r][be]
            w.writerow([r, be] + [f"{d[c]:.2f}" for c in CLASSES] + [f"{d['mean']:.2f}"])
    print(f"[out] {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--cfg_name", default="log-Mel")
    args = ap.parse_args()
    curve_path = f"runs/section_rungC/{args.cfg_name}/metric_curve.csv"
    if not os.path.isfile(curve_path):
        raise SystemExit(f"missing {curve_path} — run Rung C first")
    curve = read_curve(curve_path)
    peak = fig_vs_epoch(curve)
    best_r = max(peak, key=lambda r: peak[r][1])  # best readout by its peak mean
    fig_per_class_best(curve, peak, best_r)
    write_summary(curve)


if __name__ == "__main__":
    main()
