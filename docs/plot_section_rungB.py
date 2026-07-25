"""Aggregate + plot Rung B of the "labels close the gap?" ladder
(diagnostics/section_finetune_rungB.py).

Rung B = UNFREEZE ResNet34 and fine-tune it end-to-end under the same ArcFace section loss.
The only change vs Rung A is frozen -> trainable encoder, so the delta over Rung A's frozen
ceiling isolates *representation adaptation*. Two figures, same folder convention as
docs/plot_section_rungA.py / plot_backbone_ablation.py:

    docs/plots/mimii_section_rungB/
        auroc_vs_epoch/   plot.png data.csv   # THE headline — peaks early, so show the curve
        per_class_best/   plot.png data.csv   # ladder at Rung B's best epoch, per machine
        rungB_summary.csv                     # best-epoch per-class, every readout

Reads LIVE from runs/section_rungB/<cfg>/metric_curve.csv (+ train_log.csv). Rung A anchors
(frozen+Maha, frozen+classification) and STgram-MFN are documented constants matching the
Rung A figures / plot_backbone_ablation.py. Why the curve is the headline: this repo's rule
is MIMII AUROC PEAKS EARLY with no best-ckpt selection, so the honest comparison is Rung A
vs Rung B *at their peaks*, and the curve makes the peak (and any late decline) visible.
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
OUT_ROOT = "docs/plots/mimii_section_rungB"

# ---- documented constants (same provenance as the Rung A figures) ----
FROZEN_MAHA = {"fan": 58.44, "pump": 72.06, "slider": 90.78, "valve": 70.02,
               "ToyCar": 75.68, "ToyConveyor": 63.79, "mean": 71.80}
RUNGA_NEGCOS = {"fan": 76.10, "pump": 85.27, "slider": 81.46, "valve": 90.76,
                "ToyCar": 83.61, "ToyConveyor": 62.84, "mean": 80.01}  # sub=2, frozen+classification
STGRAM_MFN = {"fan": 87.09, "pump": 90.94, "slider": 98.87, "valve": 98.59,
              "ToyCar": 94.72, "ToyConveyor": 74.27, "mean": 90.75}

# Okabe-Ito, assigned by job. anchor=blue, RungA=orange, RungB=vermillion, STgram=green.
C_ANCHOR, C_RUNGA, C_RUNGB, C_TARGET = "#0072B2", "#E69F00", "#D55E00", "#009E73"
INK, MUTED, GRID = "#222222", "#666666", "#dddddd"
READOUT_STYLE = {  # for the vs-epoch curve
    "neg_cos":    dict(color=C_RUNGB,   marker="o", ls="-"),
    "logit_nll":  dict(color="#CC79A7", marker="s", ls="-"),
    "maha_embed": dict(color="#56B4E9", marker="^", ls="--"),
}


def read_curve(path):
    """metric_curve.csv -> {readout: {epoch: {class:auroc, 'mean':mean}}} (values in %)."""
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


# --------------------------------------------------------------------------- #
def fig_vs_epoch(curve):
    """Mean AUROC vs epoch for each readout — shows the peak (peaks-early rule)."""
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

    ax.axhline(RUNGA_NEGCOS["mean"], color=C_RUNGA, lw=1.6, ls=":", zorder=2)
    ax.annotate(f"Rung A frozen ceiling  {RUNGA_NEGCOS['mean']:.1f}",
                (max(next(iter(curve.values()))), RUNGA_NEGCOS["mean"]),
                xytext=(0, 4), textcoords="offset points", ha="right", fontsize=8.5, color=C_RUNGA)
    ax.axhline(STGRAM_MFN["mean"], color=C_TARGET, lw=1.6, ls=":", zorder=2)
    ax.annotate(f"STgram-MFN target  {STGRAM_MFN['mean']:.1f}",
                (1, STGRAM_MFN["mean"]), xytext=(0, 4), textcoords="offset points",
                ha="left", fontsize=8.5, color=C_TARGET)

    ax.set_xlabel("epoch", color=INK)
    ax.set_ylabel("mean image-level AUROC (%)", color=INK)
    ncstr = f"  (neg_cos peak {peak['neg_cos'][1]:.1f} @ ep{peak['neg_cos'][0]})" if "neg_cos" in peak else ""
    ax.set_title("Rung B — fine-tuning the encoder under the section loss" + ncstr,
                 fontsize=11, color=INK, pad=8)
    ax.legend(frameon=False, fontsize=9, loc="lower right")

    readouts = [r for r in READOUT_STYLE if r in curve]
    all_eps = sorted({e for r in readouts for e in curve[r]})
    header = ["epoch"] + [f"{r}_mean" for r in readouts]
    table = [[e] + [f"{curve[r][e]['mean']:.2f}" if e in curve[r] else "" for r in readouts]
             for e in all_eps]
    emit("auroc_vs_epoch", fig, header, table)
    return peak


def fig_per_class_best(curve, peak):
    """Grouped bars at Rung B's best neg_cos epoch: the full ladder per machine."""
    be = peak["neg_cos"][0]
    rungb = curve["neg_cos"][be]
    cats = CLASSES + ["mean"]
    x = np.arange(len(cats))
    w = 0.20
    fig, ax = plt.subplots(figsize=(10, 4.7))
    style_axes(ax)
    series = [("frozen+Maha", FROZEN_MAHA, C_ANCHOR),
              ("Rung A (frozen+cls)", RUNGA_NEGCOS, C_RUNGA),
              (f"Rung B (FT, ep{be})", rungb, C_RUNGB),
              ("STgram-MFN", STGRAM_MFN, C_TARGET)]
    offs = np.linspace(-1.5, 1.5, len(series)) * w
    bars = []
    for (lab, d, col), off in zip(series, offs):
        bars.append(ax.bar(x + off, [d[c] for c in cats], w, color=col, zorder=3, label=lab))
    ax.axvline(len(CLASSES) - 0.5, color=MUTED, lw=0.8, ls=":", zorder=1)
    for b in bars:  # label the mean cluster only
        h = b[-1].get_height()
        ax.annotate(f"{h:.1f}", (b[-1].get_x() + b[-1].get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                    fontsize=8, color=INK, fontweight="bold")
    ax.set_ylim(50, 100)
    ax.set_ylabel("image-level AUROC (%)", color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=9)
    ax.get_xticklabels()[-1].set_fontweight("bold")
    ax.set_title("The ladder at Rung B's peak — how far fine-tuning closes the STgram gap",
                 fontsize=11, color=INK, pad=10)
    ax.legend(frameon=False, fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=4)

    header = ["class", "frozen_maha", "rungA_negcos", "rungB_negcos", "stgram_mfn"]
    table = [[c, f"{FROZEN_MAHA[c]:.2f}", f"{RUNGA_NEGCOS[c]:.2f}", f"{rungb[c]:.2f}",
              f"{STGRAM_MFN[c]:.2f}"] for c in cats]
    emit("per_class_best", fig, header, table)


def write_summary(curve):
    os.makedirs(OUT_ROOT, exist_ok=True)
    path = os.path.join(OUT_ROOT, "rungB_summary.csv")
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
    curve_path = f"runs/section_rungB/{args.cfg_name}/metric_curve.csv"
    if not os.path.isfile(curve_path):
        raise SystemExit(f"missing {curve_path} — run Rung B first")
    curve = read_curve(curve_path)
    peak = fig_vs_epoch(curve)
    fig_per_class_best(curve, peak)
    write_summary(curve)


if __name__ == "__main__":
    main()
