"""Aggregate + plot Rung E of the "labels close the gap?" ladder
(diagnostics/section_combined_rungE.py).

Rung E = B+C combined: fine-tune the ResNet teacher AND the Mamba student together under the
section loss. Three figures capture the two-framings story (does encoder adaptation stack with
the Mamba decoder?):

    docs/plots/mimii_section_rungE/
        auroc_vs_epoch/    plot.png data.csv   # neg_cos vs maha_embed, with B/C/STgram ref lines
        maha_stacking/     plot.png data.csv   # A->B->C->E on a FIXED readout (maha): the levers stack
        per_class_best/    plot.png data.csv   # best-model-per-rung B/C/E/STgram: E rescues slider,
                                               # residual moves to ToyCar/ToyConveyor
        rungE_summary.csv

Reads Rung E LIVE from runs/section_rungE/<cfg>/metric_curve.csv. A/B/C and STgram are
documented constants from their own CONCLUSION.md figures.
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
OUT_ROOT = "docs/plots/mimii_section_rungE"

# ---- documented constants (from Rung A/B/C CONCLUSION.md) ----
MAHA_BY_RUNG = {"A frozen": 67.0, "B FT enc": 81.8, "C Mamba": 84.4}   # maha_embed best mean
RUNGB_BEST = {"fan": 82.77, "pump": 87.53, "slider": 85.08, "valve": 94.58,
              "ToyCar": 92.82, "ToyConveyor": 72.41, "mean": 85.86}   # neg_cos ep16 (B's best)
RUNGC_BEST = {"fan": 83.09, "pump": 86.16, "slider": 93.06, "valve": 88.04,
              "ToyCar": 88.41, "ToyConveyor": 67.84, "mean": 84.43}   # maha ep9 (C's best)
STGRAM = {"fan": 87.09, "pump": 90.94, "slider": 98.87, "valve": 98.59,
          "ToyCar": 94.72, "ToyConveyor": 74.27, "mean": 90.75}
RUNGB_MEAN, RUNGC_MEAN = 85.86, 84.43

C_A, C_B, C_C, C_E, C_T = "#E69F00", "#D55E00", "#CC79A7", "#7B3294", "#009E73"
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


def fig_vs_epoch(curve):
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    style_axes(ax)
    for r, col in [("maha_embed", C_E), ("neg_cos", "#0072B2")]:
        if r not in curve:
            continue
        eps = sorted(curve[r])
        ax.plot(eps, [curve[r][e]["mean"] for e in eps], color=col, marker="o", ms=3, lw=2,
                zorder=3, label=r)
        pe = max(eps, key=lambda e: curve[r][e]["mean"])
        ax.scatter([pe], [curve[r][pe]["mean"]], s=90, facecolors="none", edgecolors=col, lw=2, zorder=4)
    xr = max(e for r in curve for e in curve[r])
    for lab, val, col in [(f"Rung B FT enc {RUNGB_MEAN:.1f}", RUNGB_MEAN, C_B),
                          (f"Rung C Mamba {RUNGC_MEAN:.1f}", RUNGC_MEAN, C_C),
                          (f"STgram-MFN {STGRAM['mean']:.1f}", STGRAM["mean"], C_T)]:
        ax.axhline(val, color=col, lw=1.4, ls=":", zorder=2)
        ax.annotate(lab, (xr, val), xytext=(0, 3), textcoords="offset points", ha="right",
                    fontsize=8.2, color=col)
    mp = max(curve["maha_embed"], key=lambda e: curve["maha_embed"][e]["mean"])
    ax.set_xlabel("epoch", color=INK)
    ax.set_ylabel("mean image-level AUROC (%)", color=INK)
    ax.set_title(f"Rung E — fine-tune encoder + Mamba student  (maha_embed peak "
                 f"{curve['maha_embed'][mp]['mean']:.1f} @ ep{mp})", fontsize=10.5, color=INK, pad=8)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    readouts = [r for r in ("maha_embed", "neg_cos") if r in curve]
    all_eps = sorted({e for r in readouts for e in curve[r]})
    emit("auroc_vs_epoch", fig, ["epoch"] + [f"{r}_mean" for r in readouts],
         [[e] + [f"{curve[r][e]['mean']:.2f}" if e in curve[r] else "" for r in readouts] for e in all_eps])


def fig_maha_stacking(curve):
    """A->B->C->E on the FIXED maha_embed readout: do the levers stack?"""
    mp = max(curve["maha_embed"], key=lambda e: curve["maha_embed"][e]["mean"])
    e_maha = curve["maha_embed"][mp]["mean"]
    labels = list(MAHA_BY_RUNG) + ["E FT+Mamba"]
    vals = list(MAHA_BY_RUNG.values()) + [e_maha]
    cols = [C_A, C_B, C_C, C_E]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    style_axes(ax)
    bars = ax.bar(x, vals, 0.6, color=cols, zorder=3)
    for b in bars:
        ax.annotate(f"{b.get_height():.1f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                    fontsize=9, color=INK, fontweight="bold")
    # incremental deltas
    for i in range(1, len(vals)):
        ax.annotate(f"+{vals[i]-vals[i-1]:.1f}", (x[i] - 0.5, max(vals[i], vals[i-1]) + 2),
                    ha="center", fontsize=8, color=MUTED)
    ax.axhline(STGRAM["mean"], color=C_T, lw=1.4, ls=":", zorder=2)
    ax.annotate(f"STgram {STGRAM['mean']:.1f}", (len(labels) - 0.5, STGRAM["mean"]),
                xytext=(0, 3), textcoords="offset points", ha="right", fontsize=8.2, color=C_T)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(60, 92)
    ax.set_ylabel("maha_embed mean AUROC (%)", color=INK)
    ax.set_title("On a fixed readout the levers stack:\nencoder FT and the Mamba decoder each add",
                 fontsize=10.5, color=INK, pad=8)
    emit("maha_stacking", fig, ["rung", "maha_embed_mean"],
         [[l, f"{v:.2f}"] for l, v in zip(labels, vals)])


def fig_per_class_best(curve):
    """Best model per rung (B neg_cos, C maha, E maha) vs STgram, per machine."""
    mp = max(curve["maha_embed"], key=lambda e: curve["maha_embed"][e]["mean"])
    runge = curve["maha_embed"][mp]
    cats = CLASSES + ["mean"]
    x = np.arange(len(cats))
    w = 0.2
    fig, ax = plt.subplots(figsize=(10, 4.7))
    style_axes(ax)
    series = [("Rung B (FT enc, neg_cos)", RUNGB_BEST, C_B),
              ("Rung C (Mamba, maha)", RUNGC_BEST, C_C),
              (f"Rung E (FT+Mamba, maha, ep{mp})", runge, C_E),
              ("STgram-MFN", STGRAM, C_T)]
    offs = np.linspace(-1.5, 1.5, len(series)) * w
    bars = []
    for (lab, d, col), off in zip(series, offs):
        bars.append(ax.bar(x + off, [d[c] for c in cats], w, color=col, zorder=3, label=lab))
    ax.axvline(len(CLASSES) - 0.5, color=MUTED, lw=0.8, ls=":", zorder=1)
    for b in bars:
        h = b[-1].get_height()
        ax.annotate(f"{h:.1f}", (b[-1].get_x() + b[-1].get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8,
                    color=INK, fontweight="bold")
    ax.set_ylim(50, 100)
    ax.set_ylabel("image-level AUROC (%)", color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=9)
    ax.get_xticklabels()[-1].set_fontweight("bold")
    ax.set_title("Rung E is the new best — Mamba decoder closes slider; residual moves to ToyCar/ToyConveyor",
                 fontsize=10, color=INK, pad=10)
    ax.legend(frameon=False, fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=4)
    emit("per_class_best", fig, ["class", "rungB", "rungC", "rungE", "stgram"],
         [[c, f"{RUNGB_BEST[c]:.2f}", f"{RUNGC_BEST[c]:.2f}", f"{runge[c]:.2f}", f"{STGRAM[c]:.2f}"]
          for c in cats])


def write_summary(curve):
    os.makedirs(OUT_ROOT, exist_ok=True)
    path = os.path.join(OUT_ROOT, "rungE_summary.csv")
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
    curve_path = f"runs/section_rungE/{args.cfg_name}/metric_curve.csv"
    if not os.path.isfile(curve_path):
        raise SystemExit(f"missing {curve_path} — run Rung E first")
    curve = read_curve(curve_path)
    fig_vs_epoch(curve)
    fig_maha_stacking(curve)
    fig_per_class_best(curve)
    write_summary(curve)


if __name__ == "__main__":
    main()
