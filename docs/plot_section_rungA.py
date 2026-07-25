"""Aggregate + plot Rung A of the "labels close the gap?" ladder
(diagnostics/section_classifier_probe.py).

Rung A = frozen ResNet34 + a trainable section-classifier head (ArcFace over the 23 MIMII
sections), scored the STgram way (fit-to-your-own-section). It isolates the *objective +
classification readout* with ZERO representation change (the encoder never moves), so its
result is read against two anchors:
  - frozen ResNet34 + Mahalanobis      (generative readout on the SAME frozen features)
  - STgram-MFN                          (supervised audio SOTA target / ceiling)

Sibling of docs/plot_backbone_ablation.py: cross-sectional bars (no epoch axis), each figure a
self-contained folder holding BOTH the PNG and the CSV behind it, plus one master summary CSV:

    docs/plots/mimii_section_rungA/
        rungA_summary.csv                    # every sub x readout x class
        per_class_grouped/  plot.png data.csv  # the complementary-readout story, per machine
        subcluster_sweep/   plot.png data.csv  # ArcFace sub-clusters are a non-lever

Numbers are read LIVE from runs/section_probe/sweep_sub{1,2,4}/auroc.csv (the variable part of
the experiment). STgram-MFN is a documented constant (same provenance as plot_backbone_ablation:
STgram-MFN/results/.../result.csv). The per-class figure uses sub=2 (the best-mean sweep point);
the sweep figure uses all three.
"""
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASSES = ["fan", "pump", "slider", "valve", "ToyCar", "ToyConveyor"]
OUT_ROOT = "docs/plots/mimii_section_rungA"
SUBS = [1, 2, 4]
BEST_SUB = 2  # max-mean sweep point, used for the per-class figure

# STgram-MFN supervised SOTA, per-class AUROC (%) — documented constant (see module docstring).
STGRAM_MFN = {
    "fan": 87.09, "pump": 90.94, "slider": 98.87, "valve": 98.59,
    "ToyCar": 94.72, "ToyConveyor": 74.27, "mean": 90.75,
}

# Okabe-Ito colorblind-safe palette, assigned in fixed order by the job each series does.
# anchor (frozen generative readout) -> blue; Rung-A discriminative readout -> orange;
# STgram target/ceiling -> bluish-green. Validated pairwise-distinct under CVD.
C_ANCHOR = "#0072B2"   # maha_concat_raw  (frozen ResNet34 + Maha)
C_RUNGA  = "#E69F00"   # neg_cos          (frozen + classification readout)
C_TARGET = "#009E73"   # STgram-MFN
INK, MUTED, GRID = "#222222", "#666666", "#dddddd"


def read_probe_csv(path):
    """runs/section_probe/sweep_sub<K>/auroc.csv -> {readout: {class: auroc, 'mean': mean}}."""
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            out[r["readout"]] = {c: float(r[c]) for c in CLASSES}
            out[r["readout"]]["mean"] = float(r["mean_AUROC"])
    return out


def emit(name, fig, header, table):
    """Write a figure folder: <OUT_ROOT>/<name>/{plot.png, data.csv}. Mirrors the siblings."""
    d = os.path.join(OUT_ROOT, name)
    os.makedirs(d, exist_ok=True)
    fig.savefig(os.path.join(d, "plot.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    with open(os.path.join(d, "data.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(table)
    print(f"[out] {d}/plot.png  +  data.csv")


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------- #
def fig_per_class(data_best):
    """Grouped bars: frozen+Maha vs Rung-A classification vs STgram target, per machine + mean."""
    anchor = data_best["maha_concat_raw"]
    runga = data_best["neg_cos"]
    cats = CLASSES + ["mean"]
    x = np.arange(len(cats))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    style_axes(ax)

    b1 = ax.bar(x - w, [anchor[c] for c in cats], w, color=C_ANCHOR, zorder=3,
                label="frozen ResNet34 + Maha (generative)")
    b2 = ax.bar(x,      [runga[c] for c in cats], w, color=C_RUNGA, zorder=3,
                label="Rung A: frozen + classification (neg-cos)")
    b3 = ax.bar(x + w, [STGRAM_MFN[c] for c in cats], w, color=C_TARGET, zorder=3,
                label="STgram-MFN (supervised target)")

    # separator + emphasis for the mean group; direct-label ONLY the headline means
    ax.axvline(len(CLASSES) - 0.5, color=MUTED, lw=0.8, ls=":", zorder=1)
    for b in (b1, b2, b3):
        h = b[-1].get_height()
        ax.annotate(f"{h:.1f}", (b[-1].get_x() + b[-1].get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, color=INK, fontweight="bold")

    ax.set_ylim(50, 100)
    ax.set_ylabel("image-level AUROC (%)", color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=9)
    ax.get_xticklabels()[-1].set_fontweight("bold")
    ax.set_title("Rung A — a classification readout rescues Maha's weak classes, "
                 "but head-only leaves a gap to STgram",
                 fontsize=11, color=INK, pad=10)
    ax.legend(frameon=False, fontsize=8.5, loc="lower center",
              bbox_to_anchor=(0.5, -0.30), ncol=3)

    header = ["class", "frozen_maha", "rungA_negcos", "stgram_mfn", "rungA_minus_maha"]
    table = [[c, f"{anchor[c]:.2f}", f"{runga[c]:.2f}", f"{STGRAM_MFN[c]:.2f}",
              f"{runga[c] - anchor[c]:+.2f}"] for c in cats]
    emit("per_class_grouped", fig, header, table)


def fig_subcluster(per_sub):
    """ArcFace sub-clusters {1,2,4} vs mean AUROC for each readout — the non-lever panel."""
    readouts = [("neg_cos", C_RUNGA, "o", "-"),
                ("logit_nll", "#CC79A7", "s", "-"),
                ("maha_embed", "#56B4E9", "^", "--")]
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    style_axes(ax)
    xs = np.array(SUBS)

    for name, color, mk, ls in readouts:
        ys = [per_sub[s][name]["mean"] for s in SUBS]
        ax.plot(xs, ys, ls, color=color, marker=mk, ms=8, lw=2, zorder=3, label=name)

    # reference lines: frozen+Maha anchor (constant across sub) and STgram ceiling
    anchor = per_sub[BEST_SUB]["maha_concat_raw"]["mean"]
    ax.axhline(anchor, color=C_ANCHOR, lw=1.6, ls=":", zorder=2)
    ax.annotate(f"frozen+Maha anchor  {anchor:.1f}", (xs[-1], anchor),
                xytext=(0, -13), textcoords="offset points", ha="right",
                fontsize=8.5, color=C_ANCHOR)
    ax.axhline(STGRAM_MFN["mean"], color=C_TARGET, lw=1.6, ls=":", zorder=2)
    ax.annotate(f"STgram-MFN target  {STGRAM_MFN['mean']:.1f}", (xs[0], STGRAM_MFN["mean"]),
                xytext=(0, 4), textcoords="offset points", ha="left",
                fontsize=8.5, color=C_TARGET)

    ax.set_xticks(SUBS)
    ax.set_xlabel("ArcFace sub-clusters per section", color=INK)
    ax.set_ylabel("mean image-level AUROC (%)", color=INK)
    ax.set_ylim(60, 92)
    ax.set_title("Sub-clusters are a non-lever\n(head saturates ~80; ~+5 to STgram remains)",
                 fontsize=11, color=INK, pad=8)
    ax.legend(frameon=False, fontsize=9, loc="center right")

    header = ["sub"] + [r[0] for r in readouts] + ["maha_concat_raw"]
    table = [[s] + [f"{per_sub[s][r[0]]['mean']:.2f}" for r in readouts]
             + [f"{per_sub[s]['maha_concat_raw']['mean']:.2f}"] for s in SUBS]
    emit("subcluster_sweep", fig, header, table)


def write_summary(per_sub):
    os.makedirs(OUT_ROOT, exist_ok=True)
    path = os.path.join(OUT_ROOT, "rungA_summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sub", "readout"] + CLASSES + ["mean"])
        for s in SUBS:
            for readout, vals in per_sub[s].items():
                w.writerow([s, readout] + [f"{vals[c]:.2f}" for c in CLASSES] + [f"{vals['mean']:.2f}"])
    print(f"[out] {path}")


def main():
    per_sub = {}
    for s in SUBS:
        p = f"runs/section_probe/sweep_sub{s}/auroc.csv"
        if not os.path.isfile(p):
            raise SystemExit(f"missing {p} — run the Rung-A sweep first")
        per_sub[s] = read_probe_csv(p)
    fig_per_class(per_sub[BEST_SUB])
    fig_subcluster(per_sub)
    write_summary(per_sub)


if __name__ == "__main__":
    main()
