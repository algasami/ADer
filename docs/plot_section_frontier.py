"""ToyCar / ToyConveyor frontier analysis — why the labels-ladder residual concentrates here.

Rung E (docs/plots/mimii_section_rungE/CONCLUSION.md) closed the labels/objective ladder at
86.9 mean AUROC, -3.9 to STgram-MFN (90.75), with the residual gap now concentrated in two
classes: ToyCar (-9.2) and ToyConveyor (-5.7) — everything else (fan/pump/slider/valve) is
within ~1-2 pts of STgram. This script answers "why these two, and is there headroom left":

    docs/plots/mimii_section_frontier/
        ladder_by_class/     plot.png data.csv   # ToyCar/ToyConveyor across every rung of the
                                                  # ladder (native -> A -> B -> C -> D -> E ->
                                                  # STgram) using each rung's OWN winning readout
        stgram_per_id/        plot.png data.csv  # STgram-MFN's own per-machine-ID AUROC for
                                                  # these two classes: is the ceiling itself
                                                  # dataset-intrinsic, not a MambaAD-track failure?
        cross_rung_oracle/    plot.png data.csv  # per-class BEST across all 5 rungs x all
                                                  # readouts (a val-selectable oracle, not a
                                                  # single trained model) vs the realized Rung E
                                                  # and vs STgram

All ladder numbers are documented constants pulled from each rung's own CONCLUSION.md /
*_summary.csv (see root CLAUDE.md: MIMII AUROC peaks early, so these are each rung's own peak
epoch under its own winning readout, not a shared checkpoint). The "native" row is the
cos-residual sp_max avg-peak (ep10) from runs/..._toy_20260722-215044/metric.txt (the e50 x
log-Mel run used throughout the scorer/scan ablations) — no section labels involved.
STgram-MFN per-ID numbers are read straight from STgram-MFN/results/STgram-MFN(m=0.7,s=30)/result.csv.
"""
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_ROOT = "docs/plots/mimii_section_frontier"
CLASSES = ["fan", "pump", "slider", "valve", "ToyCar", "ToyConveyor"]

# ---- ladder, ToyCar/ToyConveyor only, each rung's own winning readout (documented) ----
LADDER = [
    ("native\n(recon)",  61.6, 61.5, "no labels; cos-residual sp_max, avg-peak ep10"),
    ("A\nfrozen+cls",    83.22, 62.96, "neg_cos, winning readout (mean 80.20)"),
    ("D\njoint",         82.89, 68.76, "maha_embed, winning readout (mean 83.98)"),
    ("C\nMamba+cls",     88.41, 67.84, "maha_embed, winning readout (mean 84.43)"),
    ("B\nFT encoder",    92.82, 72.41, "neg_cos, winning readout (mean 85.86)"),
    ("E\nFT enc+Mamba",  85.48, 68.57, "maha_embed, winning readout (mean 86.85)"),
    ("STgram-MFN",       94.72, 74.27, "supervised SOTA baseline"),
]

# ---- STgram-MFN per-machine-ID (STgram-MFN/results/.../result.csv) ----
STGRAM_IDS = {
    "ToyCar":      [("id_01", 83.75), ("id_02", 95.66), ("id_03", 99.47), ("id_04", 99.99)],
    "ToyConveyor": [("id_01", 85.51), ("id_02", 62.44), ("id_03", 74.85)],
}

# ---- cross-rung oracle: best value for each class across ALL rungs x ALL readouts ----
# (source: docs/plots/mimii_section_rung{A,B,C,D,E}/rung*_summary.csv, max over every row)
ORACLE = {"fan": 84.54, "pump": 89.66, "slider": 96.57, "valve": 96.85,
          "ToyCar": 92.82, "ToyConveyor": 72.41}
ORACLE_SRC = {"fan": "E maha", "pump": "E neg_cos", "slider": "E maha", "valve": "E maha",
              "ToyCar": "B neg_cos", "ToyConveyor": "B neg_cos"}
RUNGE_BEST = {"fan": 84.54, "pump": 89.09, "slider": 96.57, "valve": 96.85,
              "ToyCar": 85.48, "ToyConveyor": 68.57, "mean": 86.85}  # rung E's OWN maha_embed row
STGRAM = {"fan": 87.09, "pump": 90.94, "slider": 98.87, "valve": 98.59,
          "ToyCar": 94.72, "ToyConveyor": 74.27, "mean": 90.75}

C_CAR, C_CONV, C_T, C_ORACLE, C_E = "#0072B2", "#D55E00", "#009E73", "#7B3294", "#CC79A7"
INK, MUTED, GRID = "#222222", "#666666", "#dddddd"


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


def fig_ladder_by_class():
    labels = [l for l, *_ in LADDER]
    car = [v for _, v, *_ in LADDER]
    conv = [v for _, _, v, _ in LADDER]
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 4.8))
    style_axes(ax)
    b1 = ax.bar(x - w / 2, car, w, color=C_CAR, zorder=3, label="ToyCar")
    b2 = ax.bar(x + w / 2, conv, w, color=C_CONV, zorder=3, label="ToyConveyor")
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"{b.get_height():.1f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                        fontsize=8, color=INK, fontweight="bold")
    ax.axvline(len(labels) - 1.5, color=MUTED, lw=0.8, ls=":", zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(50, 102)
    ax.set_ylabel("image-level AUROC (%)", color=INK)
    ax.set_title("ToyCar peaks at Rung B, then DROPS at Rung E — the Mamba decoder trades\n"
                 "ToyCar away while rescuing slider/valve; ToyConveyor never breaks 73",
                 fontsize=10.5, color=INK, pad=10)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    emit("ladder_by_class", fig, ["rung", "ToyCar", "ToyConveyor", "note"],
         [[l.replace("\n", " "), f"{c:.2f}", f"{v:.2f}", n] for l, c, v, n in LADDER])


def fig_stgram_per_id():
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharey=True)
    for ax, (cls, col) in zip(axes, [("ToyCar", C_CAR), ("ToyConveyor", C_CONV)]):
        style_axes(ax)
        ids, vals = zip(*STGRAM_IDS[cls])
        x = np.arange(len(ids))
        bars = ax.bar(x, vals, 0.55, color=col, zorder=3)
        avg = np.mean(vals)
        ax.axhline(avg, color=INK, lw=1.2, ls="--", zorder=2)
        ax.annotate(f"class avg {avg:.1f}", (len(ids) - 0.5, avg), xytext=(0, 3),
                    textcoords="offset points", ha="right", fontsize=8, color=INK)
        for b in bars:
            ax.annotate(f"{b.get_height():.1f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                        fontsize=8.5, color=INK, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(ids, fontsize=9)
        ax.set_title(f"STgram-MFN — {cls}", fontsize=10, color=INK)
    axes[0].set_ylabel("image-level AUROC (%)", color=INK)
    fig.suptitle("Even the SOTA baseline is not uniformly strong: ToyConveyor id_02 (62.4) and\n"
                 "ToyCar id_01 (83.8) are structurally hard — one weak ID drags each class average",
                 fontsize=10.5, color=INK, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    rows = [[cls, i, f"{v:.2f}"] for cls in STGRAM_IDS for i, v in STGRAM_IDS[cls]]
    emit("stgram_per_id", fig, ["class", "id", "auroc"], rows)


def fig_cross_rung_oracle():
    cats = CLASSES + ["mean"]
    oracle = {**ORACLE, "mean": float(np.mean(list(ORACLE.values())))}
    x = np.arange(len(cats))
    w = 0.25
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    style_axes(ax)
    series = [("Rung E (realized, single model)", RUNGE_BEST, C_E),
              ("Cross-rung oracle (best of A-E, any readout)", oracle, C_ORACLE),
              ("STgram-MFN", STGRAM, C_T)]
    offs = np.linspace(-1, 1, len(series)) * w
    for (lab, d, col), off in zip(series, offs):
        bars = ax.bar(x + off, [d[c] for c in cats], w, color=col, zorder=3, label=lab)
        b = bars[-1]
        ax.annotate(f"{b.get_height():.1f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                    fontsize=8, color=INK, fontweight="bold")
    ax.axvline(len(CLASSES) - 0.5, color=MUTED, lw=0.8, ls=":", zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=9)
    ax.get_xticklabels()[-1].set_fontweight("bold")
    ax.set_ylim(50, 102)
    ax.set_ylabel("image-level AUROC (%)", color=INK)
    ax.set_title(f"Cross-rung oracle (val-selectable, per class) = {oracle['mean']:.1f}, "
                 f"-{STGRAM['mean']-oracle['mean']:.1f} to STgram —\n"
                 f"vs single-model Rung E {RUNGE_BEST['mean']:.1f} (-{STGRAM['mean']-RUNGE_BEST['mean']:.1f})",
                 fontsize=10.5, color=INK, pad=10)
    ax.legend(frameon=False, fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=1)
    emit("cross_rung_oracle", fig, ["class", "rungE", "oracle", "oracle_source", "stgram"],
         [[c, f"{RUNGE_BEST[c]:.2f}", f"{oracle[c]:.2f}", ORACLE_SRC.get(c, "E"), f"{STGRAM[c]:.2f}"]
          for c in cats])


def main():
    fig_ladder_by_class()
    fig_stgram_per_id()
    fig_cross_rung_oracle()


if __name__ == "__main__":
    main()
