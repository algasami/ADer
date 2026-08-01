"""Seed-repeat check on Rung B vs Rung E (docs/plots/mimii_section_rungE/CONCLUSION.md's open
item): is E's "+0.95, new best of every rung" claim a real effect or single-seed noise?

Reruns Rung B and Rung E at seed 1 and seed 2 (`--seed`, added alongside this check) on top of
the original seed-0 runs, using the identical protocol (50 ep, sub=2, same LRs). Data pulled
straight from each run's own printed best_summary / log (mean image-level AUROC %, maha_embed
for E, best readout for B).

    docs/plots/mimii_section_rungE/seed_repeat/   plot.png data.csv
"""
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_ROOT = "docs/plots/mimii_section_rungE"

B = {"seed 0": 85.86, "seed 1": 85.58, "seed 2": 86.22}
E = {"seed 0": 86.85, "seed 1": 85.44, "seed 2": 86.21}
STGRAM = 90.75

C_B, C_E, C_T, MUTED, INK, GRID = "#D55E00", "#7B3294", "#009E73", "#666666", "#222222", "#dddddd"


def style_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)


def main():
    seeds = list(B)
    x = np.arange(len(seeds))
    w = 0.32
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    style_axes(ax)
    b1 = ax.bar(x - w / 2, [B[s] for s in seeds], w, color=C_B, zorder=3, label="Rung B (FT encoder)")
    b2 = ax.bar(x + w / 2, [E[s] for s in seeds], w, color=C_E, zorder=3, label="Rung E (FT encoder + Mamba)")
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"{b.get_height():.2f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                        fontsize=8.5, color=INK, fontweight="bold")
    ax.axhline(np.mean(list(B.values())), color=C_B, lw=1.2, ls=":", zorder=2)
    ax.axhline(np.mean(list(E.values())), color=C_E, lw=1.2, ls=":", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(seeds, fontsize=9.5)
    ax.set_ylim(83, 89)
    ax.set_ylabel("mean image-level AUROC (%)", color=INK)
    b_mean, e_mean = np.mean(list(B.values())), np.mean(list(E.values()))
    ax.set_title(f"E's seed-0 lead over B (+0.95) does NOT reproduce — over 3 seeds\n"
                 f"B={b_mean:.2f}±{np.std(list(B.values())):.2f}  E={e_mean:.2f}±{np.std(list(E.values())):.2f}  "
                 f"(mean diff {e_mean-b_mean:+.2f}, within per-seed spread)",
                 fontsize=10, color=INK, pad=10)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    d = os.path.join(OUT_ROOT, "seed_repeat")
    os.makedirs(d, exist_ok=True)
    fig.savefig(os.path.join(d, "plot.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    with open(os.path.join(d, "data.csv"), "w", newline="") as f:
        w_ = csv.writer(f)
        w_.writerow(["seed", "rungB_mean_auroc", "rungE_mean_auroc"])
        for s in seeds:
            w_.writerow([s, f"{B[s]:.2f}", f"{E[s]:.2f}"])
    print(f"[out] {d}/plot.png  +  data.csv")


if __name__ == "__main__":
    main()
