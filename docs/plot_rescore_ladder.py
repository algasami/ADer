"""Figure for the honest re-score of the labels/objective ladder (rungs A-F).

Reads `docs/plots/ladder_honest/per_run.csv` (written by `docs/rescore_ladder.py`) and draws
the two corrections as a walk, one row per rung:

    pooled-clip @ test-selected   ->   mean-of-per-ID @ test-selected   ->   @ held-out

i.e. the metric correction (FINAL_REPORT §5.2) and then the selection correction (§5.3),
separated, so it is visible which one moves a given rung and by how much. A connected dot plot
is the right form here because the quantity of interest is the *change* along a fixed
sequence of conventions, not the three levels independently.

    python docs/rescore_ladder.py && python docs/plot_rescore_ladder.py
    -> docs/plots/ladder_honest/{plot.png,data.csv}
"""
import os
import csv
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "docs/plots/ladder_honest"
STGRAM = 90.75

# Categorical slots 1-3 of the reference palette, in fixed order, unmodified. Documented as
# all-pairs validated in both modes; slot 3 (aqua) is below 3:1 on a light surface, so the
# relief rule applies -> every point carries a visible direct label and data.csv is the table.
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, RULE = "#0b0b0b", "#52514e", "#e6e6e3", "#9a9992"

STEPS = [("pooled_test_selected", "pooled-clip, test-selected  (the published convention)", C1),
         ("meanid_test_selected", "mean-of-per-ID, test-selected  (§5.2 metric fixed)", C2),
         ("meanid_global_heldout", "mean-of-per-ID, held-out  (§5.2 + §5.3, honest)", C3)]

RUNG_LABEL = {'A': 'A  frozen enc + head', 'B': 'B  fine-tuned encoder',
              'C': 'C  Mamba, frozen enc', 'D': 'D  joint distill+classify',
              'E': 'E  B+C combined', 'F': 'F  E + per-class readout',
              'F+': 'F+ per-class epoch too\n     (6 checkpoints)'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=os.path.join(OUT, 'per_run.csv'))
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    rows = []
    for rung in [r for r in 'ABCDE' if r in set(df['rung'])]:
        s = df[df['rung'] == rung]
        rows.append(dict(rung=rung, n=len(s),
                         **{k: s[k].mean() for k, _l, _c in STEPS},
                         sd=s['meanid_global_heldout'].std(ddof=0) if len(s) > 1 else np.nan))
    # Rung F is Rung E scored under the per-class readout policy -- not a separate run.
    # F+ additionally lets the epoch vary per class, which is up to 6 checkpoints, not a model.
    e = df[df['rung'] == 'E']
    if len(e):
        for tag, tcol, hcol in (('F', 'meanid_test_perclass_readout',
                                 'meanid_perclass_readout_heldout'),
                                ('F+', 'meanid_oracle', 'meanid_perclass_full_heldout')):
            rows.append(dict(rung=tag, n=len(e), pooled_test_selected=np.nan,
                             meanid_test_selected=e[tcol].mean(),
                             meanid_global_heldout=e[hcol].mean(),
                             sd=e[hcol].std(ddof=0) if len(e) > 1 else np.nan))

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'data.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['rung', 'n_seeds'] + [k for k, _l, _c in STEPS] + ['heldout_sd'])
        for r in rows:
            w.writerow([r['rung'], r['n']] + [f"{r[k]:.2f}" if np.isfinite(r[k]) else ''
                                              for k, _l, _c in STEPS] +
                       [f"{r['sd']:.2f}" if np.isfinite(r['sd']) else ''])

    fig, ax = plt.subplots(figsize=(9.4, 0.92 * len(rows) + 2.4))
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=0)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)

    ys = np.arange(len(rows))[::-1]
    for y, r in zip(ys, rows):
        vals = [(r[k], c) for k, _l, c in STEPS if np.isfinite(r[k])]
        xs = [v for v, _ in vals]
        ax.plot(xs, [y] * len(xs), color=RULE, lw=1.4, zorder=2, solid_capstyle="round")
        for v, c in vals:
            ax.scatter([v], [y], s=78, color=c, zorder=4, edgecolor="white", linewidth=1.6)
        # Label the two ends, outward, so a short walk cannot collide with itself; the middle
        # step gets a label only when it is far enough from both ends to sit above the line.
        lo_v, hi_v = min(xs), max(xs)
        ax.annotate(f"{lo_v:.2f}", (lo_v, y), xytext=(-9, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=8.4, color=INK, fontweight="bold", zorder=5)
        if hi_v > lo_v:
            ax.annotate(f"{hi_v:.2f}", (hi_v, y), xytext=(9, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=8.4, color=INK, fontweight="bold",
                        zorder=5)
        if len(vals) == 3:
            mid = vals[1][0]
            if min(abs(mid - lo_v), abs(mid - hi_v)) > 0.45:
                ax.annotate(f"{mid:.2f}", (mid, y), xytext=(0, 10),
                            textcoords="offset points", ha="center", va="bottom",
                            fontsize=8.0, color=MUTED, fontweight="bold", zorder=5)

    ax.axvline(STGRAM, color=MUTED, lw=1.2, ls="--", zorder=1)
    ax.annotate(f"STgram-MFN {STGRAM}", (STGRAM, ys[0] + 0.62), xytext=(-6, 0),
                textcoords="offset points", ha="right", va="center",
                fontsize=8.6, color=MUTED, style="italic")

    ax.set_yticks(ys)
    ax.set_yticklabels([RUNG_LABEL.get(r['rung'], r['rung']) +
                        (f"  (n={r['n']})" if r['n'] else "") for r in rows], fontsize=9.5,
                       color=INK)
    ax.set_ylim(-0.75, len(rows) - 0.25)
    lo = min(v for r in rows for k, _l, _c in STEPS if np.isfinite(v := r[k]))
    ax.set_xlim(lo - 2.2, max(STGRAM + 1.2, max(r['meanid_test_selected'] for r in rows) + 1.5))
    ax.set_xlabel("mean AUROC (%)", color=INK, fontsize=9.5)
    handles = [plt.Line2D([], [], marker='o', ls='', color=c, markersize=8,
                          markeredgecolor='white', markeredgewidth=1.4, label=l)
               for _k, l, c in STEPS]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.10), frameon=False,
              fontsize=8.6, labelcolor=INK, ncol=1, handletextpad=0.5, borderaxespad=0.0)
    ax.set_title("Rungs A-F re-scored under the honest rule\n"
                 "epoch AND readout chosen on a held-out half, mean-of-per-ID as STgram-MFN "
                 "reports it", fontsize=10.5, color=INK, pad=12, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "plot.png"), dpi=170, bbox_inches="tight",
                facecolor="white")
    print(f"[out] {OUT}/plot.png  +  data.csv")


if __name__ == '__main__':
    main()
