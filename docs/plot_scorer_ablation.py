"""Aggregate + plot the scorer-ablation sweep (docs/run_scorer_ablation.sh).

Reads one 42-column metric file per test-time score readout (produced by
docs/reeval_sp_mean.py re-scoring the SAME e50 x log-Mel run at every saved epoch) and
emits, for each figure, a self-contained folder holding BOTH the PNG and the CSV of the
numbers behind it:

    docs/plots/mimii_scorer/
        scorer_summary.csv                         # master table (all scorers/classes/metrics)
        auroc_avg_vs_epoch_<family>/  plot.png  data.csv
        auroc_per_class_vs_epoch_<family>/ plot.png  data.csv
        best_per_class_<family>/      plot.png  data.csv

`<family>` is the pixel pooling used to turn the anomaly map into a clip score: sp_max
(max-pool) or sp_mean (mean-pool). Image-level scorers (Maha/kNN) broadcast a constant map,
so their sp_max == sp_mean by construction; only cos-residual differs between the two.

Column layout of each metric file matches trainer/_base_trainer.py:_finish (see
docs/plot_mimii_val_metrics.py:build_col): outer loop over CLASSES, inner over the 6 metrics
[mAUROC_sp_max, mAP_sp_max, mF1_max_sp_max, mAUROC_sp_mean, AP_sp_mean, F1_max_sp_mean], with
Avg interleaved after the last class. Rows are epoch-aligned; only saved-epoch rows are
nonzero.
"""
import argparse
import csv
import os
import numpy as np
import matplotlib.pyplot as plt

CLASSES = ["fan", "pump", "slider", "valve", "ToyCar", "ToyConveyor"]

# Test-time score readouts, in legend order. style: (color, linestyle, is_reference).
# Teacher readouts ignore the decoder (frozen encoder) -> flat "ceiling" reference (dashed).
SCORERS = {
    "cos-residual": dict(color="#111111", ls="-",  ref=False, label="cos-residual (native)"),
    "student-maha": dict(color="#1f77b4", ls="-",  ref=False, label="student-maha"),
    "student-knn":  dict(color="#2ca02c", ls="-",  ref=False, label="student-knn"),
    "teacher-maha": dict(color="#d62728", ls="--", ref=True,  label="teacher-maha (frozen)"),
    "teacher-knn":  dict(color="#ff7f0e", ls="--", ref=True,  label="teacher-knn (frozen)"),
}

# within-class metric index per family: AUROC / AP / F1
FAMILIES = {
    "sp_max":  dict(auroc=0, ap=1, f1=2, names=["AUROC", "AP", "F1-max"]),
    "sp_mean": dict(auroc=3, ap=4, f1=5, names=["AUROC", "AP", "F1-max"]),
}
OUT_ROOT = "docs/plots/mimii_scorer"


def build_col(ncols):
    """(class, metric_idx) -> column, auto-detecting metrics-per-class M (mirrors the sibling
    plot script). Regular classes are contiguous M-blocks; ToyConveyor + Avg interleave the tail."""
    C = len(CLASSES)
    if ncols % (C + 1) != 0:
        raise ValueError(f"{ncols} cols not divisible by {C + 1}; unexpected layout")
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


def load_scorer(path):
    """Return (epochs, values[rows, cols], col_map) for the saved (nonzero) epochs."""
    a = np.loadtxt(path)
    if a.ndim == 1:
        a = a[None, :]
    col = build_col(a.shape[1])
    rows = np.where(a[:, col[("Avg", 0)]] > 0)[0]
    return rows + 1, a[rows], col


def emit(name, fig, header, table):
    """Write a figure folder: <OUT_ROOT>/<name>/{plot.png, data.csv}."""
    d = os.path.join(OUT_ROOT, name)
    os.makedirs(d, exist_ok=True)
    png = os.path.join(d, "plot.png")
    fig.savefig(png, dpi=150)
    plt.close(fig)
    with open(os.path.join(d, "data.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(table)
    print("wrote", d)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=os.path.join(OUT_ROOT, "_data"),
                    help="dir holding <scorer>.txt metric files")
    ap.add_argument("--run-dir", default="", help="(informational) the re-scored run dir")
    args = ap.parse_args()

    os.makedirs(OUT_ROOT, exist_ok=True)
    data = {}
    for key in SCORERS:
        p = os.path.join(args.data_dir, f"{key}.txt")
        if not os.path.exists(p):
            print(f"!! missing {p} — skipping {key}")
            continue
        data[key] = load_scorer(p)
    if not data:
        raise SystemExit(f"no scorer metric files found in {args.data_dir}")

    # ================= master summary CSV (peak + final, both families) =================
    summ_rows = []
    for key, (ep, vals, col) in data.items():
        avg_auroc_sp_max = vals[:, col[("Avg", FAMILIES["sp_max"]["auroc"])]]
        peak_i = int(np.argmax(avg_auroc_sp_max))
        for fam_name, fam in FAMILIES.items():
            for cls in CLASSES + ["Avg"]:
                y_auroc = vals[:, col[(cls, fam["auroc"])]]
                y_ap = vals[:, col[(cls, fam["ap"])]]
                y_f1 = vals[:, col[(cls, fam["f1"])]]
                # per-class own peak (by that class's AUROC) + the run-level (Avg-AUROC) peak epoch
                cpeak_i = int(np.argmax(y_auroc))
                summ_rows.append([
                    key, cls, fam_name,
                    round(float(y_auroc[peak_i]), 3), int(ep[peak_i]),          # at Avg-peak epoch
                    round(float(y_auroc[cpeak_i]), 3), int(ep[cpeak_i]),        # at own-peak epoch
                    round(float(y_auroc[-1]), 3), int(ep[-1]),                  # final epoch
                    round(float(y_ap[peak_i]), 3), round(float(y_f1[peak_i]), 3),
                ])
    with open(os.path.join(OUT_ROOT, "scorer_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scorer", "class", "family",
                    "AUROC_at_avgpeak", "avgpeak_epoch",
                    "AUROC_at_ownpeak", "ownpeak_epoch",
                    "AUROC_final", "final_epoch", "AP_at_avgpeak", "F1_at_avgpeak"])
        w.writerows(summ_rows)
    print("wrote", os.path.join(OUT_ROOT, "scorer_summary.csv"))

    # ================= per-family figures =================
    for fam_name, fam in FAMILIES.items():
        ai = fam["auroc"]

        # ---- Fig A: Avg AUROC vs epoch, one line per scorer (peak ring) ----
        fig, ax = plt.subplots(figsize=(9, 5.5))
        epochs_union = sorted({int(e) for (ep, _, _) in data.values() for e in ep})
        header = ["epoch"] + list(data.keys())
        table = {e: [e] + [""] * len(data) for e in epochs_union}
        for j, (key, (ep, vals, col)) in enumerate(data.items()):
            st = SCORERS[key]
            y = vals[:, col[("Avg", ai)]]
            ax.plot(ep, y, marker="o", ms=4, lw=1.9, color=st["color"], ls=st["ls"],
                    label=st["label"])
            pk = int(np.argmax(y))
            ax.scatter([ep[pk]], [y[pk]], s=95, facecolors="none", edgecolors=st["color"],
                       linewidths=1.9, zorder=5)
            for e, v in zip(ep, y):
                table[int(e)][1 + j] = round(float(v), 3)
        ax.axhline(50, ls=":", c="gray", lw=1, alpha=0.7)
        ax.text(1, 50.5, "chance", color="gray", fontsize=8)
        ax.set_xlabel("epoch")
        ax.set_ylabel(f"mean AUROC ({fam_name}) [%]")
        ax.set_title(f"MambaAD / MIMII e50 log-Mel — scorer ablation\n"
                     f"Avg image-level AUROC vs. epoch ({fam_name}); open ring = peak")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="lower right")
        fig.tight_layout()
        emit(f"auroc_avg_vs_epoch_{fam_name}", fig, header,
             [table[e] for e in epochs_union])

        # ---- Fig B: per-class AUROC vs epoch (6 panels) ----
        fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
        rows_csv = []
        for ax, cls in zip(axes.ravel(), CLASSES):
            for key, (ep, vals, col) in data.items():
                st = SCORERS[key]
                y = vals[:, col[(cls, ai)]]
                ax.plot(ep, y, marker="o", ms=3.3, lw=1.5, color=st["color"], ls=st["ls"],
                        label=st["label"])
                for e, v in zip(ep, y):
                    rows_csv.append([cls, key, int(e), round(float(v), 3)])
            ax.axhline(50, ls=":", c="gray", lw=1, alpha=0.7)
            ax.set_title(f"{cls} — AUROC ({fam_name})")
            ax.set_ylabel("AUROC [%]")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        for ax in axes[-1]:
            ax.set_xlabel("epoch")
        fig.suptitle(f"MambaAD / MIMII e50 log-Mel — per-class AUROC vs. epoch ({fam_name})",
                     fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        emit(f"auroc_per_class_vs_epoch_{fam_name}", fig,
             ["class", "scorer", "epoch", "AUROC"], rows_csv)

        # ---- Fig C: best-epoch per-class AUROC, grouped bars ----
        # each scorer at its own Avg-AUROC peak epoch (fair single-number comparison)
        fig, ax = plt.subplots(figsize=(13, 6))
        keys = list(data.keys())
        x = np.arange(len(CLASSES) + 1)  # classes + Avg
        w = 0.8 / max(len(keys), 1)
        bar_rows = []
        for j, key in enumerate(keys):
            ep, vals, col = data[key]
            pk = int(np.argmax(vals[:, col[("Avg", ai)]]))
            ys = [float(vals[pk, col[(c, ai)]]) for c in CLASSES]
            ys.append(float(vals[pk, col[("Avg", ai)]]))
            st = SCORERS[key]
            ax.bar(x + (j - (len(keys) - 1) / 2) * w, ys, w, color=st["color"],
                   label=f"{st['label']} @ep{int(ep[pk])}", alpha=0.9)
            for c, yv in zip(CLASSES + ["Avg"], ys):
                bar_rows.append([key, int(ep[pk]), c, round(yv, 3)])
        ax.axhline(50, ls=":", c="gray", lw=1, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(CLASSES + ["Avg"])
        ax.set_ylabel(f"AUROC ({fam_name}) [%]")
        ax.set_title(f"MambaAD / MIMII e50 log-Mel — per-class AUROC at each scorer's peak ({fam_name})")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        emit(f"best_per_class_{fam_name}", fig,
             ["scorer", "peak_epoch", "class", "AUROC"], bar_rows)

    # ---- console recap ----
    print("\n==> scorer recap (Avg AUROC, sp_max family):")
    for key, (ep, vals, col) in data.items():
        y = vals[:, col[("Avg", FAMILIES["sp_max"]["auroc"])]]
        print(f"   {key:14s} start(ep{ep[0]})={y[0]:6.2f}  peak={y.max():6.2f}@ep{ep[int(np.argmax(y))]:<3d}  final={y[-1]:6.2f}")


if __name__ == "__main__":
    main()
