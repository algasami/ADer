"""Aggregate + plot the audio-backbone ablation (diagnostics/audio_backbone_probe.py).

Sibling of docs/plot_scan_ablation.py / docs/plot_scorer_ablation.py, but for the *feature
extractor* front. Unlike those two sweeps there is NO epoch axis here: the backbone probe is a
single-shot frozen-feature test — each backbone's own front-end embeds the raw MIMII wavs once,
then the SAME Maha/kNN scorers are fit on the SAME train-normal / test split (imported from
frozen_encoder_probe.py). So the figures are cross-sectional (bar charts), not vs-epoch curves,
but the folder convention is identical to the sibling scripts: each figure is a self-contained
folder holding BOTH the PNG and the CSV of the numbers behind it, plus one master summary CSV:

    docs/plots/mimii_backbone/
        backbone_summary.csv                 # every backbone x scorer x class
        mean_auroc_bar/        plot.png data.csv   # the ladder: RN34 -> CNN14 -> AST -> STgram-MFN
        auroc_per_class_grouped/ plot.png data.csv # per-class, best scorer per backbone
        scorer_variants/       plot.png data.csv   # AST/CNN14 internal embedding x scorer variants
        gap_decomposition/     plot.png data.csv   # waterfall: features (+5) vs objective (+14)

The two audio backbones are read LIVE from runs/audio_probe/{ast,cnn14}/auroc.csv (the variable
part of the experiment). The two fixed references are documented constants:
  - ResNet34 (ImageNet, image) maha_concat  -> RESNET34_REF (matches audio_backbone_probe.py:83,
    from runs/frozen_probe/mambaad_mimii_toy/auroc.csv), the MambaAD-track ceiling / baseline.
  - STgram-MFN (supervised audio SOTA)       -> STGRAM_MFN (from
    STgram-MFN/results/STgram-MFN(m=0.7,s=30)/result.csv, per-class + Total Average).
Best scorer per audio backbone = the max-mean row (Maha wins everywhere -> ast_meanpatch_maha,
cnn14_emb_maha). All scorers are image-level (constant map) so sp_max == sp_mean; no family split.
"""
import argparse
import csv
import os
import numpy as np
import matplotlib.pyplot as plt

CLASSES = ["fan", "pump", "slider", "valve", "ToyCar", "ToyConveyor"]
OUT_ROOT = "docs/plots/mimii_backbone"

# ---- fixed reference tracks (constants; see module docstring for provenance) ----
RESNET34_REF = {  # ImageNet ResNet34, image, maha_concat (best pooled frozen scorer)
    "fan": 58.44, "pump": 72.06, "slider": 90.78, "valve": 70.02,
    "ToyCar": 75.68, "ToyConveyor": 63.79, "mean": 71.80,
}
STGRAM_MFN = {  # supervised audio SOTA baseline, per-class AUROC (%)
    "fan": 87.09, "pump": 90.94, "slider": 98.87, "valve": 98.59,
    "ToyCar": 94.72, "ToyConveyor": 74.27, "mean": 90.75,
}

# per-track plotting style. RN34 = baseline (black); CNN14 = the dead end (red);
# AST = best frozen (green); STgram-MFN = the supervised target/ceiling (blue).
TRACKS = {
    "ResNet34":   dict(color="#111111", label="ResNet34 (ImageNet, image)"),
    "CNN14":      dict(color="#d62728", label="CNN14 (AudioSet, audio)"),
    "AST":        dict(color="#2ca02c", label="AST (AudioSet, audio)"),
    "STgram-MFN": dict(color="#1f77b4", label="STgram-MFN (supervised SOTA)"),
}


def read_probe_csv(path):
    """runs/audio_probe/<bb>/auroc.csv -> list of dicts {method, fan.., mean_AUROC, mean_AP}."""
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "method": r["method"],
                **{c: float(r[c]) for c in CLASSES},
                "mean_AUROC": float(r["mean_AUROC"]),
                "mean_AP": float(r.get("mean_AP", "nan")),
            })
    return rows


def best_row(rows):
    """The max mean_AUROC row (best scorer/embedding for that backbone)."""
    return max(rows, key=lambda r: r["mean_AUROC"])


def emit(name, fig, header, table):
    """Write a figure folder: <OUT_ROOT>/<name>/{plot.png, data.csv}. Mirrors plot_scan_ablation."""
    d = os.path.join(OUT_ROOT, name)
    os.makedirs(d, exist_ok=True)
    fig.savefig(os.path.join(d, "plot.png"), dpi=150)
    plt.close(fig)
    with open(os.path.join(d, "data.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(table)
    print("wrote", d)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe-dir", default="runs/audio_probe",
                    help="dir holding <backbone>/auroc.csv from audio_backbone_probe.py")
    args = ap.parse_args()
    os.makedirs(OUT_ROOT, exist_ok=True)

    ast_rows = read_probe_csv(os.path.join(args.probe_dir, "ast", "auroc.csv"))
    cnn_rows = read_probe_csv(os.path.join(args.probe_dir, "cnn14", "auroc.csv"))
    ast_best, cnn_best = best_row(ast_rows), best_row(cnn_rows)

    # per-track best-scorer AUROC vectors (6 classes + mean), in narrative ladder order
    def vec(d):
        return [d[c] for c in CLASSES] + [d["mean"]]
    track_vec = {
        "ResNet34":   vec(RESNET34_REF),
        "CNN14":      [cnn_best[c] for c in CLASSES] + [cnn_best["mean_AUROC"]],
        "AST":        [ast_best[c] for c in CLASSES] + [ast_best["mean_AUROC"]],
        "STgram-MFN": vec(STGRAM_MFN),
    }
    track_scorer = {"ResNet34": "maha_concat", "CNN14": cnn_best["method"],
                    "AST": ast_best["method"], "STgram-MFN": "supervised"}
    rn34_mean = RESNET34_REF["mean"]

    # ===================== master summary CSV =====================
    with open(os.path.join(OUT_ROOT, "backbone_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["backbone", "scorer", "is_best"] + CLASSES
                   + ["mean_AUROC", "delta_vs_ResNet34"])
        w.writerow(["ResNet34", "maha_concat", 1]
                   + [RESNET34_REF[c] for c in CLASSES] + [rn34_mean, 0.0])
        for bb, rows, best in (("CNN14", cnn_rows, cnn_best), ("AST", ast_rows, ast_best)):
            for r in rows:
                w.writerow([bb, r["method"], int(r is best)]
                           + [round(r[c], 2) for c in CLASSES]
                           + [round(r["mean_AUROC"], 2),
                              round(r["mean_AUROC"] - rn34_mean, 2)])
        w.writerow(["STgram-MFN", "supervised", 1]
                   + [STGRAM_MFN[c] for c in CLASSES]
                   + [STGRAM_MFN["mean"], round(STGRAM_MFN["mean"] - rn34_mean, 2)])
    print("wrote", os.path.join(OUT_ROOT, "backbone_summary.csv"))

    order = ["ResNet34", "CNN14", "AST", "STgram-MFN"]

    # ===================== Fig 1: mean-AUROC ladder =====================
    fig, ax = plt.subplots(figsize=(9, 5.5))
    means = [track_vec[t][-1] for t in order]
    bars = ax.bar(range(len(order)), means,
                  color=[TRACKS[t]["color"] for t in order], alpha=0.9, width=0.62)
    ax.axhline(50, ls=":", c="gray", lw=1, alpha=0.7)
    ax.text(-0.45, 50.6, "chance", color="gray", fontsize=8)
    ax.axhline(rn34_mean, ls="--", c="#111111", lw=1, alpha=0.5)
    for i, (t, b) in enumerate(zip(order, bars)):
        m = means[i]
        d = m - rn34_mean
        lbl = f"{m:.1f}" if t == "ResNet34" else f"{m:.1f}\n({d:+.1f})"
        ax.text(b.get_x() + b.get_width() / 2, m + 0.6, lbl,
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([TRACKS[t]["label"] for t in order], fontsize=8.5)
    ax.set_ylabel("mean image-level AUROC [%]")
    ax.set_ylim(45, 97)
    ax.set_title("MIMII frozen-backbone probe — mean AUROC (best scorer per backbone)\n"
                 "Δ vs ImageNet ResNet34 baseline (dashed); features buy +5, objective buys +14")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    emit("mean_auroc_bar", fig, ["backbone", "scorer", "mean_AUROC", "delta_vs_ResNet34"],
         [[t, track_scorer[t], round(track_vec[t][-1], 2),
           round(track_vec[t][-1] - rn34_mean, 2)] for t in order])

    # ===================== Fig 2: per-class grouped bars =====================
    fig, ax = plt.subplots(figsize=(14, 6.5))
    x = np.arange(len(CLASSES) + 1)  # classes + Avg
    w = 0.8 / len(order)
    rows_csv = []
    for j, t in enumerate(order):
        ys = track_vec[t]
        ax.bar(x + (j - (len(order) - 1) / 2) * w, ys, w,
               color=TRACKS[t]["color"], alpha=0.9,
               label=f"{TRACKS[t]['label']} [{track_scorer[t]}]")
        for c, yv in zip(CLASSES + ["Avg"], ys):
            rows_csv.append([t, track_scorer[t], c, round(yv, 2)])
    ax.axhline(50, ls=":", c="gray", lw=1, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES + ["Avg"])
    ax.set_ylabel("image-level AUROC [%]")
    ax.set_ylim(45, 102)
    ax.set_title("MIMII frozen-backbone probe — per-class AUROC (best scorer per backbone)\n"
                 "slider separates for all; fan/ToyConveyor/valve are where frozen features fall short")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8, ncol=2, loc="upper center")
    fig.tight_layout()
    emit("auroc_per_class_grouped", fig, ["backbone", "scorer", "class", "AUROC"], rows_csv)

    # ===================== Fig 3: AST/CNN14 scorer x embedding variants =====================
    variants = [("AST", r) for r in ast_rows] + [("CNN14", r) for r in cnn_rows]
    fig, ax = plt.subplots(figsize=(15, 6.5))
    w = 0.8 / len(variants)
    rows_csv = []
    for j, (bb, r) in enumerate(variants):
        base = TRACKS[bb]["color"]
        # maha = solid, knn = hatched; shade AST pooler-vs-meanpatch by alpha
        is_knn = r["method"].endswith("knn")
        alpha = 0.55 if ("pooler" in r["method"]) else 0.95
        ys = [r[c] for c in CLASSES] + [r["mean_AUROC"]]
        ax.bar(x + (j - (len(variants) - 1) / 2) * w, ys, w, color=base, alpha=alpha,
               hatch="//" if is_knn else None, edgecolor="white", linewidth=0.4,
               label=r["method"])
        for c, yv in zip(CLASSES + ["Avg"], ys):
            rows_csv.append([bb, r["method"], c, round(yv, 2)])
    ax.axhline(rn34_mean, ls="--", c="#111111", lw=1, alpha=0.6)
    ax.text(x[-1] + 0.35, rn34_mean + 0.3, f"ResNet34 baseline {rn34_mean:.1f}",
            color="#111111", fontsize=8, ha="right")
    ax.axhline(50, ls=":", c="gray", lw=1, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES + ["Avg"])
    ax.set_ylabel("image-level AUROC [%]")
    ax.set_ylim(45, 100)
    ax.set_title("MIMII backbone probe — AST vs CNN14 scorer/embedding variants\n"
                 "Maha (solid) > kNN (hatched); AST mean-patch (opaque) > pooler token (faded)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8, ncol=3, loc="upper center")
    fig.tight_layout()
    emit("scorer_variants", fig, ["backbone", "method", "class", "AUROC"], rows_csv)

    # ===================== Fig 4: gap-decomposition waterfall =====================
    ast_mean, stg_mean = track_vec["AST"][-1], track_vec["STgram-MFN"][-1]
    d_feat = ast_mean - rn34_mean
    d_obj = stg_mean - ast_mean
    steps = [
        ("ImageNet RN34\n+ UAD Maha", rn34_mean, 0.0, "#111111", "abs"),
        ("+ AST audio\nfeatures", d_feat, rn34_mean, "#2ca02c", "delta"),
        ("+ supervised\nobjective", d_obj, ast_mean, "#9467bd", "delta"),
        ("STgram-MFN\n(SOTA)", stg_mean, 0.0, "#1f77b4", "abs"),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 6))
    wtab = []
    for i, (lbl, val, bottom, color, kind) in enumerate(steps):
        if kind == "abs":
            ax.bar(i, val, 0.62, color=color, alpha=0.9)
            ax.text(i, val + 0.7, f"{val:.1f}", ha="center", va="bottom",
                    fontsize=11, fontweight="bold")
            wtab.append([lbl.replace("\n", " "), "level", round(val, 2), round(val, 2)])
        else:
            ax.bar(i, val, 0.62, bottom=bottom, color=color, alpha=0.9)
            ax.text(i, bottom + val + 0.7, f"+{val:.1f}", ha="center", va="bottom",
                    fontsize=11, fontweight="bold", color=color)
            wtab.append([lbl.replace("\n", " "), "gain", round(val, 2), round(bottom + val, 2)])
        # connector to next bar top
        if i < len(steps) - 1:
            top = val if kind == "abs" else bottom + val
            ax.plot([i + 0.31, i + 1 - 0.31], [top, top], ls=":", c="gray", lw=1)
    ax.axhline(50, ls=":", c="gray", lw=1, alpha=0.7)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([s[0] for s in steps], fontsize=9)
    ax.set_ylabel("mean image-level AUROC [%]")
    ax.set_ylim(45, 97)
    ax.set_title("MIMII gap decomposition — features vs learning objective\n"
                 f"backbone swap recovers +{d_feat:.1f}; the remaining +{d_obj:.1f} is the "
                 f"supervised objective")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    emit("gap_decomposition", fig, ["stage", "kind", "value", "cumulative"], wtab)

    # ---- console recap ----
    print("\n==> backbone recap (mean AUROC, best scorer):")
    for t in order:
        print(f"   {t:11s} {track_scorer[t]:18s} {track_vec[t][-1]:6.2f}  "
              f"(Δ {track_vec[t][-1] - rn34_mean:+.2f} vs RN34)")


if __name__ == "__main__":
    main()
