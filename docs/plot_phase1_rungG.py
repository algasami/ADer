"""
Phase 1 figures — Rung G: the input pipeline is the lever, the encoder is not
============================================================================

`docs/plots/phase1_rungG/` was written as prose only (CONCLUSION.md); this script
regenerates the figures it describes from the CSVs the runs already wrote. Nothing
here retrains anything.

Two figures, one claim each:

  input_lever.png  The attribution ladder. Rung B's +2.76 over PNGs splits into
                   recipe (+1.00), input (+1.38) and schedule (+0.38) -- so the input
                   pipeline is worth ~+1.4, the largest single lever in the campaign,
                   but half the headline figure.
  encoder_ab.png   AST vs ResNet34 on *identical* cached fbanks: +0.01 at matched
                   epochs, and ResNet34 wins run to convergence. The +5.0 that
                   motivated Rung G was a frozen-feature effect that does not survive
                   fine-tuning -- panel (c) shows the two regimes side by side.

Scoring convention (inherited from the ladder, and optimistic on purpose so the
figures match the text): each run is quoted at the best epoch of its best readout,
both chosen on the test set. The Phase 2 runs are the ones with an honest held-out
rule -- see docs/plot_phase2_asnorm.py. Rung B's headline readout is `neg_cos`;
every fbank arm's is `maha_embed`, which is itself part of the finding.

Usage
-----
    python docs/plot_phase1_rungG.py
    python docs/plot_phase1_rungG.py --figs encoder
"""

import argparse
import csv
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CLASSES = ['fan', 'pump', 'slider', 'valve', 'ToyCar', 'ToyConveyor']
READOUTS = ['neg_cos', 'logit_nll', 'maha_embed']
STGRAM = 90.75
RUNG_F = 86.62                       # previous best deployable (PNG, single seed)
RESNET34_FROZEN = 71.80              # frozen ImageNet ResNet34 + Maha (audio_backbone_probe)

C_BLUE, C_ORANGE, C_AQUA, C_VIOLET = '#2a78d6', '#eb6834', '#1baf7a', '#4a3aa7'
INK, MUTED = '#0b0b0b', '#52514e'


def best_of_readout(d):
    """One run -> its headline: best epoch of the best readout, both chosen on test."""
    cv = pd.read_csv(os.path.join(d, 'metric_curve.csv'))
    best = None
    for r in READOUTS:
        sub = cv[cv.readout == r]
        if sub.empty:
            continue
        i = sub['mean'].idxmax()
        cand = dict(run=os.path.basename(d.rstrip('/')), readout=r,
                    mean=sub.loc[i, 'mean'] * 100, epoch=int(sub.loc[i, 'epoch']),
                    n_epochs=int(cv.epoch.max()),
                    **{c: sub.loc[i, c] * 100 for c in CLASSES})
        if best is None or cand['mean'] > best['mean']:
            best = cand
    return best


def load_arm(name, patterns, curve_readout=None):
    """-> dict(name, dirs, best=DataFrame, curves=[(epoch, mean%) per run], readout).

    `patterns` is one glob or a list of them (Rung B's three seeds are three explicit
    directories, since seed 0 pre-dates the _seedN naming).
    """
    if isinstance(patterns, str):
        patterns = [patterns]
    dirs = [d for p in patterns for d in sorted(glob.glob(p))
            if os.path.isfile(os.path.join(d, 'metric_curve.csv'))]
    if not dirs:
        return None
    best = pd.DataFrame([best_of_readout(d) for d in dirs])
    ro = curve_readout or best.readout.mode()[0]
    curves = []
    for d in dirs:
        cv = pd.read_csv(os.path.join(d, 'metric_curve.csv'))
        sub = cv[cv.readout == ro].sort_values('epoch')
        curves.append((sub.epoch.values, sub['mean'].values * 100))
    return dict(name=name, dirs=dirs, best=best, curves=curves, readout=ro)


def band(curves):
    """Seed mean + min/max envelope over runs that share an epoch grid."""
    n = min(len(y) for _, y in curves)
    ys = np.vstack([y[:n] for _, y in curves])
    return curves[0][0][:n], ys.mean(0), ys.min(0), ys.max(0)


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=10.5, color=INK)
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.grid(alpha=.25, lw=.6)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8.5, colors=MUTED)


# --------------------------------------------------------------------------- #
# figure 1 — the attribution ladder
# --------------------------------------------------------------------------- #
def fig_input(arms, out_dir):
    b, mix, f30, f70 = (arms[k] for k in ('rungB', 'png_mixup', 'fbank30', 'fbank70'))
    v = {k: a['best']['mean'].mean() for k, a in arms.items()}
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2),
                             gridspec_kw=dict(width_ratios=[1.05, 1.1, 1.2]))

    # (a) waterfall: which change bought which part of the +2.76
    ax = axes[0]
    steps = [('Rung B\n(PNG 256x256)', v['rungB'], None, C_BLUE),
             ('+ recipe\n(mixup, lr, batch)', v['png_mixup'], v['png_mixup'] - v['rungB'], C_AQUA),
             ('+ INPUT\n(PNG -> fbank)', v['fbank30'], v['fbank30'] - v['png_mixup'], C_ORANGE),
             ('+ schedule\n(30 -> 70 ep)', v['fbank70'], v['fbank70'] - v['fbank30'], C_VIOLET)]
    base = 84.5
    for i, (lbl, top, delta, col) in enumerate(steps):
        if delta is None:                       # the anchor bar sits on the floor
            ax.bar(i, top - base, .62, bottom=base, color=col, zorder=2)
            ax.text(i, top + .07, f'{top:.2f}', ha='center', fontsize=9, color=INK)
        else:                                   # floating bar spanning just this step
            ax.bar(i, delta, .62, bottom=top - delta, color=col, zorder=2)
            ax.plot([i - .81, i - .31], [top - delta] * 2, color=MUTED, lw=.9, ls=':', zorder=1)
            ax.text(i, top + .07, f'{delta:+.2f}  ->  {top:.2f}', ha='center', fontsize=9,
                    color=col, fontweight='bold')
    for y, lbl, col in ((STGRAM, f'STgram-MFN {STGRAM}', INK),
                        (RUNG_F, f'Rung F {RUNG_F} (previous best)', MUTED)):
        ax.axhline(y, ls=':', c=col, lw=1.3)
        ax.text(-.45, y + .08, lbl, fontsize=8.5, color=col)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([s[0] for s in steps], fontsize=8.5)
    ax.set_ylim(base, 91.6)
    style(ax, '(a) where the +2.76 over Rung B actually came from', '',
          'mean AUROC (%), best readout & epoch')

    # (b) the same four arms as curves -- each at its own headline readout
    ax = axes[1]
    for a, col in ((b, C_BLUE), (mix, C_AQUA), (f70, C_ORANGE)):
        ep, mu, lo, hi = band(a['curves'])
        ax.fill_between(ep, lo, hi, color=col, alpha=.15, lw=0)
        ax.plot(ep, mu, color=col, lw=2,
                label=f"{a['name']} - {a['readout']} ({len(a['curves'])} seed"
                      f"{'s' if len(a['curves']) > 1 else ''})")
    ax.scatter([f30['best'].epoch.iloc[0]], [f30['best']['mean'].iloc[0]], s=70,
               facecolor='white', edgecolor=C_ORANGE, lw=1.8, zorder=4)
    ax.annotate(f"matched 30 ep: {f30['best']['mean'].iloc[0]:.2f}",
                xy=(f30['best'].epoch.iloc[0], f30['best']['mean'].iloc[0]),
                xytext=(34, 84.5), fontsize=8.5, color=C_ORANGE,
                arrowprops=dict(arrowstyle='-', color=C_ORANGE, lw=1, shrinkA=4, shrinkB=6))
    ax.axhline(STGRAM, ls=':', c=INK, lw=1.3)
    ax.text(1, STGRAM + .2, f'STgram-MFN {STGRAM}', fontsize=8.5, color=INK)
    ax.axhline(RUNG_F, ls=':', c=MUTED, lw=1.3)
    ax.text(1, RUNG_F + .2, f'Rung F {RUNG_F}', fontsize=8.5, color=MUTED)
    ax.set_ylim(74, 92)
    ax.legend(fontsize=8.5, frameon=False, loc='lower right')
    style(ax, '(b) per-epoch, at each arm\'s headline readout', 'epoch', 'mean AUROC (%)')

    # (c) per class -- what the input change bought, and where it did not
    ax = axes[2]
    order = CLASSES + ['mean']
    x, w = np.arange(len(order)), .26
    for k, (a, col) in enumerate(((b, C_BLUE), (mix, C_AQUA), (f70, C_ORANGE))):
        vals = [a['best'][c].mean() for c in CLASSES] + [a['best']['mean'].mean()]
        ax.bar(x + (k - 1) * (w + .015), vals, w, color=col, label=a['name'], zorder=2)
    d = [f70['best'][c].mean() - b['best'][c].mean() for c in CLASSES] + \
        [v['fbank70'] - v['rungB']]
    for xi, dv in zip(x, d):
        ax.text(xi, 101, f'{dv:+.1f}', ha='center', fontsize=8,
                color=C_ORANGE if dv > 0 else C_BLUE)
    ax.text(-.4, 105, 'fbank 70 ep - Rung B, per class', fontsize=8, color=MUTED, ha='left')
    ax.set_xticks(x)
    ax.set_xticklabels(order, fontsize=8.5, rotation=20, ha='right')
    ax.set_ylim(50, 108)
    ax.legend(fontsize=8.5, frameon=False, loc='lower right')
    style(ax, '(c) per class at each arm\'s headline checkpoint', '', 'AUROC (%)')

    rows = pd.concat([a['best'].assign(arm=a['name']) for a in arms.values()])
    rows.to_csv(os.path.join(out_dir, 'input_lever.csv'), index=False)
    pd.DataFrame([dict(step=s[0].replace('\n', ' '), cumulative=s[1], delta=s[2])
                  for s in steps]).to_csv(os.path.join(out_dir, 'attribution.csv'),
                                          index=False)
    fig.tight_layout()
    p = os.path.join(out_dir, 'input_lever.png')
    fig.savefig(p, dpi=150, facecolor='white')
    plt.close(fig)
    print(f'[fig] {p}')
    print(rows.groupby('arm')[['mean']].agg(['mean', 'std', 'count']).round(2).to_string())


# --------------------------------------------------------------------------- #
# figure 2 — the encoder A/B, and the frozen-vs-finetuned reversal
# --------------------------------------------------------------------------- #
def fig_encoder(arms, probe_csv, out_dir):
    ast, f30, f70 = arms['ast'], arms['fbank30'], arms['fbank70']
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5))

    # (a) identical cached fbanks, only the encoder differs
    ax = axes[0, 0]
    ep, mu, lo, hi = band(ast['curves'])
    ax.fill_between(ep, lo, hi, color=C_VIOLET, alpha=.15, lw=0)
    ax.plot(ep, mu, color=C_VIOLET, lw=2, label=f'AST (3 seeds) - {ast["readout"]}')
    e70, y70 = f70['curves'][0]
    ax.plot(e70, y70, color=C_ORANGE, lw=2, label=f'ResNet34 control - {f70["readout"]}')
    ax.axvline(30, color=MUTED, lw=1, ls='--')
    ax.text(30.6, 78.6, 'matched\nbudget', fontsize=8, color=MUTED)
    # AST's headline is a mean over three per-seed peaks, so it is a level, not a point on
    # any one curve; ResNet34's is a single real checkpoint, so mark that one.
    ast_mu = ast['best']['mean'].mean()
    ax.plot([1, 30], [ast_mu] * 2, color=C_VIOLET, lw=1.2, dashes=(4, 3))
    ax.text(1.5, 89.5, f"AST best-epoch mean {ast_mu:.2f} +/- "
                               f"{ast['best']['mean'].std(ddof=0):.2f}",
            fontsize=8.5, color=C_VIOLET)
    ax.scatter([f70['best'].epoch.iloc[0]], [f70['best']['mean'].iloc[0]], s=70,
               facecolor='white', edgecolor=C_ORANGE, lw=1.8, zorder=4)
    ax.annotate(f"ResNet34 {f70['best']['mean'].iloc[0]:.2f} @ep"
                f"{int(f70['best'].epoch.iloc[0])}",
                xy=(f70['best'].epoch.iloc[0], f70['best']['mean'].iloc[0]),
                xytext=(46, 83.5), fontsize=8.5, color=C_ORANGE,
                arrowprops=dict(arrowstyle='-', color=C_ORANGE, lw=1, shrinkA=4, shrinkB=7))
    ax.axhline(STGRAM, ls=':', c=INK, lw=1.3)
    ax.text(1, STGRAM + .25, f'STgram-MFN {STGRAM}', fontsize=8.5, color=INK)
    ax.set_ylim(76, 92)
    ax.legend(fontsize=8.5, frameon=False, loc='lower right')
    style(ax, '(a) same cached fbanks, same objective -- only the encoder differs',
          'epoch', 'mean AUROC (%)')

    # (b) per-class: complementary, not one-dominates. Against the MATCHED-budget control
    # (30 ep), so the comparison is encoder-only -- the 70-epoch control also has +0.38 of
    # schedule in it.
    ax = axes[0, 1]
    delta = [ast['best'][c].mean() - f30['best'][c].iloc[0] for c in CLASSES]
    cols = [C_VIOLET if d > 0 else C_ORANGE for d in delta]
    ax.barh(np.arange(len(CLASSES)), delta, .6, color=cols, zorder=2)
    for i, d in enumerate(delta):
        ax.text(d + (.12 if d > 0 else -.12), i, f'{d:+.2f}', va='center',
                ha='left' if d > 0 else 'right', fontsize=8.5, color=INK)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(np.arange(len(CLASSES)))
    ax.set_yticklabels(CLASSES, fontsize=8.5)
    ax.set_xlim(-6, 6)
    ax.text(3.4, 5.35, 'AST better', fontsize=8.5, color=C_VIOLET, ha='center')
    ax.text(-3.4, 5.35, 'ResNet34 better', fontsize=8.5, color=C_ORANGE, ha='center')
    style(ax, '(b) AST - ResNet34 at the matched 30-epoch budget, per class\n'
              'the two encoders are complementary, not one-dominates', 'AUROC (%)', '')

    # (c) the reversal: +5 frozen, ~0 fine-tuned
    ax = axes[1, 0]
    ast_frozen = max(float(r['mean_AUROC']) for r in csv.DictReader(open(probe_csv))
                     if 'maha' in r['method'])
    pts = {'ResNet34': (RESNET34_FROZEN, f30['best']['mean'].iloc[0], C_ORANGE),
           'AST': (ast_frozen, ast['best']['mean'].mean(), C_VIOLET)}
    for (lbl, (fr, ft, col)), dy in zip(pts.items(), (-.55, .55)):   # the two ends nearly
        ax.plot([0, 1], [fr, ft], color=col, lw=2.2, marker='o', ms=8, label=lbl)
        ax.text(-.04, fr, f'{fr:.1f}', ha='right', va='center', fontsize=9, color=col)
        ax.text(1.04, ft + dy, f'{ft:.2f}', ha='left', va='center', fontsize=9, color=col)
    d_fr = pts['AST'][0] - pts['ResNet34'][0]
    d_ft = pts['AST'][1] - pts['ResNet34'][1]
    ax.annotate('', xy=(0.06, pts['AST'][0]), xytext=(0.06, pts['ResNet34'][0]),
                arrowprops=dict(arrowstyle='<->', color=MUTED, lw=1.2))
    ax.text(.09, (pts['AST'][0] + pts['ResNet34'][0]) / 2, f'{d_fr:+.1f}\nAST ahead',
            fontsize=8.5, color=MUTED, va='center')
    ax.text(.72, 84.5, f'{d_ft:+.2f} at matched epochs\n(and ResNet34 wins at 70 ep)',
            fontsize=8.5, color=MUTED, ha='center')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['frozen features\n+ Maha readout', 'fine-tuned end-to-end\n(same input)'],
                       fontsize=9)
    ax.set_xlim(-.25, 1.25)
    ax.set_ylim(68, 92)
    ax.legend(fontsize=8.5, frameon=False, loc='lower right')
    style(ax, '(c) the +5 that motivated Rung G is a FROZEN-regime effect', '',
          'mean AUROC (%)')

    # (d) lr probe -- a flat knob
    ax = axes[1, 1]
    probes = [(k, v) for k, v in arms.items() if k.startswith('lr:')]
    if probes:
        for i, (k, a) in enumerate(probes):
            for e, y in a['curves']:
                ax.plot(e, y, color=[C_BLUE, C_AQUA, C_VIOLET][i % 3], lw=2,
                        label=f"{k[3:]}  best {a['best']['mean'].iloc[0]:.2f}")
        spread = max(a['best']['mean'].iloc[0] for _, a in probes) - \
            min(a['best']['mean'].iloc[0] for _, a in probes)
        ax.text(.98, .06, f'total spread {spread:.2f} AUROC over an 8-epoch probe\n'
                          '-> lr is not a sensitive knob for AST here',
                transform=ax.transAxes, ha='right', fontsize=8.5, color=MUTED)
        ax.legend(fontsize=8.5, frameon=False, loc='lower right', bbox_to_anchor=(1, .16))
        style(ax, '(d) AST learning-rate probe (maha_embed)', 'epoch', 'mean AUROC (%)')
    else:
        ax.axis('off')

    pd.DataFrame([dict(encoder='AST', frozen=ast_frozen, finetuned_30ep=ast['best']['mean'].mean()),
                  dict(encoder='ResNet34', frozen=RESNET34_FROZEN,
                       finetuned_30ep=f30['best']['mean'].iloc[0])]) \
        .to_csv(os.path.join(out_dir, 'frozen_vs_finetuned.csv'), index=False)
    pd.DataFrame({'class': CLASSES, 'ast_30ep': [ast['best'][c].mean() for c in CLASSES],
                  'resnet34_30ep': [f30['best'][c].iloc[0] for c in CLASSES],
                  'delta_matched': delta,
                  'resnet34_70ep': [f70['best'][c].iloc[0] for c in CLASSES]}) \
        .to_csv(os.path.join(out_dir, 'encoder_per_class.csv'), index=False)
    fig.tight_layout()
    p = os.path.join(out_dir, 'encoder_ab.png')
    fig.savefig(p, dpi=150, facecolor='white')
    plt.close(fig)
    print(f'[fig] {p}')
    print(f'  AST frozen {ast_frozen:.2f} vs ResNet34 frozen {RESNET34_FROZEN:.2f} '
          f'({ast_frozen - RESNET34_FROZEN:+.2f}); fine-tuned {d_ft:+.2f} at matched epochs')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default='docs/plots/phase1_rungG')
    ap.add_argument('--probe_csv', default='runs/audio_probe/ast/auroc.csv')
    ap.add_argument('--figs', nargs='+', default=['input', 'encoder'])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Rung B's three seeds are the same dirs docs/plot_phase0_aug.py treats as the control
    # (seed 0 pre-dates the _seedN naming).
    spec = [('rungB', ['runs/section_rungB/log-Mel', 'runs/section_rungB/log-Mel_seed1',
                       'runs/section_rungB/log-Mel_seed2']),
            ('png_mixup', 'runs/phase1_pngctl/png_mixup_seed*'),
            ('fbank30', 'runs/section_rungG/rn34fbank_seed0'),
            ('fbank70', 'runs/section_rungG/rn34fbank70_seed0'),
            ('ast', 'runs/section_rungG/ast_seed*'),
            ('lr: flat 1e-5', 'runs/section_rungG/lrflat1e5_seed0'),
            ('lr: 5e-5 llrd .85', 'runs/section_rungG/lr5e5d85_seed0'),
            ('lr: 1e-4 llrd .75', 'runs/section_rungG/lr1e4d75_seed0')]
    arms = {}
    for name, pat in spec:
        a = load_arm(name, pat)
        if a:
            arms[name] = a
        else:
            print(f'[skip] {name}: nothing matched {pat}')

    labels = {'rungB': 'Rung B (PNG)', 'png_mixup': 'PNG + mixup (matched recipe)',
              'fbank30': 'fbank 30 ep', 'fbank70': 'fbank 70 ep (ResNet34)',
              'ast': 'AST (fbank)'}
    for k, v in labels.items():
        if k in arms:
            arms[k]['name'] = v

    if 'input' in args.figs:
        need = ['rungB', 'png_mixup', 'fbank30', 'fbank70']
        if all(k in arms for k in need):
            fig_input({k: arms[k] for k in need}, args.out_dir)
        else:
            print(f'[skip] input_lever: missing {[k for k in need if k not in arms]}')
    if 'encoder' in args.figs:
        need = ['ast', 'fbank30', 'fbank70']
        if all(k in arms for k in need):
            fig_encoder(arms, args.probe_csv, args.out_dir)
        else:
            print(f'[skip] encoder_ab: missing {[k for k in need if k not in arms]}')


if __name__ == '__main__':
    main()
