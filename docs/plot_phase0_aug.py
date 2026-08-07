"""
Phase 0 — does spectrogram augmentation move Rung B?
====================================================

Every rung of the labels/objective ladder (A-F) trained with NO augmentation at all:
`MambaAD/configs/mambaad/mimii/_base.py:94-100` assigns `test_transforms =
train_transforms` (the same list object), so train views were deterministic
Resize->CenterCrop->ToTensor->Normalize. This script compares that control against
Rung B re-run with train-only spectrogram augmentation (`diagnostics/spec_augment.py`),
3 seeds each.

It answers THREE questions, not one — the mean is the least interesting:

  Q1  Does the best-epoch mean AUROC move, relative to the ~0.5-1pt seed noise that the
      B/E seed-repeat established? (paired per-seed deltas, not just mean-of-means)
  Q2  Does the "AUROC peaks early then decays" signature change? If augmentation pushes
      the peak epoch later and flattens the post-peak decay, then part of what the ladder
      read as a scoring artifact was really missing regularization.
  Q3  Does the train-accuracy curve stop saturating? The control hits ~95% by epoch 2 and
      99.6% by epoch 50 on a 23-way task — if that persists under augmentation, the
      augmentation is too weak to be the explanation either way.

Usage
-----
    python docs/plot_phase0_aug.py                    # defaults below
    python docs/plot_phase0_aug.py --treat_glob 'runs/phase0_aug/mild_seed*'

Outputs (docs/plots/phase0_aug/): summary.csv, per_class.csv, phase0_aug.png
"""

import argparse
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


def load_run(d):
    """-> (curve_df, train_df, label). Missing train_log is tolerated."""
    curve = pd.read_csv(os.path.join(d, 'metric_curve.csv'))
    tp = os.path.join(d, 'train_log.csv')
    train = pd.read_csv(tp) if os.path.isfile(tp) else None
    return curve, train, os.path.basename(d.rstrip('/'))


def last_epoch(runs):
    """Lowest final epoch across an arm's runs -- these curves are written incrementally,
    so an arm that is still training looks exactly like a finished one to pandas. Peaks
    taken from a 3-epoch curve are not a result; the caller must gate on this."""
    return min(int(cv.epoch.max()) for cv, _, _ in runs) if runs else 0


def peak_stats(curve, readout):
    """Best mean AUROC, the epoch it happened, and how much is lost by the final epoch."""
    sub = curve[curve.readout == readout]
    if sub.empty:
        return None
    i = sub['mean'].idxmax()
    best, ep = sub.loc[i, 'mean'] * 100, int(sub.loc[i, 'epoch'])
    final = sub.sort_values('epoch')['mean'].iloc[-1] * 100
    return dict(best=best, epoch=ep, final=final, decay=best - final,
                per_class={c: sub.loc[i, c] * 100 for c in CLASSES})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ctrl_dirs', nargs='+', default=[
        'runs/section_rungB/log-Mel',        # seed 0 (pre-dates the _seedN naming)
        'runs/section_rungB/log-Mel_seed1',
        'runs/section_rungB/log-Mel_seed2'])
    ap.add_argument('--arms', nargs='+',
                    default=['aug=runs/phase0_aug/standard_seed*',
                             'aug-nomixup=runs/phase0_aug/nomixup_seed*'],
                    help='treatment arms as name=glob; compared against the no-aug control')
    ap.add_argument('--out_dir', default='docs/plots/phase0_aug')
    ap.add_argument('--min_epochs', type=int, default=None,
                    help='epochs an arm must have to be quoted (default: the control\'s)')
    ap.add_argument('--allow_incomplete', action='store_true',
                    help='include still-training arms -- for inspection only, not results')
    args = ap.parse_args()

    arms = [('no-aug', [load_run(d) for d in args.ctrl_dirs if os.path.isdir(d)])]
    for spec in args.arms:
        name, _, pat = spec.partition('=')
        runs = [load_run(d) for d in sorted(glob.glob(pat)) if os.path.isdir(d)]
        if runs:                      # skip arms that have not been run yet
            arms.append((name, runs))
        else:
            print(f'[skip] arm "{name}": no runs matched {pat}')
    if len(arms) < 2:
        raise SystemExit('no treatment arms found')

    # An arm that is still training writes metric_curve.csv incrementally and is
    # indistinguishable from a finished one once loaded -- so gate on epoch count
    # before any peak is quoted, or a 3-epoch curve gets reported as a verdict.
    need = args.min_epochs or last_epoch(arms[0][1])
    complete, partial = [], []
    for name, runs in arms:
        ep = last_epoch(runs)
        (complete if ep >= need else partial).append((name, runs, ep))
        print(f'{name:12s} ep{ep:>3}/{need}  {[l for _, _, l in runs]}')
    if partial:
        print('\n' + '!' * 78)
        for name, _, ep in partial:
            print(f'! INCOMPLETE ARM "{name}": only {ep}/{need} epochs -- STILL TRAINING.')
        print('! Excluded from the tables below. Re-run when finished, or pass '
              '--allow_incomplete\n! to inspect the partial curves (peaks from a partial '
              'curve are NOT a result).')
        print('!' * 78)
    if not args.allow_incomplete:
        arms = [(n, r) for n, r, _ in complete]
        if len(arms) < 2:
            raise SystemExit('\nno complete treatment arm yet; nothing to compare.')
    os.makedirs(args.out_dir, exist_ok=True)

    # ---------------- Q1/Q2: per-seed peaks --------------------------------- #
    rows = []
    for arm, runs in arms:
        for i, (curve, _, label) in enumerate(runs):
            for r in READOUTS:
                st = peak_stats(curve, r)
                if st:
                    rows.append(dict(arm=arm, seed=i, run=label, readout=r,
                                     best=st['best'], peak_epoch=st['epoch'],
                                     final=st['final'], decay=st['decay']))
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(args.out_dir, 'summary.csv'), index=False)

    print('\n=== Q1: best-epoch mean AUROC (%) by arm ===')
    agg = summary.groupby(['readout', 'arm'])['best'].agg(['mean', 'std', 'count'])
    print(agg.round(2).to_string())

    print('\n=== Q1b: PAIRED per-seed deltas vs the no-aug control ===')
    print('(the B/E seed-repeat put run-to-run noise at ~0.3-0.6pt; judge against that)')
    for arm, _ in arms[1:]:
        print(f'  [{arm}]')
        for r in READOUTS:
            a = summary[(summary.readout == r) & (summary.arm == arm)].sort_values('seed')
            c = summary[(summary.readout == r) & (summary.arm == 'no-aug')].sort_values('seed')
            n = min(len(a), len(c))
            if n == 0:
                continue
            d = a['best'].values[:n] - c['best'].values[:n]
            print(f'    {r:11s} ' + ' / '.join(f'{x:+.2f}' for x in d) +
                  f'   -> mean {d.mean():+.2f} +/- {d.std(ddof=0):.2f}')

    print('\n=== best-of-readout per arm (how the ladder is actually scored) ===')
    bor = summary.loc[summary.groupby(['arm', 'seed'])['best'].idxmax()]
    print(bor.groupby('arm').apply(
        lambda g: pd.Series({'best_of_readout': g.best.mean(), 'std': g.best.std(ddof=0),
                             'winning_readout': '/'.join(sorted(set(g.readout)))}),
        include_groups=False).round(2).to_string())

    print('\n=== Q2: peak epoch and post-peak decay (the "peaks early" signature) ===')
    pk = summary.groupby(['readout', 'arm'])[['peak_epoch', 'decay']].mean()
    print(pk.round(2).to_string())

    # per-class at the best epoch of the best readout, per arm
    pc_rows = []
    for arm, runs in arms:
        for r in READOUTS:
            vals = [peak_stats(cv, r) for cv, _, _ in runs]
            vals = [v for v in vals if v]
            if not vals:
                continue
            pc = {c: np.mean([v['per_class'][c] for v in vals]) for c in CLASSES}
            pc_rows.append(dict(arm=arm, readout=r, **pc,
                                mean=np.mean([v['best'] for v in vals])))
    per_class = pd.DataFrame(pc_rows)
    per_class.to_csv(os.path.join(args.out_dir, 'per_class.csv'), index=False)
    print('\n=== per-class at each arm\'s best epoch (seed-averaged) ===')
    print(per_class.round(2).to_string(index=False))

    # ---------------- figure ------------------------------------------------ #
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    palette = ['#b0413e', '#2b6cb0', '#2f855a', '#805ad5', '#d69e2e']
    colors = {name: palette[i % len(palette)] for i, (name, _) in enumerate(arms)}

    for ax, r in zip(axes.flat[:2], ['neg_cos', 'maha_embed']):
        for arm, runs in arms:
            for j, (curve, _, _) in enumerate(runs):
                sub = curve[curve.readout == r].sort_values('epoch')
                if sub.empty:
                    continue
                ax.plot(sub.epoch, sub['mean'] * 100, color=colors[arm], alpha=.75, lw=1.4,
                        label=arm if j == 0 else None)
        ax.axhline(STGRAM, ls='--', c='k', lw=1, label='STgram-MFN 90.75')
        ax.set_title(f'Q1/Q2 - {r}: mean AUROC per epoch (3 seeds/arm)')
        ax.set_xlabel('epoch'); ax.set_ylabel('mean AUROC (%)')
        ax.legend(fontsize=8); ax.grid(alpha=.3)

    # Q3: train accuracy — the overfit signature
    ax = axes.flat[2]
    for arm, runs in arms:
        for j, (_, tr, _) in enumerate(runs):
            if tr is None:
                continue
            ax.plot(tr.epoch, tr.train_acc, color=colors[arm], alpha=.75, lw=1.4,
                    label=arm if j == 0 else None)
    ax.set_title('Q3 - train accuracy (23-way section task)')
    ax.set_xlabel('epoch'); ax.set_ylabel('train acc (%)')
    ax.legend(fontsize=8); ax.grid(alpha=.3)

    # peak-epoch scatter
    ax = axes.flat[3]
    for arm, _ in arms:
        s = summary[summary.arm == arm]
        ax.scatter(s.peak_epoch, s.best, c=colors[arm], label=arm, s=45, alpha=.8)
    ax.axhline(STGRAM, ls='--', c='k', lw=1)
    ax.set_title('Q2 - where the peak lands vs how high it is')
    ax.set_xlabel('peak epoch'); ax.set_ylabel('best mean AUROC (%)')
    ax.legend(fontsize=8); ax.grid(alpha=.3)

    fig.tight_layout()
    out = os.path.join(args.out_dir, 'phase0_aug.png')
    fig.savefig(out, dpi=140)
    print(f'\n[fig] {out}')


if __name__ == '__main__':
    main()
