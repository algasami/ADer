"""
Held-out epoch selection — de-optimize the campaign's headline numbers
======================================================================

Every number on this ladder reports `max over epochs` of a metric computed on the TEST set.
That uses test labels to choose the checkpoint, so it answers "how good was the best epoch,
judged with the answers in hand" rather than "how good is the checkpoint you would ship". On
the Phase 2 runs the gap is ~0.7 AUROC — 91.38 test-selected vs 90.70 at the final epoch —
which was the entire apparent margin over STgram-MFN.

This script computes the honest version from `meanid_folds.csv`, which logs the STgram-
convention mean-of-per-ID AUROC on each half of a fixed section x label stratified split of
the test clips:

    fold A -> choose the epoch;  report that epoch's score on fold B   (and vice versa)

Fold B never took part in choosing, so its score is an unbiased estimate of the deploy-time
result. Averaging both directions uses all the data and halves the variance.

Reported alongside, for context:
  * `test-selected` — the optimistic convention (max over epochs on the full test set)
  * `final epoch`   — no selection at all, the pessimistic floor
  * `held-out`      — the honest number, which should land between the two

Note the fundamental constraint: MIMII's train split is normal-only, so AUROC cannot be
computed on it and a "proper" validation set carved from training data is impossible.
Splitting the test set is the standard workaround and is what Rung F already did for its
per-class readout policy.

Usage
-----
    python docs/select_heldout_epoch.py 'runs/section_rungG/rn34as_seed*'
    python docs/select_heldout_epoch.py 'runs/section_rungH/*' --mode section_asnorm
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

STGRAM = 90.75


def one_run(d, mode):
    f = os.path.join(d, 'meanid_folds.csv')
    if not os.path.isfile(f):
        return None
    df = pd.read_csv(f)
    df = df[df['mode'] == mode].sort_values('epoch')
    if df.empty:
        return None
    # A picks -> report B, and B picks -> report A. Neither reported number was involved
    # in its own selection.
    epA = int(df.loc[df.meanid_foldA.idxmax(), 'epoch'])
    epB = int(df.loc[df.meanid_foldB.idxmax(), 'epoch'])
    b_given_a = float(df.loc[df.epoch == epA, 'meanid_foldB'].iloc[0]) * 100
    a_given_b = float(df.loc[df.epoch == epB, 'meanid_foldA'].iloc[0]) * 100
    return dict(
        run=os.path.basename(d),
        test_selected=float(df.meanid_full.max()) * 100,
        final=float(df.meanid_full.iloc[-1]) * 100,
        heldout=0.5 * (b_given_a + a_given_b),
        ep_A=epA, ep_B=epB,
        ep_test=int(df.loc[df.meanid_full.idxmax(), 'epoch']),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('globs', nargs='+', help='run directories (globs ok)')
    ap.add_argument('--mode', default='section_asnorm')
    args = ap.parse_args()

    dirs = [d for g in args.globs for d in sorted(glob.glob(g)) if os.path.isdir(d)]
    rows = [r for r in (one_run(d, args.mode) for d in dirs) if r]
    if not rows:
        raise SystemExit('no runs with meanid_folds.csv found '
                         '(only runs from the Rung H version of the script log it)')
    df = pd.DataFrame(rows)
    print(f'mode = {args.mode}\n')
    print(df.round(2).to_string(index=False))
    print(f'\n{"selection rule":<34s} {"mean":>7s} {"std":>6s}   vs STgram {STGRAM}')
    for lbl, col in (('test-selected (OPTIMISTIC)', 'test_selected'),
                     ('held-out 2-fold (HONEST)', 'heldout'),
                     ('final epoch (no selection)', 'final')):
        v = df[col]
        print(f'{lbl:<34s} {v.mean():>7.2f} {v.std(ddof=0):>6.2f}   {v.mean() - STGRAM:+.2f}')
    print(f'\nselection optimism = {df.test_selected.mean() - df.heldout.mean():+.2f} '
          f'(test-selected minus held-out)')


if __name__ == '__main__':
    main()
