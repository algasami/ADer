"""
Re-score the labels/objective ladder (rungs A-F) under the honest rule
=====================================================================

`docs/FINAL_REPORT.md` §5 leaves one item outstanding: rungs A-F are quoted as **pooled-clip
AUROC at a test-selected epoch**, while the final system is quoted as **mean-of-per-ID AUROC at
a held-out-selected epoch**. Any table mixing them mixes conventions. This script removes the
mismatch by recomputing every rung four ways, from the per-clip scores dumped by
`diagnostics/heldout_eval.ScoreDump`:

                       | epoch+readout picked on TEST | picked on a HELD-OUT half
    pooled-clip AUROC  | the historical convention     |
    mean-of-per-ID     |                               | the honest number

Two selections, not one
-----------------------
Every rung's headline was `max over epochs` **and** `max over readouts` on the test set — the
readout choice (neg_cos vs logit_nll vs maha_embed vs the fusions) is the same sin one level up,
and it is the larger of the two for rungs whose readouts disagree. The honest rule here picks
BOTH on fold A and reports fold B, then both directions are averaged.

Rung F
------
Rung F was never a training run — it is Rung E plus a per-class readout policy chosen on a
held-out half. It has no script and no run directory, so it is reconstructed here from Rung E's
dumped scores by the `perclass` rule below, which is that mechanism made explicit: choose
(epoch, readout) per class on fold A, report that class on fold B.

Fold noise is averaged away
---------------------------
One 2-fold draw is noisy: the reported half is ~5400 clips and the selection half picks among
50 epochs x 3-7 readouts. The estimate is therefore repeated over `--fold_seeds` independent
stratified draws (20 by default) and averaged, which costs milliseconds once scores are on disk
and removes a variance component that would otherwise be mistaken for a rung difference.

Usage
-----
    python docs/rescore_ladder.py                       # all rungs, default run globs
    python docs/rescore_ladder.py --runs 'runs/section_rungB/*_seed*'
"""

import os
import sys
import glob
import argparse
from collections import OrderedDict

import numpy as np
from scipy.stats import rankdata

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from diagnostics.heldout_eval import make_folds_keyed

CLASSES = ["fan", "pump", "slider", "valve", "ToyCar", "ToyConveyor"]
STGRAM = 90.75

# rung -> (label, default run glob). Rung F is derived from E, not trained.
LADDER = OrderedDict([
    ('A', ('frozen encoder + head', 'runs/section_probe/log-Mel*seed*')),
    ('B', ('fine-tuned encoder', 'runs/section_rungB/log-Mel*seed*')),
    ('C', ('Mamba student, frozen enc', 'runs/section_rungC/log-Mel*seed*')),
    ('D', ('joint distill + classify', 'runs/section_rungD/log-Mel*seed*')),
    ('E', ('B+C combined', 'runs/section_rungE/log-Mel*seed*')),
])
# The globs are deliberately loose: the July run directories sit alongside the re-runs under
# the same names, and are excluded automatically because only a re-run has scores_by_epoch.npz.
# the numbers the campaign published, for a reproduction check (pooled-clip, test-selected,
# best readout; seed 0 unless noted). Sources: runs/section_*/*/best_summary.csv.
HISTORICAL = {'A': 80.01, 'B': 85.86, 'C': 84.43, 'D': 83.98, 'E': 86.85}


# --------------------------------------------------------------------------- #
# vectorized AUROC: one pass for all (epoch x readout) score vectors at once
# --------------------------------------------------------------------------- #
def auroc_rows(S, y):
    """AUROC of every row of S (K, n) against labels y (n,). NaN rows -> NaN.

    Rank-based, so it is one sort per row rather than one sklearn call per row: the honest
    rule needs ~40 masks x 29 groups x 150 score vectors per run, which is far too many
    roc_auc_score calls to do naively.
    """
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.full(len(S), np.nan)
    ok = np.isfinite(S).all(axis=1)
    out = np.full(len(S), np.nan)
    if not ok.any():
        return out
    r = rankdata(S[ok], axis=1)                       # average ranks handle ties
    out[ok] = (r[:, y == 1].sum(axis=1) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return out


def group_auroc(S, y, groups, order, mask=None):
    """(K, len(order)) AUROC per group, restricted to `mask`."""
    out = np.full((len(S), len(order)), np.nan)
    for j, g in enumerate(order):
        m = groups == g
        if mask is not None:
            m = m & mask
        if m.sum() < 2:
            continue
        out[:, j] = auroc_rows(S[:, m], y[m])
    return out


# --------------------------------------------------------------------------- #
def load_run(d):
    z = np.load(os.path.join(d, 'scores_by_epoch.npz'), allow_pickle=True)
    S = z['scores']                                    # (E, R, N)
    E, R, N = S.shape
    return dict(
        dir=d, S=S.reshape(E * R, N).astype(np.float64), E=E, R=R,
        epochs=z['epochs'], readouts=[str(r) for r in z['readouts']],
        cls=z['cls'], sec=z['sec'], y=z['y'].astype(int), keys=z['clip_key'],
        # index k of the flattened (epoch, readout) axis -> which epoch / which readout
        k_ep=np.repeat(z['epochs'], R), k_ro=np.tile(np.arange(R), E),
    )


def metrics_for_mask(run, sections, mask):
    """Per-(epoch,readout) class-level scores under both metric conventions.

    Returns (pooled_by_class (K,6), meanid_by_class (K,6)) — the per-class breakdown, because
    the Rung F policy selects per class and the overall figure is just their mean.
    """
    pooled = group_auroc(run['S'], run['y'], run['cls'], CLASSES, mask)
    sec_au = group_auroc(run['S'], run['y'], run['sec'], sections, mask)
    meanid = np.full_like(pooled, np.nan)
    for j, c in enumerate(CLASSES):
        cols = [i for i, s in enumerate(sections) if s.split('/')[0] == c]
        if cols:
            meanid[:, j] = np.nanmean(sec_au[:, cols], axis=1)
    return pooled, meanid


def _argmax_nan(v):
    return int(np.nanargmax(v)) if np.isfinite(v).any() else -1


def selections(per_class_sel, per_class_rep, k_ep, k_ro):
    """The selection rules, given per-class metric arrays on the selection half and on the
    reporting half. Each returns a mean over the 6 classes (in AUROC units, 0-1).

    Three policies of increasing freedom, all chosen on the selection half only:

      global   one (epoch, readout) for every class -- the honest version of each rung's
               published headline, and one deployable model.
      perclass_readout   ONE epoch (so: one checkpoint), but the readout picked per class.
               This is Rung F's actual mechanism.
      perclass_full      epoch AND readout picked per class. Strictly stronger, but it is a
               different artifact -- it needs up to 6 checkpoints kept, so it is an upper
               bound on the policy family rather than a deployable model.
    """
    out = {}
    k = _argmax_nan(np.nanmean(per_class_sel, axis=1))
    out['global'] = float(np.nanmean(per_class_rep[k])) if k >= 0 else np.nan

    # Rung F: freeze the epoch at the globally-selected one, then choose the readout per class
    vals = []
    if k >= 0:
        same_ep = np.where(k_ep == k_ep[k])[0]
        for j in range(per_class_sel.shape[1]):
            sub = per_class_sel[same_ep, j]
            if np.isfinite(sub).any():
                vals.append(per_class_rep[same_ep[int(np.nanargmax(sub))], j])
    out['perclass_readout'] = float(np.nanmean(vals)) if vals else np.nan

    vals = []
    for j in range(per_class_sel.shape[1]):
        kj = _argmax_nan(per_class_sel[:, j])
        if kj >= 0:
            vals.append(per_class_rep[kj, j])
    out['perclass_full'] = float(np.nanmean(vals)) if vals else np.nan
    return out


def score_run(d, fold_seeds):
    run = load_run(d)
    sections = sorted(set(run['sec'].tolist()))
    res = {'dir': d, 'n_epochs': run['E'], 'readouts': run['readouts']}

    full_p, full_m = metrics_for_mask(run, sections, None)
    # The historical convention averaged over whichever classes a readout produced, so the
    # aggregations below are nan-tolerant. That is only faithful if there is nothing to
    # tolerate -- say so loudly if a readout is missing a class.
    n_nan = int(np.isnan(full_p).sum() + np.isnan(full_m).sum())
    res['n_nan_class_cells'] = n_nan
    if n_nan:
        print(f'  [warn] {d}: {n_nan} NaN (epoch x readout x class) cells — a readout is '
              f'undefined for some class; means are taken over the rest')
    for name, arr in (('pooled', full_p), ('meanid', full_m)):
        mean_over_cls = np.nanmean(arr, axis=1)
        k = _argmax_nan(mean_over_cls)
        res[f'{name}_test_selected'] = mean_over_cls[k] * 100
        res[f'{name}_test_ep'] = int(run['k_ep'][k])
        res[f'{name}_test_readout'] = run['readouts'][run['k_ro'][k]]
        # per-class readout at the test-selected epoch: the test-side twin of Rung F's rule
        same_ep = np.where(run['k_ep'] == run['k_ep'][k])[0]
        res[f'{name}_test_perclass_readout'] = float(np.nanmean(
            [np.nanmax(arr[same_ep, j]) for j in range(arr.shape[1])])) * 100
        # per-class oracle on test, for reference (what the cross-rung oracle did)
        res[f'{name}_oracle'] = float(np.nanmean(np.nanmax(arr, axis=0))) * 100
        # final epoch, best readout on test -- isolates epoch selection from readout selection
        last = np.where(run['k_ep'] == run['epochs'][-1])[0]
        res[f'{name}_final'] = float(np.nanmax(mean_over_cls[last])) * 100

    RULES = ('global', 'perclass_readout', 'perclass_full')
    acc = {f'{m}_{r}': [] for m in ('pooled', 'meanid') for r in RULES}
    for fs in fold_seeds:
        fold = make_folds_keyed(run['keys'], run['sec'], run['y'], seed=fs)
        halves = {}
        for h in (0, 1):
            halves[h] = metrics_for_mask(run, sections, fold == h)
        for mi, name in enumerate(('pooled', 'meanid')):
            for a, b in ((0, 1), (1, 0)):                 # A picks -> report B, and vice versa
                s = selections(halves[a][mi], halves[b][mi], run['k_ep'], run['k_ro'])
                for r in RULES:
                    acc[f'{name}_{r}'].append(s[r])
    for k, v in acc.items():
        res[f'{k}_heldout'] = float(np.nanmean(v)) * 100
    return res


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', nargs='*', default=None,
                    help='run-dir globs (default: the ladder A-E)')
    ap.add_argument('--fold_seeds', type=int, default=20)
    ap.add_argument('--out_dir', default='docs/plots/ladder_honest')
    args = ap.parse_args()

    fold_seeds = list(range(args.fold_seeds))
    jobs = ([(None, g) for g in args.runs] if args.runs
            else [(r, g) for r, (_lbl, g) in LADDER.items()])

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for rung, g in jobs:
        dirs = [d for d in sorted(glob.glob(g))
                if os.path.isfile(os.path.join(d, 'scores_by_epoch.npz'))]
        if not dirs:
            print(f'[skip] rung {rung}: no runs with scores_by_epoch.npz at {g}')
            continue
        for d in dirs:
            r = score_run(d, fold_seeds)
            r['rung'] = rung or os.path.basename(os.path.dirname(d))
            rows.append(r)
            print(f"  {r['rung']}  {os.path.basename(d):22s}  "
                  f"pooled/test {r['pooled_test_selected']:.2f}  "
                  f"meanid/heldout {r['meanid_global_heldout']:.2f}")

    if not rows:
        raise SystemExit('nothing to score')

    import pandas as pd
    df = pd.DataFrame(rows)
    csv = os.path.join(args.out_dir, 'per_run.csv')
    df.to_csv(csv, index=False)

    print('\n' + '=' * 100)
    print('LADDER UNDER THE HONEST RULE  (AUROC %, mean +/- std over seeds)')
    print('=' * 100)
    hdr = (f"{'rung':5s} {'n':>2s} | {'pooled/test':>13s} {'(historical)':>13s} | "
           f"{'meanid/test':>12s} {'meanid/heldout':>15s} {'vs STgram':>10s}")
    print(hdr)
    print('-' * len(hdr))
    for rung in df['rung'].unique():
        s = df[df['rung'] == rung]
        hist = HISTORICAL.get(rung)
        def f(col):
            v = s[col]
            return f'{v.mean():.2f}' + (f'+-{v.std(ddof=0):.2f}' if len(v) > 1 else '      ')
        print(f'{rung:5s} {len(s):2d} | {f("pooled_test_selected"):>13s} '
              f'{(f"{hist:.2f}" if hist else "-"):>13s} | {f("meanid_test_selected"):>12s} '
              f'{f("meanid_global_heldout"):>15s} '
              f'{s["meanid_global_heldout"].mean() - STGRAM:>+10.2f}')

    # Rung F = Rung E + a per-class readout policy at one epoch, chosen on the held-out half
    e = df[df['rung'] == 'E']
    if len(e):
        print(f'{"F":5s} {len(e):2d} | {"(from E)":>13s} {"86.62":>13s} | '
              f'{e["meanid_test_perclass_readout"].mean():>12.2f} '
              f'{e["meanid_perclass_readout_heldout"].mean():>15.2f} '
              f'{e["meanid_perclass_readout_heldout"].mean() - STGRAM:>+10.2f}')
        print(f'{"F+":5s} {len(e):2d} | {"(from E)":>13s} {"-":>13s} | '
              f'{e["meanid_oracle"].mean():>12.2f} '
              f'{e["meanid_perclass_full_heldout"].mean():>15.2f} '
              f'{e["meanid_perclass_full_heldout"].mean() - STGRAM:>+10.2f}'
              f'   <- epoch AND readout per class: up to 6 checkpoints, not one model')

    print('\nselection optimism (test-selected minus held-out, mean-of-per-ID):')
    for rung in df['rung'].unique():
        s = df[df['rung'] == rung]
        print(f'  {rung}: {s["meanid_test_selected"].mean() - s["meanid_global_heldout"].mean():+.2f}')
    print(f'\n[out] {csv}')


if __name__ == '__main__':
    main()
