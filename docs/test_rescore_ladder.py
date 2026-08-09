"""Self-checks for the honest re-score of the ladder. Run: `python docs/test_rescore_ladder.py`

Two things could silently invalidate the re-scored table, and neither would raise an error:

1. The `clip_scores` refactor (needed so per-clip scores could be dumped) might not reproduce
   the original `score_epoch` arithmetic — every rung's re-run number would then be measuring
   something subtly different from the published one.
2. The held-out estimator might be selecting on the half it reports. That failure mode looks
   like "selection optimism is ~zero", which is exactly what rungs A/B measured — so the
   finding is only trustworthy if the estimator is shown to detect optimism when it is there.

This campaign has already been bitten twice by silent methodology bugs (`update_log_term`
no-oping on a name mismatch; `SCANS` scrambling non-square feature maps without raising), so
these are checks rather than assumptions.
"""
import os
import sys
import tempfile
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from diagnostics.section_finetune_rungB import clip_scores, auroc_by_class, CLASSES
from diagnostics.frozen_encoder_probe import fit_mahalanobis, maha_score
from diagnostics.heldout_eval import ScoreDump, clip_keys, make_folds_keyed
from docs.rescore_ladder import score_run


def _score_epoch_original(test_pack, sec2idx, s_scale, train_pack, eval_maha):
    """Rung B's scoring exactly as it stood before the refactor (git 8389610..d4d1231)."""
    Ete, Lte, cls_te, sec_te, y_anom = test_pack
    logp = Lte - torch.logsumexp(torch.from_numpy(Lte), dim=1, keepdim=True).numpy()
    assigned = np.array([sec2idx.get(s, -1) for s in sec_te])
    maha_banks = None
    if eval_maha and train_pack is not None:
        Etr, cls_tr = train_pack
        maha_banks = {c: fit_mahalanobis(Etr[cls_tr == c]) for c in CLASSES if (cls_tr == c).any()}
    out = defaultdict(dict)
    for c in CLASSES:
        m = cls_te == c
        y = y_anom[m]
        if y.min() == y.max():
            continue
        a = assigned[m]
        out['neg_cos'][c] = roc_auc_score(y, -(Lte[m][np.arange(len(a)), a] / s_scale))
        out['logit_nll'][c] = roc_auc_score(y, -logp[m][np.arange(len(a)), a])
        if maha_banks is not None and c in maha_banks:
            mu, ic = maha_banks[c]
            out['maha_embed'][c] = roc_auc_score(y, maha_score(Ete[m], mu, ic))
    for r in list(out.keys()):
        out[r]['mean'] = float(np.mean([out[r][c] for c in CLASSES if c in out[r]]))
    return out


def test_refactor_is_exact():
    rng = np.random.default_rng(0)
    N, D = 1200, 16
    cls_te = rng.choice(CLASSES, N)
    sec_te = np.array([f'{c}/id_{rng.integers(0, 2):02d}' for c in cls_te])
    sec2idx = {s: i for i, s in enumerate(sorted(set(sec_te)))}
    pack = (rng.normal(size=(N, D)).astype(np.float32),
            rng.normal(size=(N, len(sec2idx))).astype(np.float32),
            cls_te, sec_te, rng.integers(0, 2, N))
    train = (rng.normal(size=(900, D)).astype(np.float32), rng.choice(CLASSES, 900))

    old = _score_epoch_original(pack, sec2idx, 30.0, train, True)
    new = auroc_by_class(clip_scores(pack, sec2idx, 30.0, train, True), cls_te, pack[4])
    assert sorted(old) == sorted(new), (sorted(old), sorted(new))
    worst = max(abs(old[r][k] - new[r][k]) for r in old for k in old[r])
    assert worst == 0.0, f'refactor changed the numbers by {worst:.3e}'
    print(f'PASS  clip_scores == original score_epoch   (max diff {worst:.1e})')


def test_folds_are_order_invariant():
    rng = np.random.default_rng(1)
    n = 4000
    cls = rng.choice(CLASSES, n)
    sec = np.array([f'{c}/id_{rng.integers(0, 3):02d}' for c in cls])
    y = rng.integers(0, 2, n)
    keys = np.array([f'{c}/test/x/{i}.png' for i, c in enumerate(cls)])
    f = make_folds_keyed(keys, sec, y, seed=0)
    p = rng.permutation(n)
    assert np.array_equal(f[p], make_folds_keyed(keys[p], sec[p], y[p], seed=0)), \
        'fold assignment depends on row order'
    for s in np.unique(sec):                       # per-ID AUROC needs both labels in each half
        for h in (0, 1):
            m = (sec == s) & (f == h)
            assert m.sum() and 0 < y[m].mean() < 1, f'section {s} degenerate in fold {h}'
    assert len(np.unique(clip_keys(['a/fan/test/normal/x.png', 'a/pump/test/normal/x.png']))) == 2, \
        'clip keys collide across machine classes'
    print('PASS  folds are order-invariant, stratified, and keys are unique')


def test_estimator_detects_optimism():
    """Every epoch has the same true signal; the max over epochs is pure winner's curse."""
    rng = np.random.default_rng(0)
    n_per_sec, n_ep = 400, 50
    cls, sec, y, keys = [], [], [], []
    for c in CLASSES:
        for i in range(3):
            cls += [c] * n_per_sec
            sec += [f'{c}/id_{i:02d}'] * n_per_sec
            y += [0] * (n_per_sec // 2) + [1] * (n_per_sec // 2)
            keys += [f'{c}/test/x/{i}_{j}.png' for j in range(n_per_sec)]
    cls, sec, y, keys = map(np.array, (cls, sec, y, keys))

    d = tempfile.mkdtemp()
    dump = ScoreDump(d, keys, cls, sec, y)
    signal = y * 0.55
    for ep in range(1, n_ep + 1):
        dump.add(ep, {'r': signal + rng.normal(size=len(y))})

    r = score_run(d, list(range(20)))
    opt = r['meanid_test_selected'] - r['meanid_global_heldout']
    assert opt > 0.4, f'estimator missed known winners-curse inflation (found {opt:+.2f})'
    assert r['meanid_final'] <= r['meanid_global_heldout'] + 1e-9, \
        'held-out fell below the no-selection floor'
    print(f'PASS  estimator exposes a planted winners curse '
          f'(test {r["meanid_test_selected"]:.2f} > held-out {r["meanid_global_heldout"]:.2f} '
          f'> final {r["meanid_final"]:.2f}, optimism {opt:+.2f})')


if __name__ == '__main__':
    test_refactor_is_exact()
    test_folds_are_order_invariant()
    test_estimator_detects_optimism()
    print('\nall self-checks passed')
