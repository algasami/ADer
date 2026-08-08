"""
AS-norm / per-section scoring — Phase 2, step 1 (branch `aug-ast-phases`)
========================================================================

Every readout in this campaign fits its Mahalanobis bank **per machine class** (fan, pump, ...)
and scores a clip against that. But a class pools 3-4 machine *units* (`id_00`, `id_02`, ...)
whose normal sound differs a lot, and the per-ID breakdowns show the difficulty is wildly
uneven within a class -- ToyConveyor `id_02` is the worst unit in every run on this ladder AND
in STgram-MFN's own results. Pooling those units means:

  * the bank is fit on a mixture, so its distances mean different things per unit, and
  * the class-level AUROC is dominated by whichever unit has the widest score spread.

This module adds the two orthogonal fixes and lets them be measured separately:

  BANK   `class`   -- one Maha bank per class            (the existing behaviour)
         `section` -- one Maha bank per type x id section

  NORM   `raw`     -- use the distance directly          (the existing behaviour)
         `asnorm`  -- z-normalize each clip's score using the mean/std of the scores of the
                      NORMAL TRAIN clips of that clip's own section

AS-norm makes scores commensurable across sections, which matters twice over: it is the fix for
the uneven-difficulty problem above, and it is a prerequisite for any later score fusion (a
weighted sum of two models' raw distances is otherwise dominated by whichever has the larger
scale).

Caveat, stated because it flatters the numbers slightly: the normalization statistics are fit on
the same normal-train clips the bank is fit on, so the train scores are optimistically low. That
is the standard AS-norm formulation, and the test clips are still strictly held out, but a
held-out normal split would be cleaner.
"""

import numpy as np

from diagnostics.frozen_encoder_probe import fit_mahalanobis, maha_score

MODES = ('class_raw', 'class_asnorm', 'section_raw', 'section_asnorm')


def _banks(E, keys):
    """Fit one Maha bank per unique key. Returns {key: (mu, inv_cov)}."""
    out = {}
    for k in np.unique(keys):
        m = keys == k
        if m.sum() >= 2:                 # a covariance needs at least two samples
            out[k] = fit_mahalanobis(E[m])
    return out


def _apply(E, keys, banks):
    """Score each row against its own key's bank. NaN where the key has no bank."""
    s = np.full(len(E), np.nan, dtype=np.float64)
    for k, (mu, ic) in banks.items():
        m = keys == k
        if m.any():
            s[m] = maha_score(E[m], mu, ic)
    return s


def score_all_modes(Ete, cls_te, sec_te, Etr, cls_tr, sec_tr):
    """Return {mode: per-clip test score} for every bank x norm combination.

    Higher = more anomalous, for every mode (z-normalization preserves that, since sigma > 0).
    """
    bank_keys = {'class': (cls_tr, cls_te), 'section': (sec_tr, sec_te)}
    out = {}
    for bank, (ktr, kte) in bank_keys.items():
        banks = _banks(Etr, ktr)
        s_te = _apply(Ete, kte, banks)
        out[f'{bank}_raw'] = s_te
        # AS-norm: per-section statistics of the TRAIN scores under this same bank
        s_tr = _apply(Etr, ktr, banks)
        z = np.full_like(s_te, np.nan)
        for sec in np.unique(sec_te):
            mtr, mte = sec_tr == sec, sec_te == sec
            if not mte.any():
                continue
            ref = s_tr[mtr]
            ref = ref[np.isfinite(ref)]
            if len(ref) < 2:
                z[mte] = s_te[mte]       # cannot normalize; fall back to the raw score
                continue
            mu, sd = float(ref.mean()), float(ref.std())
            z[mte] = (s_te[mte] - mu) / (sd if sd > 1e-12 else 1.0)
        out[f'{bank}_asnorm'] = z
    return out


def auroc_by_class(scores, cls_te, y_anom, classes, roc_auc_score):
    """{class: auroc} + 'mean', skipping classes with a single label or non-finite scores."""
    out = {}
    for c in classes:
        m = (cls_te == c) & np.isfinite(scores)
        if m.sum() < 2:
            continue
        y = y_anom[m]
        if y.min() == y.max():
            continue
        out[c] = roc_auc_score(y, scores[m])
    if out:
        out['mean'] = float(np.mean([out[c] for c in classes if c in out]))
    return out


def auroc_by_section(scores, sec_te, y_anom, roc_auc_score):
    """{section: (n, auroc)} -- the per-ID view, for the ToyConveyor id_02 question."""
    out = {}
    for s in np.unique(sec_te):
        m = (sec_te == s) & np.isfinite(scores)
        if m.sum() < 2:
            continue
        y = y_anom[m]
        if y.min() == y.max():
            continue
        out[s] = (int(m.sum()), roc_auc_score(y, scores[m]))
    return out
