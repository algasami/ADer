"""
Held-out re-scoring support for the labels/objective ladder (rungs A-F)
======================================================================

`docs/FINAL_REPORT.md` §5 lists four methodology corrections. Two of them invalidate every
number rungs A-F ever reported:

  5.2  STgram-MFN reports the **mean of per-machine-ID AUROCs**; the ladder computed the
       **pooled-clip AUROC**. Different quantities.
  5.3  "Best epoch" was chosen on the **test set** throughout — and so was "best readout",
       which is the same sin one level up and is what every rung's headline number did.

Re-scoring the ladder honestly needs per-clip scores at every epoch, and those were never
persisted: the rungs logged only per-class AUROC aggregates, kept no checkpoints, and (for
the folds) no split at all. So the runs have to be repeated. This module is what makes the
repeat sufficient — dump the score of every test clip at every epoch, once, and every
selection rule afterwards is post-hoc arithmetic on `scores_by_epoch.npz`.

Two deliberate choices
----------------------
**Folds are keyed by clip path, not by row index.** `asnorm.make_folds` shuffles positional
indices, so the split it returns depends on DataLoader order — fine within one script, but it
would silently give rungs A-F different partitions of the same clips and make their honest
numbers incomparable. Keying on the path makes the split a property of the dataset, so every
rung (and any future run on the same clips) held out exactly the same halves.

**The fold seed is a nuisance parameter, so average it away.** One 2-fold draw carries real
sampling noise: the reported half is ~5400 clips and the selection half picks among ~50
epochs x 3-7 readouts. With per-clip scores on disk, repeating the estimate over many draws
costs milliseconds and removes a source of variance that would otherwise be indistinguishable
from a rung difference. `docs/rescore_ladder.py` averages over 20 draws by default.
"""

import os

import numpy as np


def clip_keys(paths):
    """Stable per-clip identifiers: `<class>/<split>/<label>/<file>`, the meta.json form.

    Four components, not fewer: MIMII filenames repeat verbatim across machine classes
    (`fan/test/normal/id_00_00000000.png` and `pump/test/normal/id_00_00000000.png` both
    exist), so a shorter key collides on ~half the split and would put two different clips
    in the same fold bucket.
    """
    out = []
    for p in paths:
        p = str(p).replace('\\', '/').rstrip('/')
        parts = p.split('/')
        out.append('/'.join(parts[-4:]) if len(parts) >= 4 else p)
    return np.array(out)


def make_folds_keyed(keys, sec, y_anom, seed=0):
    """Order-invariant 2-fold split of the test clips, stratified by section x label.

    Same contract as `asnorm.make_folds` (choose the epoch on one half, report the other),
    but the assignment depends only on the clip identity, so two scripts that enumerate the
    test split in different orders still hold out the same clips.
    """
    keys = np.asarray(keys)
    sec = np.asarray(sec)
    y_anom = np.asarray(y_anom)
    fold = np.zeros(len(keys), dtype=np.int8)
    rng = np.random.default_rng(seed)
    for s in np.unique(sec):
        for lab in (0, 1):
            idx = np.where((sec == s) & (y_anom == lab))[0]
            if len(idx) == 0:
                continue
            idx = idx[np.argsort(keys[idx], kind='stable')]   # canonical order first
            idx = idx.copy()
            rng.shuffle(idx)
            fold[idx[len(idx) // 2:]] = 1
    return fold


class ScoreDump:
    """Accumulates per-clip test scores for every readout at every epoch.

    Written to `<out_dir>/scores_by_epoch.npz` after each epoch, so a run that dies at epoch
    37 still leaves 36 usable epochs behind. Size is trivial: 50 epochs x 7 readouts x 10868
    clips of float32 is ~15 MB.

    Readouts may appear at some epochs and not others (a Maha bank needs >= 2 samples per
    class); missing entries are NaN, which every downstream AUROC helper already skips.
    """

    def __init__(self, out_dir, keys, cls, sec, y_anom, name='scores_by_epoch.npz'):
        self.path = os.path.join(out_dir, name)
        self.keys = np.asarray(keys)
        self.cls = np.asarray(cls)
        self.sec = np.asarray(sec)
        self.y = np.asarray(y_anom).astype(np.int8)
        self.epochs = []
        self.rows = []          # list of {readout: array}
        self.readouts = []      # union, in first-seen order

    def add(self, epoch, scores):
        """`scores`: {readout: per-clip array of length n_clips}, higher = more anomalous."""
        row = {}
        for r, v in scores.items():
            v = np.asarray(v, dtype=np.float64)
            if v.shape != self.y.shape:
                raise ValueError(f"readout '{r}': got {v.shape}, expected {self.y.shape} "
                                 f"(one score per test clip, in test-split order)")
            if r not in self.readouts:
                self.readouts.append(r)
            row[r] = v
        self.epochs.append(int(epoch))
        self.rows.append(row)
        self.save()

    def save(self):
        n = len(self.y)
        mat = np.full((len(self.rows), len(self.readouts), n), np.nan, dtype=np.float32)
        for i, row in enumerate(self.rows):
            for j, r in enumerate(self.readouts):
                if r in row:
                    mat[i, j] = row[r]
        np.savez_compressed(
            self.path, scores=mat,
            epochs=np.array(self.epochs, dtype=np.int32),
            readouts=np.array(self.readouts),
            clip_key=self.keys, cls=self.cls, sec=self.sec, y=self.y,
        )
