"""
Modular test-time score readouts ("scorers") for the MambaAD teacher->student pipeline.
=======================================================================================

Motivation
----------
`diagnostics/frozen_encoder_probe.py` + `diagnostics/student_feature_probe.py` found that
scoring MIMII clips by a **Mahalanobis / kNN distance on globally-average-pooled features**
(teacher *or* student) separates normal vs. anomalous clips better than MambaAD's native
per-pixel cosine-residual `sp_max`/`sp_mean` readout. That is a *readout* change, not a
training change: the same trained checkpoint can be re-scored several ways.

This module turns the readout into a swappable knob so the finding can be run as a clean
ablation instead of only inside a one-off probe script. A scorer takes the teacher/student
feature lists MambaAD already produces and returns a per-image anomaly map that the existing
`util.metric.Evaluator` consumes unchanged.

Design
------
* `CosResidualScorer` — the historical default. Wraps `Evaluator.cal_anomaly_map`, so its
  numbers are byte-for-byte the old path. Needs no fitting.
* `MahaScorer` / `KNNScorer` — fit a per-class bank on the **normal-only train split**
  (`needs_fit=True`; the trainer runs one extra forward pass), then score each test clip by
  distance to that bank. `source='student'` scores the decoder's reconstructed features;
  `source='teacher'` scores the frozen encoder features (fit is then static across epochs).

Image-level scorers have no spatial map, so they broadcast the per-clip scalar into a constant
anomaly map: `sp_max == sp_mean == score`. Both metric families therefore report the same
number, and the 42-column `metric.txt` / existing plotting tooling keep working untouched.

Wiring: configs set `ABL_SCORER` (see `mimii/_base.py`), which becomes `cfg.scorer`;
`MAMBAADTrainer` builds it via `get_scorer` and calls it in `test()`. `cfg.scorer` absent
=> `CosResidualScorer`, so every non-MIMII config is unaffected.
"""

import numpy as np
import torch
import torch.nn.functional as F

from util.registry import Registry
from util.metric import Evaluator

SCORER = Registry('Scorer')


def _gap_concat(feats):
    """Global-average-pool each feature tap and concat -> (B, sum_C) tensor on feats' device.
    Matches the `*_concat` feature used by the frozen/student probes so numbers are comparable."""
    pooled = [f.mean(dim=(2, 3)) for f in feats]
    return torch.cat(pooled, dim=1)


def _const_amap(scores, out_size):
    """Broadcast a per-image scalar score into a constant (B, H, W) map so that the evaluator's
    sp_max and sp_mean pooling both collapse to the scalar."""
    b = scores.shape[0]
    m = np.asarray(scores, dtype=np.float32).reshape(b, 1, 1)
    return np.ascontiguousarray(np.broadcast_to(m, (b, out_size[0], out_size[1])))


class BaseScorer:
    # needs_fit: trainer runs a normal-train pass (reset -> fit_batch* -> finalize_fit) first.
    needs_fit = False
    # static_fit: the fit depends only on the frozen teacher, so it is safe to fit once and
    # reuse across test epochs (the trainer honours this to avoid refitting every eval).
    static_fit = False

    def reset(self):
        pass

    def fit_batch(self, feats_t, feats_s, cls_names):
        pass

    def finalize_fit(self):
        pass

    def score_batch(self, feats_t, feats_s, cls_names, out_size):
        """feats_*: list of (B, C, h, w) tensors. cls_names: (B,) np.ndarray of str.
        out_size: [H, W]. Returns a (B, H, W) numpy anomaly map (higher = more anomalous)."""
        raise NotImplementedError


@SCORER.register_module
class CosResidualScorer(BaseScorer):
    """Native MambaAD readout: per-pixel 1 - cos(ft, fs) summed over taps -> anomaly map.
    Numerically identical to the historical `MAMBAADTrainer.test()` path."""
    needs_fit = False

    def __init__(self, amap_mode='add', gaussian_sigma=4, uni_am=False, use_cos=True):
        self.amap_mode = amap_mode
        self.gaussian_sigma = gaussian_sigma
        self.uni_am = uni_am
        self.use_cos = use_cos

    def score_batch(self, feats_t, feats_s, cls_names, out_size):
        amap, _ = Evaluator.cal_anomaly_map(
            feats_t, feats_s, out_size, uni_am=self.uni_am, use_cos=self.use_cos,
            amap_mode=self.amap_mode, gaussian_sigma=self.gaussian_sigma)
        return amap


class _FeatureBankScorer(BaseScorer):
    """Shared machinery for global-feature distance scorers (Maha / kNN).

    Fits a per-class bank on GAP-concat features of the normal-only train split, then scores
    each test clip by distance to its class bank. `source` picks which feature stream to use.
    """
    needs_fit = True

    def __init__(self, source='student'):
        assert source in ('student', 'teacher'), f"source must be student|teacher, got {source}"
        self.source = source
        self.static_fit = (source == 'teacher')   # teacher is frozen -> fit once
        self._buf = {}    # cls -> list of (b, D) numpy chunks (during fit)
        self._bank = {}   # cls -> fitted bank object

    def _select(self, feats_t, feats_s):
        return feats_t if self.source == 'teacher' else feats_s

    def reset(self):
        self._buf, self._bank = {}, {}

    @torch.no_grad()
    def fit_batch(self, feats_t, feats_s, cls_names):
        vecs = _gap_concat(self._select(feats_t, feats_s)).cpu().numpy()
        for c in np.unique(cls_names):
            self._buf.setdefault(c, []).append(vecs[cls_names == c])

    def finalize_fit(self):
        for c, chunks in self._buf.items():
            self._bank[c] = self._fit_one(np.concatenate(chunks, axis=0))
        self._buf = {}

    @torch.no_grad()
    def score_batch(self, feats_t, feats_s, cls_names, out_size):
        vecs = _gap_concat(self._select(feats_t, feats_s)).cpu().numpy()
        scores = np.zeros(vecs.shape[0], dtype=np.float32)
        for c in np.unique(cls_names):
            if c not in self._bank:      # class unseen at fit time (shouldn't happen) -> 0
                continue
            m = cls_names == c
            scores[m] = self._score_one(c, vecs[m])
        return _const_amap(scores, out_size)

    def _fit_one(self, cat):
        raise NotImplementedError

    def _score_one(self, c, vecs):
        raise NotImplementedError


@SCORER.register_module
class MahaScorer(_FeatureBankScorer):
    """Mahalanobis distance on GAP-concat features (Rippel et al. / PaDiM-image), per class.
    Ledoit-Wolf-style shrinkage toward a scaled identity keeps the covariance invertible."""

    def __init__(self, source='student', shrinkage=0.01):
        super().__init__(source)
        self.shrinkage = shrinkage

    def _fit_one(self, cat):
        mu = cat.mean(axis=0)
        xc = cat - mu
        cov = (xc.T @ xc) / max(len(cat) - 1, 1)
        d = cov.shape[0]
        tr = np.trace(cov) / d
        a = self.shrinkage
        cov = (1 - a) * cov + a * tr * np.eye(d) + 1e-6 * np.eye(d)
        inv = np.linalg.pinv(cov).astype(np.float32)
        return mu.astype(np.float32), inv

    def _score_one(self, c, vecs):
        mu, inv = self._bank[c]
        xc = vecs - mu
        return np.sqrt(np.einsum('ni,ij,nj->n', xc, inv, xc, optimize=True)).astype(np.float32)


@SCORER.register_module
class KNNScorer(_FeatureBankScorer):
    """Mean cosine distance to the k nearest normal train clips on GAP-concat features
    (DN2 / SPADE-image), per class. Higher = more anomalous."""

    def __init__(self, source='student', k=5):
        super().__init__(source)
        self.k = k

    def _fit_one(self, cat):
        return torch.from_numpy(cat).float()   # keep bank on CPU; moved to device at score time

    def _score_one(self, c, vecs):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tr = F.normalize(self._bank[c].to(device), dim=1)
        te = F.normalize(torch.from_numpy(vecs).float().to(device), dim=1)
        kk = min(self.k, tr.shape[0])
        out = np.empty(len(vecs), dtype=np.float32)
        for i in range(0, te.shape[0], 256):
            chunk = te[i:i + 256]
            dist = 1.0 - chunk @ tr.T
            topk = dist.topk(kk, dim=1, largest=False).values
            out[i:i + chunk.shape[0]] = topk.mean(dim=1).cpu().numpy()
        return out


def get_scorer(cfg_scorer):
    """Build a scorer from a `cfg.scorer` Namespace (type + kwargs). None -> default cos-residual,
    so configs that never set `ABL_SCORER` behave exactly as before."""
    if cfg_scorer is None:
        return CosResidualScorer()
    name = getattr(cfg_scorer, 'type', None) or cfg_scorer['type']
    kwargs = getattr(cfg_scorer, 'kwargs', None)
    if kwargs is None:
        kwargs = {}
    return SCORER.get_module(name)(**kwargs)
