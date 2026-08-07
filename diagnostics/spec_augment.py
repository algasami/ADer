"""
Spectrogram-domain batch augmentation — shared by the augmentation/AST phases
=============================================================================

Every rung of the labels/objective ladder (A-F) trained with ZERO augmentation:
`MambaAD/configs/mambaad/mimii/_base.py` sets `test_transforms = train_transforms`
(the same list object), i.e. a deterministic Resize->CenterCrop->ToTensor->Normalize.
That is a plausible partial explanation for two ladder-wide symptoms -- `logit_nll`
hitting 99% train accuracy by epoch 5, and "AUROC peaks early then decays" -- so
Phase 0 of this branch measures augmentation as its own lever before the encoder swap.

Everything here operates on an already-normalized GPU batch (B, C, H, W) so it can be
dropped into a training loop without touching the shared config/engine (which would
silently change every other config in the repo).

    AXIS CONVENTION: the MIMII PNGs are (W=313 time, H=128 mel) -> after the config's
    Resize(256, 256) the tensor is (B, C, H=256 freq, W=256 time). So dim 2 is
    FREQUENCY and dim 3 is TIME. Masking widths below are quoted in *post-resize*
    rows/cols; 1 mel bin = 2 rows.

Why these three and not the usual audio set:
  * time crop / time shift  -- MIMII clips are 10 s of quasi-stationary machine noise,
    so a random sub-window is label-preserving and multiplies effective data. Safest,
    highest-value augmentation for this dataset.
  * SpecAugment             -- time masking is safe for stationary sources; FREQUENCY
    masking is deliberately kept narrow, because machine identity *is* a spectral
    signature and the training objective is machine-section discrimination. Masking
    wide frequency bands teaches invariance to the exact cue the loss depends on.
  * mixup                   -- pairs well with a cosine-margin loss; needs the two-label
    treatment in `mixup_arcface_loss` below, not a plain CE on a mixed target.

Deliberately NOT included: pitch shift, time stretch, speed perturbation, frequency
warp. For rotating machinery these move the fundamental and its harmonics, i.e. they
change the machine's identity -- which is the label. (They are useful for this task only
in the inverted role of *pseudo-anomaly* generation, which is a separate experiment.)
"""

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
def rand_time_crop(x, min_frac):
    """Per-sample random crop along TIME (dim 3), resampled back to full width.

    Implemented as a batched affine resample so every sample in the batch gets its own
    window in one kernel. `min_frac`=1.0 (or None/0) is a no-op.
    """
    if not min_frac or min_frac >= 1.0:
        return x
    b = x.size(0)
    frac = torch.empty(b, device=x.device).uniform_(min_frac, 1.0)
    # keep the window inside the clip: |center| <= 1 - frac in normalized coords
    center = (torch.rand(b, device=x.device) * 2.0 - 1.0) * (1.0 - frac)
    theta = torch.zeros(b, 2, 3, device=x.device, dtype=x.dtype)
    theta[:, 0, 0] = frac      # time scale  (< 1 => zoom in on a sub-window)
    theta[:, 0, 2] = center    # time offset
    theta[:, 1, 1] = 1.0       # frequency axis untouched
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode='bilinear', padding_mode='reflection',
                         align_corners=False)


def rand_time_shift(x, max_frac):
    """Per-sample circular shift along TIME. Label-preserving for stationary sources."""
    if not max_frac:
        return x
    w = x.size(3)
    max_px = int(round(max_frac * w))
    if max_px < 1:
        return x
    shifts = torch.randint(-max_px, max_px + 1, (x.size(0),), device=x.device)
    idx = (torch.arange(w, device=x.device).view(1, -1) - shifts.view(-1, 1)) % w
    idx = idx.view(-1, 1, 1, w).expand(-1, x.size(1), x.size(2), -1)
    return x.gather(3, idx)


def _mask_axis(x, dim, n_masks, max_w, value=0.0):
    """Zero out `n_masks` random bands of width U[0, max_w] along `dim`, per sample."""
    if n_masks < 1 or max_w < 1:
        return x
    b, length = x.size(0), x.size(dim)
    ar = torch.arange(length, device=x.device).view(1, -1)
    shape = [b, 1, 1, 1]
    shape[dim] = length
    for _ in range(n_masks):
        w = torch.randint(0, max_w + 1, (b,), device=x.device)
        span = (length - w).clamp(min=1).float()
        start = (torch.rand(b, device=x.device) * span).long()
        band = (ar >= start.view(-1, 1)) & (ar < (start + w).view(-1, 1))
        x = x.masked_fill(band.view(shape), value)
    return x


def spec_augment(x, n_freq=0, freq_w=0, n_time=0, time_w=0):
    """SpecAugment masking. Widths are in post-resize rows (freq) / cols (time).

    Masked cells are set to 0, which is the per-channel *mean* in the normalized space
    the batch arrives in -- the standard choice.
    """
    x = _mask_axis(x, 2, n_freq, freq_w)   # dim 2 = frequency
    x = _mask_axis(x, 3, n_time, time_w)   # dim 3 = time
    return x


# --------------------------------------------------------------------------- #
def mixup_batch(x, alpha):
    """Standard mixup with one lambda per batch.

    Returns (mixed_x, perm, lam). The caller keeps BOTH label sets and mixes the
    *losses* -- see `mixup_arcface_loss` for why a mixed one-hot target is wrong here.
    """
    if not alpha:
        return x, None, 1.0
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    lam = max(lam, 1.0 - lam)          # keep the dominant label dominant
    perm = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1.0 - lam) * x[perm], perm, lam


def mixup_arcface_loss(model, emb, y_a, y_b, lam, ce):
    """Mixup loss for an ArcFace/cosine-margin head.

    A margin head needs to know WHICH class to push the margin against, so a mixed
    one-hot target is not well defined. The correct form applies the margin once per
    label set and mixes the two cross-entropies -- two cheap head evaluations, one
    shared backbone forward.
    """
    if y_b is None or lam >= 1.0:
        return ce(model.logits(emb, y_a), y_a)
    return lam * ce(model.logits(emb, y_a), y_a) + \
        (1.0 - lam) * ce(model.logits(emb, y_b), y_b)


# --------------------------------------------------------------------------- #
class BatchAug:
    """Bundle of the geometric/masking augs, applied to a normalized GPU batch."""

    def __init__(self, time_crop_min=0.0, time_shift=0.0,
                 n_freq=0, freq_w=0, n_time=0, time_w=0):
        self.time_crop_min = time_crop_min
        self.time_shift = time_shift
        self.n_freq, self.freq_w = n_freq, freq_w
        self.n_time, self.time_w = n_time, time_w

    @property
    def enabled(self):
        return bool(self.time_crop_min or self.time_shift or
                    (self.n_freq and self.freq_w) or (self.n_time and self.time_w))

    def __call__(self, x):
        if not self.enabled:
            return x
        x = rand_time_crop(x, self.time_crop_min)
        x = rand_time_shift(x, self.time_shift)
        x = spec_augment(x, self.n_freq, self.freq_w, self.n_time, self.time_w)
        return x

    def describe(self):
        if not self.enabled:
            return 'none'
        return (f'time_crop_min={self.time_crop_min} time_shift={self.time_shift} '
                f'freq_mask={self.n_freq}x{self.freq_w} time_mask={self.n_time}x{self.time_w}')

    def as_dict(self):
        return dict(time_crop_min=self.time_crop_min, time_shift=self.time_shift,
                    n_freq=self.n_freq, freq_w=self.freq_w,
                    n_time=self.n_time, time_w=self.time_w)
