"""
Generate STgram images for MambaAD
Has a few params down there

Channel scaling is FIXED and GLOBAL: per-channel value percentiles are estimated once
over train/normal clips of all categories, then applied to every clip (with clipping).
Per-clip min-max scaling is deliberately avoided — it erases absolute level cues and
makes channel scales inconsistent across clips.
"""
import argparse
import glob
import json
import os
import re
import sys

import cv2
import librosa
import numpy as np
import torch

# import stgram-mfn front-end
sys.path.insert(0, 'STgram-MFN')
from net import TgramNet          # noqa: E402
from utils import Wave2Mel        # noqa: E402

# audio processing parameters -- must match the checkpoint @ STgram-MFN/config.yaml
SR = 16000
N_FFT = 1024
WIN_LENGTH = 1024
HOP_LENGTH = 512
N_MELS = 128
POWER = 2.0
SECS = 10
N_SAMPLES = SR * SECS         # 160000 -> 313 frames (TgramNet hardcoded LayerNorm(313))

DEFAULT_CKPT = 'STgram-MFN/runs/STgram-MFN(m=0.7,s=30)/model/best_checkpoint.pth.tar'
SRC_ROOT = 'data/dcase-2020'

# automatically cropped to 10 s to match STgram-MFN's dataloader
CATS = [('data_fan/fan', 'fan'),
        ('data_pump/pump', 'pump'),
        ('data_slider/slider', 'slider'),
        ('data_valve/valve', 'valve'),
        ('data_ToyCar/ToyCar', 'ToyCar'),
        ('data_ToyConveyor/ToyConveyor', 'ToyConveyor')]
SPLITS = ['train', 'test']
# labels of (input prefix, output dir)
LABELS = [('normal', 'normal'), ('anomaly', 'abnormal')]

CHANNELS = ['sgram', 'tgram', 'delta']
# stride used when pooling values for the percentile estimate (313 is prime, no aliasing)
VALUE_SUBSAMPLE = 61
# percentiles defining the global range; values outside are clipped
PCT_LO, PCT_HI = 0.1, 99.9


def scale_fixed(X, lo, hi):
    """Map the fixed global range [lo, hi] to [0, 255], clipping outliers."""
    X = np.clip(X, lo, hi)
    return (X - lo) / (hi - lo) * 255.0


def load_tgramnet(ckpt_path):
    """load TgramNet weights by filtering `tgramnet.*` subset out of the full STgramMFN checkpoint."""
    state = torch.load(ckpt_path, map_location='cpu')['model']
    tg_state = {k[len('tgramnet.'):]: v for k, v in state.items() if k.startswith('tgramnet.')}
    if not tg_state:
        raise RuntimeError(f'No tgramnet.* keys found in checkpoint {ckpt_path}')
    tgramnet = TgramNet(mel_bins=N_MELS, win_len=WIN_LENGTH, hop_len=HOP_LENGTH)
    tgramnet.load_state_dict(tg_state, strict=True)
    tgramnet.eval().cuda()
    norm = sum(p.detach().float().norm().item() ** 2 for p in tgramnet.parameters()) ** 0.5
    print(f'Loaded TgramNet: {len(tg_state)} tensors, param L2 norm = {norm:.4f}')
    return tgramnet


def crop(y):
    """force-crop to N_SAMPLES, matching STgram-MFN's dataloader
    (STgram-MFN/dataset.py:28: `x = x[: self.args.sr * self.args.secs]`).
    all dcase-2020 clips are >= 10 s, so this yields exactly 313 frames for TgramNet's LayerNorm(313)."""
    return y[:N_SAMPLES]


@torch.no_grad()
def make_feats(wav_path, wav2mel, tgramnet):
    y = librosa.load(wav_path, sr=SR, mono=True)[0]
    y = crop(y)
    x = torch.from_numpy(y).float()

    sgram = wav2mel(x).cpu().numpy()                              # (128, 313), absolute dB
    tg = tgramnet(x.view(1, 1, -1).cuda())[0].cpu().numpy()      # (128, 313)
    delta = librosa.feature.delta(sgram)
    return {'sgram': sgram, 'tgram': tg, 'delta': delta}


def compute_stats(src_root, wav2mel, tgramnet, stride):
    """Global per-channel [lo, hi] percentiles over train/normal clips of all categories."""
    pooled = {ch: [] for ch in CHANNELS}
    for in_cat, out_cat in CATS:
        wavs = sorted(glob.glob(os.path.join(src_root, in_cat, 'train', 'normal_*.wav')))
        wavs = wavs[::stride]
        print(f'stats: {out_cat}: sampling {len(wavs)} train/normal clips')
        for wav_path in wavs:
            feats = make_feats(wav_path, wav2mel, tgramnet)
            for ch, arr in feats.items():
                pooled[ch].append(arr.ravel()[::VALUE_SUBSAMPLE].copy())
    stats = {}
    for ch, chunks in pooled.items():
        vals = np.concatenate(chunks)
        lo, hi = np.percentile(vals, [PCT_LO, PCT_HI])
        stats[ch] = dict(lo=float(lo), hi=float(hi))
        print(f'stats: {ch}: lo={lo:.3f} hi={hi:.3f} (from {vals.size} values)')
    return stats


def build_image(feats, stats, third):
    chans = ['sgram', 'tgram', 'sgram' if third == 'sgram' else 'delta']
    img = np.stack([scale_fixed(feats[c], stats[c]['lo'], stats[c]['hi']) for c in chans], axis=-1)
    img = np.rint(img).astype(np.uint8)
    img = np.flip(img, axis=0)                                   # vertical flip (matches existing pipeline)
    return img


def main():
    parser = argparse.ArgumentParser(description='Generate STgram 3-channel PNGs for MambaAD.')
    parser.add_argument('--out-sgram', default='data/dcase-2020-stgram',
                        help='Output root for the [Sgram, Tgram, Sgram] variant (empty string to skip).')
    parser.add_argument('--out-delta', default='data/dcase-2020-stgram-delta',
                        help='Output root for the [Sgram, Tgram, delta] variant (empty string to skip).')
    parser.add_argument('--ckpt', default=DEFAULT_CKPT, help='STgram-MFN checkpoint (.pth.tar) with tgramnet.* keys.')
    parser.add_argument('--src-root', default=SRC_ROOT, help='Raw wav dataset root.')
    parser.add_argument('--stats-stride', type=int, default=4,
                        help='Use every Nth train/normal clip for the global scaling statistics.')
    parser.add_argument('--stats-json', default='',
                        help='Reuse a previously saved scaling.json instead of recomputing statistics.')
    args = parser.parse_args()

    outputs = [(third, root) for third, root in [('sgram', args.out_sgram), ('delta', args.out_delta)] if root]
    if not outputs:
        raise SystemExit('Nothing to do: both --out-sgram and --out-delta are empty.')

    wav2mel = Wave2Mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS, win_length=WIN_LENGTH,
                       hop_length=HOP_LENGTH, power=POWER)
    tgramnet = load_tgramnet(args.ckpt)

    if args.stats_json:
        stats = json.load(open(args.stats_json))
        print(f'Loaded scaling stats from {args.stats_json}: {stats}')
    else:
        stats = compute_stats(args.src_root, wav2mel, tgramnet, args.stats_stride)
    for _, out_root in outputs:
        os.makedirs(out_root, exist_ok=True)
        with open(os.path.join(out_root, 'scaling.json'), 'w') as f:
            json.dump(stats, f, indent=4)

    total = 0
    for in_cat, out_cat in CATS:
        for split in SPLITS:
            src_dir = os.path.join(args.src_root, in_cat, split)
            for in_label, out_label in LABELS:
                out_dirs = {}
                for third, out_root in outputs:
                    out_dir = os.path.join(out_root, out_cat, split, out_label)
                    os.makedirs(out_dir, exist_ok=True)
                    out_dirs[third] = out_dir
                pattern = re.compile(rf'^{in_label}_(.*)\.wav$')
                wavs = sorted(glob.glob(os.path.join(src_dir, '*.wav')))
                n = 0
                for wav_path in wavs:
                    m = pattern.match(os.path.basename(wav_path))
                    if not m:
                        continue
                    feats = make_feats(wav_path, wav2mel, tgramnet)
                    for third, out_dir in out_dirs.items():
                        img = build_image(feats, stats, third)
                        cv2.imwrite(os.path.join(out_dir, m.group(1) + '.png'), img)
                    n += 1
                total += n
                for third, out_dir in out_dirs.items():
                    print(f'{out_cat}/{split}/{out_label}: {n} images -> {out_dir}')
    print(f'Done. {total} clips written under {[root for _, root in outputs]}.')


if __name__ == '__main__':
    main()
