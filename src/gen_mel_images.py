"""
generate single-channel log-Mel images for MambaAD.
clips are cropped to 10 s (313 frames), matching src/gen_stgram_images.py.
"""
import argparse
import glob
import json
import os
import re

import cv2
import librosa
import numpy as np

SR = 16000
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128
FMAX = 8000
SECS = 10
N_SAMPLES = SR * SECS         # 160000 -> 313 frames

SRC_ROOT = 'data/dcase-2020'

CATS = [('data_fan/fan', 'fan'),
        ('data_pump/pump', 'pump'),
        ('data_slider/slider', 'slider'),
        ('data_valve/valve', 'valve'),
        ('data_ToyCar/ToyCar', 'ToyCar'),
        ('data_ToyConveyor/ToyConveyor', 'ToyConveyor')]
SPLITS = ['train', 'test']
# labels of (input prefix, output dir)
LABELS = [('normal', 'normal'), ('anomaly', 'abnormal')]

VALUE_SUBSAMPLE = 61
# percentiles defining the global range; values outside are clipped
PCT_LO, PCT_HI = 0.1, 99.9


def scale_fixed(X, lo, hi):
    """Map the fixed global range [lo, hi] to [0, 255], clipping outliers."""
    X = np.clip(X, lo, hi)
    return (X - lo) / (hi - lo) * 255.0


def make_logmel(wav_path):
    y = librosa.load(wav_path, sr=SR, mono=True)[0]
    y = y[:N_SAMPLES]
    S = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT,
                                       hop_length=HOP_LENGTH, fmax=FMAX)
    # absolute dB: no per-clip reference, no top_db floor tied to the clip's own max
    return librosa.power_to_db(S, ref=1.0, top_db=None)


def compute_stats(src_root, stride):
    """global [lo, hi] percentiles over train/normal clips of all categories."""
    pooled = []
    for in_cat, out_cat in CATS:
        wavs = sorted(glob.glob(os.path.join(src_root, in_cat, 'train', 'normal_*.wav')))
        wavs = wavs[::stride]
        print(f'stats: {out_cat}: sampling {len(wavs)} train/normal clips')
        for wav_path in wavs:
            pooled.append(make_logmel(wav_path).ravel()[::VALUE_SUBSAMPLE].copy())
    vals = np.concatenate(pooled)
    lo, hi = np.percentile(vals, [PCT_LO, PCT_HI])
    print(f'stats: logmel: lo={lo:.3f} hi={hi:.3f} (from {vals.size} values)')
    return dict(logmel=dict(lo=float(lo), hi=float(hi)))


def main():
    parser = argparse.ArgumentParser(description='Generate log-Mel grayscale PNGs for MambaAD.')
    parser.add_argument('--out-root', default='data/dcase-2020-spectrogram', help='Output dataset root.')
    parser.add_argument('--src-root', default=SRC_ROOT, help='Raw wav dataset root.')
    parser.add_argument('--stats-stride', type=int, default=4,
                        help='Use every Nth train/normal clip for the global scaling statistics.')
    parser.add_argument('--stats-json', default='',
                        help='Reuse a previously saved scaling.json instead of recomputing statistics.')
    args = parser.parse_args()

    if args.stats_json:
        stats = json.load(open(args.stats_json))
        print(f'Loaded scaling stats from {args.stats_json}: {stats}')
    else:
        stats = compute_stats(args.src_root, args.stats_stride)
    os.makedirs(args.out_root, exist_ok=True)
    with open(os.path.join(args.out_root, 'scaling.json'), 'w') as f:
        json.dump(stats, f, indent=4)
    lo, hi = stats['logmel']['lo'], stats['logmel']['hi']

    total = 0
    for in_cat, out_cat in CATS:
        for split in SPLITS:
            src_dir = os.path.join(args.src_root, in_cat, split)
            for in_label, out_label in LABELS:
                out_dir = os.path.join(args.out_root, out_cat, split, out_label)
                os.makedirs(out_dir, exist_ok=True)
                pattern = re.compile(rf'^{in_label}_(.*)\.wav$')
                wavs = sorted(glob.glob(os.path.join(src_dir, '*.wav')))
                n = 0
                for wav_path in wavs:
                    m = pattern.match(os.path.basename(wav_path))
                    if not m:
                        continue
                    img = np.rint(scale_fixed(make_logmel(wav_path), lo, hi)).astype(np.uint8)
                    img = np.flip(img, axis=0)   # vertical flip (matches existing pipeline)
                    cv2.imwrite(os.path.join(out_dir, m.group(1) + '.png'), img)
                    n += 1
                total += n
                print(f'{out_cat}/{split}/{out_label}: {n} images -> {out_dir}')
    print(f'Done. {total} images written under {args.out_root}.')


if __name__ == '__main__':
    main()
