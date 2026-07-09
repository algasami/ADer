"""
Generate STgram images for MambaAD
Has a few params down there
"""
import argparse
import glob
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


def scale_minmax(X, smin=0.0, smax=1.0):
    """Reproduced verbatim from src/notebooks/MIMII_Toy_spectrogram_converter.ipynb."""
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (smax - smin) + smin
    return X_scaled


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
def make_image(wav_path, wav2mel, tgramnet, third):
    y = librosa.load(wav_path, sr=SR, mono=True)[0]
    y = crop(y)
    x = torch.from_numpy(y).float()

    sgram = wav2mel(x).cpu().numpy()                              # (128, 313)
    tg = tgramnet(x.view(1, 1, -1).cuda())[0].cpu().numpy()       # (128, 313)
    ch2 = sgram if third == 'sgram' else librosa.feature.delta(sgram)

    img = np.stack([scale_minmax(sgram, 0, 255),
                    scale_minmax(tg, 0, 255),
                    scale_minmax(ch2, 0, 255)], axis=-1).astype(np.uint8)
    img = np.flip(img, axis=0)                                   # vertical flip (matches existing pipeline)
    return img


def main():
    parser = argparse.ArgumentParser(description='Generate STgram 3-channel PNGs for MambaAD.')
    parser.add_argument('--third', choices=['sgram', 'delta'], default='sgram',
                        help='Third channel: duplicate Sgram (default) or 1st temporal derivative of Sgram.')
    parser.add_argument('--out-root', default='data/dcase-2020-stgram',
                        help='Output dataset root (e.g. data/dcase-2020-stgram or data/dcase-2020-stgram-delta).')
    parser.add_argument('--ckpt', default=DEFAULT_CKPT, help='STgram-MFN checkpoint (.pth.tar) with tgramnet.* keys.')
    parser.add_argument('--src-root', default=SRC_ROOT, help='Raw wav dataset root.')
    args = parser.parse_args()

    wav2mel = Wave2Mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS, win_length=WIN_LENGTH,
                       hop_length=HOP_LENGTH, power=POWER)
    tgramnet = load_tgramnet(args.ckpt)

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
                    img = make_image(wav_path, wav2mel, tgramnet, args.third)
                    cv2.imwrite(os.path.join(out_dir, m.group(1) + '.png'), img)
                    n += 1
                total += n
                print(f'{out_cat}/{split}/{out_label}: {n} images -> {out_dir}')
    print(f'Done. {total} images written under {args.out_root} (third={args.third}).')


if __name__ == '__main__':
    main()
