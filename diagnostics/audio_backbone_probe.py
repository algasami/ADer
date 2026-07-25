"""
Audio-pretrained backbone probe — "is the ImageNet ResNet34 the real ceiling?"
==============================================================================

Companion to `frozen_encoder_probe.py`. That probe showed the frozen ImageNet
ResNet34 teacher already sets the MambaAD-track ceiling (~71.8 AUROC via
Mahalanobis on GAP features) — decoder training, scan geometry, schedule and the
cos-residual readout are all downstream and either flat or a fixed +6. The one
lever left untested is the *feature extractor itself*: the ~19-pt gap to the
audio-native STgram-MFN baseline (90.7) may simply be that ImageNet features are
wrong for spectrograms.

This probe tests exactly that. It swaps the ImageNet CNN for an
**audio-pretrained backbone used natively on the raw MIMII wavs** (NOT a
repurposed spectrogram PNG — that would reintroduce the very domain mismatch we
are trying to eliminate), then fits the *same* frozen-feature scorers
(Mahalanobis / kNN, imported verbatim from `frozen_encoder_probe.py`) on the
*same* train-normal / test split. So the only variable vs the 71.8 baseline is
"ImageNet CNN features" -> "AudioSet-pretrained features".

Split provenance
----------------
Reads the raw wavs directly from `--data_root` (default `data/dcase-2020`), whose
layout mirrors the image `meta.json` one-to-one:
    data/dcase-2020/data_<CLS>/<CLS>/train/{normal,anomaly}_id_XX_*.wav   (train = normal only)
    data/dcase-2020/data_<CLS>/<CLS>/test/ {normal,anomaly}_id_XX_*.wav   (test  = normal+anomaly)
Label comes from the filename prefix (anomaly_ -> 1, normal_ -> 0). Counts match
the spectrogram meta.json exactly (verified: slider train 2804 / test 400+890),
so AUROCs are directly comparable to the frozen ResNet34 numbers.

Backbones
---------
  ast     — Audio Spectrogram Transformer (MIT/ast-finetuned-audioset-10-10-0.4593),
            AudioSet-pretrained ViT on 128-mel / 16 kHz. Native front-end via
            ASTFeatureExtractor. Two embeddings ("taps"): the pooler output and
            the mean over patch tokens.
  cnn14   — PANNs CNN14 (AudioSet-pretrained CNN; the DCASE workhorse and the
            most controlled audio-CNN-vs-image-CNN comparison to ResNet34).
            Requires `panns_inference` (auto-downloads the checkpoint to
            ~/panns_data). Embedding: the 2048-d clipwise embedding.

Usage
-----
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/anaconda3/envs/mamba-ad/lib
    CUDA_VISIBLE_DEVICES=0 python diagnostics/audio_backbone_probe.py --backbone ast
    # quick smoke test on two classes, subsampled train bank:
    python diagnostics/audio_backbone_probe.py --backbone ast \
        --classes slider valve --limit_train 200

Outputs
-------
    runs/audio_probe/<backbone>/auroc.csv   (per-class + mean, every method;
                                             same column layout as frozen_probe)
"""

import os
import sys
import glob
import time
import argparse
from collections import defaultdict

import numpy as np
import torch

# make repo root importable regardless of CWD
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from sklearn.metrics import roc_auc_score, average_precision_score
import tabulate

# reuse the EXACT scorer math from the ResNet34 probe so numbers are comparable
from diagnostics.frozen_encoder_probe import fit_mahalanobis, maha_score, knn_score

SR = 16000
CLIP_LEN = SR * 10  # 10 s, matches the spectrogram generators' force-crop
MIMII_CLASSES = ['fan', 'pump', 'slider', 'valve', 'ToyCar', 'ToyConveyor']

# frozen ImageNet ResNet34 reference (maha_concat, best pooled scorer), for the
# side-by-side verdict print. From runs/frozen_probe/mambaad_mimii_toy/auroc.csv.
RESNET34_REF = {
    'ToyCar': 75.68, 'ToyConveyor': 63.79, 'fan': 58.44, 'pump': 72.06,
    'slider': 90.78, 'valve': 70.02, 'mean': 71.80,
}


# --------------------------------------------------------------------------- #
# wav loading
# --------------------------------------------------------------------------- #
def load_wav(path):
    """Load a wav as float32 mono @ SR, force length to CLIP_LEN (pad/truncate)."""
    import soundfile as sf
    w, sr = sf.read(path, dtype='float32')
    if w.ndim > 1:
        w = w.mean(axis=1)
    if sr != SR:
        import librosa
        w = librosa.resample(w, orig_sr=sr, target_sr=SR)
    if len(w) >= CLIP_LEN:
        w = w[:CLIP_LEN]
    else:
        w = np.pad(w, (0, CLIP_LEN - len(w)))
    return w


def list_split(data_root, cls, split):
    """Return [(wav_path, label)] for a class/split; label 1 iff filename starts 'anomaly'."""
    d = os.path.join(data_root, f'data_{cls}', cls, split)
    items = []
    for p in sorted(glob.glob(os.path.join(d, '*.wav'))):
        base = os.path.basename(p)
        items.append((p, 1 if base.startswith('anomaly') else 0))
    return items


# --------------------------------------------------------------------------- #
# backbones: each returns dict{emb_name: (N, D) float32} for a list of wav arrays
# --------------------------------------------------------------------------- #
class ASTBackbone:
    name = 'ast'

    def __init__(self, device):
        from transformers import ASTFeatureExtractor, ASTModel
        ckpt = 'MIT/ast-finetuned-audioset-10-10-0.4593'
        self.fe = ASTFeatureExtractor.from_pretrained(ckpt)
        self.model = ASTModel.from_pretrained(ckpt).eval().to(device)
        for p in self.model.parameters():
            p.requires_grad = False
        self.device = device

    @torch.no_grad()
    def embed(self, wavs):
        inp = self.fe(list(wavs), sampling_rate=SR, return_tensors='pt')
        out = self.model(inp['input_values'].to(self.device))
        lhs = out.last_hidden_state           # (B, 2+P, 768): [CLS, dist, patches...]
        pooler = out.pooler_output            # (B, 768)
        meanpatch = lhs[:, 2:, :].mean(dim=1)  # (B, 768)
        return {
            'ast_pooler': pooler.float().cpu().numpy(),
            'ast_meanpatch': meanpatch.float().cpu().numpy(),
        }


class CNN14Backbone:
    name = 'cnn14'
    # AudioSet PANNs CNN14 @ 16 kHz (matches MIMII sr; the 32 kHz default checkpoint
    # would be a sample-rate mismatch). Hyperparams per the PANNs repo 16k config.
    CKPT_URL = 'https://zenodo.org/record/3987831/files/Cnn14_16k_mAP%3D0.438.pth'
    CKPT_PATH = os.path.expanduser('~/panns_data/Cnn14_16k_mAP=0.438.pth')

    def __init__(self, device):
        from panns_inference.models import Cnn14
        if not os.path.isfile(self.CKPT_PATH):
            os.makedirs(os.path.dirname(self.CKPT_PATH), exist_ok=True)
            import urllib.request
            print(f"[cnn14] downloading 16k checkpoint -> {self.CKPT_PATH}", flush=True)
            tmp = self.CKPT_PATH + '.tmp'
            urllib.request.urlretrieve(self.CKPT_URL, tmp)  # download to tmp
            os.replace(tmp, self.CKPT_PATH)                 # atomic: no partial at final path
        self.model = Cnn14(sample_rate=SR, window_size=512, hop_size=160,
                           mel_bins=64, fmin=50, fmax=8000, classes_num=527)
        state = torch.load(self.CKPT_PATH, map_location='cpu', weights_only=False)
        self.model.load_state_dict(state['model'])
        self.model = self.model.eval().to(device)
        for p in self.model.parameters():
            p.requires_grad = False
        self.device = device

    @torch.no_grad()
    def embed(self, wavs):
        x = torch.from_numpy(np.stack(wavs)).float().to(self.device)  # (B, T)
        out = self.model(x, None)
        emb = out['embedding']                # (B, 2048)
        return {'cnn14_emb': emb.float().cpu().numpy()}


def build_backbone(name, device):
    if name == 'ast':
        return ASTBackbone(device)
    if name == 'cnn14':
        return CNN14Backbone(device)
    raise ValueError(f'unknown backbone {name!r}')


# --------------------------------------------------------------------------- #
# extraction over a list of (path,label)
# --------------------------------------------------------------------------- #
def extract_split(backbone, items, batch, tag=''):
    """items: [(path,label)]. Returns (dict{emb_name:(N,D)}, labels (N,))."""
    embs = defaultdict(list)
    labels = []
    n = len(items)
    for i in range(0, n, batch):
        chunk = items[i:i + batch]
        wavs = [load_wav(p) for p, _ in chunk]
        out = backbone.embed(wavs)
        for k, v in out.items():
            embs[k].append(v)
        labels.extend(l for _, l in chunk)
        print(f"\r  [{tag}] {min(i + batch, n)}/{n}", end='', flush=True)
    print()
    embs = {k: np.concatenate(v, axis=0) for k, v in embs.items()}
    return embs, np.asarray(labels, dtype=int)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--backbone', default='ast', choices=['ast', 'cnn14'])
    ap.add_argument('--data_root', default='data/dcase-2020')
    ap.add_argument('--classes', nargs='+', default=MIMII_CLASSES)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--knn_k', type=int, default=5)
    ap.add_argument('--limit_train', type=int, default=None,
                    help='subsample this many train-normal clips per class (speed)')
    ap.add_argument('--out_dir', default=None)
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = args.out_dir or os.path.join('runs', 'audio_probe', args.backbone)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[backbone] loading {args.backbone} ...")
    t0 = time.time()
    backbone = build_backbone(args.backbone, device)
    print(f"[backbone] ready in {time.time() - t0:.1f}s  device={device}")

    emb_names = None
    auroc = defaultdict(dict)   # method -> cls -> auroc
    ap_ = defaultdict(dict)

    for c in args.classes:
        tr_items = list_split(args.data_root, c, 'train')
        te_items = list_split(args.data_root, c, 'test')
        tr_items = [it for it in tr_items if it[1] == 0]  # train = normal only (UAD)
        if args.limit_train and len(tr_items) > args.limit_train:
            idx = np.random.permutation(len(tr_items))[:args.limit_train]
            tr_items = [tr_items[i] for i in idx]
        print(f"\n[{c}] train_normal={len(tr_items)}  "
              f"test={len(te_items)} (anom={sum(l for _, l in te_items)})")

        tr_emb, _ = extract_split(backbone, tr_items, args.batch, tag=f'{c} train')
        te_emb, y = extract_split(backbone, te_items, args.batch, tag=f'{c} test')
        if y.min() == y.max():
            print(f"  [WARN] {c}: test labels degenerate; skipping")
            continue
        emb_names = list(tr_emb.keys())

        for en in emb_names:
            tr_v, te_v = tr_emb[en], te_emb[en]
            mu, ic = fit_mahalanobis(tr_v)
            s_maha = maha_score(te_v, mu, ic)
            s_knn = knn_score(tr_v, te_v, args.knn_k, device)
            auroc[f'{en}_maha'][c] = roc_auc_score(y, s_maha)
            ap_[f'{en}_maha'][c] = average_precision_score(y, s_maha)
            auroc[f'{en}_knn'][c] = roc_auc_score(y, s_knn)
            ap_[f'{en}_knn'][c] = average_precision_score(y, s_knn)
            print(f"  {en:16s} maha={auroc[f'{en}_maha'][c]*100:5.2f}  "
                  f"knn={auroc[f'{en}_knn'][c]*100:5.2f}")

    # ----- report -----
    classes = [c for c in args.classes if any(c in auroc[m] for m in auroc)]
    methods = []
    if emb_names:
        for en in emb_names:
            methods += [f'{en}_maha', f'{en}_knn']
    header = ['method'] + classes + ['mean_AUROC', 'mean_AP']
    rows = []
    method_mean = {}
    for name in methods:
        row = [name]
        aus, aps = [], []
        for c in classes:
            v = auroc[name].get(c)
            row.append(f'{v * 100:.2f}' if v is not None else '  -  ')
            if v is not None:
                aus.append(v)
                aps.append(ap_[name][c])
        mean_au = float(np.mean(aus)) if aus else float('nan')
        method_mean[name] = mean_au
        row += [f'{mean_au * 100:.2f}', f'{np.mean(aps) * 100:.2f}' if aps else '  -  ']
        rows.append(row)

    print(f"\n=== {args.backbone} audio-pretrained feature separability "
          f"(image-level AUROC %, higher=better) ===")
    print(tabulate.tabulate(rows, headers=header, tablefmt='github'))

    # side-by-side vs frozen ImageNet ResNet34 (maha_concat)
    if method_mean:
        best_name = max(method_mean, key=method_mean.get)
        best = method_mean[best_name]
        print(f"\nBest {args.backbone} scorer: {best_name} = {best*100:.2f}% mean AUROC")
        print(f"Frozen ImageNet ResNet34 (maha_concat): {RESNET34_REF['mean']:.2f}% mean AUROC")
        delta = best * 100 - RESNET34_REF['mean']
        print(f"Delta vs ResNet34: {delta:+.2f} pts")
        print("Per-class (best {} scorer vs ResNet34):".format(args.backbone))
        for c in classes:
            v = auroc[best_name].get(c)
            if v is None:
                continue
            r = RESNET34_REF.get(c)
            print(f"  {c:12s} {args.backbone}={v*100:5.2f}  resnet34={r:5.2f}  "
                  f"delta={v*100 - r:+5.2f}")
        if delta > 3:
            print("\nVERDICT: audio-pretrained features clearly beat ImageNet ResNet34 -> the "
                  "backbone WAS a real ceiling; this is the lever.")
        elif delta < -3:
            print("\nVERDICT: audio-pretrained features are WORSE -> the ImageNet CNN was not the "
                  "problem; the ceiling is the global-pooling / UAD framing, not the backbone.")
        else:
            print("\nVERDICT: roughly a wash -> swapping the backbone alone does not move the "
                  "ceiling; the bottleneck is elsewhere (pooling / task framing).")

    # CSV (same layout as frozen_probe/auroc.csv)
    csv_path = os.path.join(out_dir, 'auroc.csv')
    with open(csv_path, 'w') as f:
        f.write(','.join(header) + '\n')
        for row in rows:
            f.write(','.join(str(x).strip() for x in row) + '\n')
    print(f"\n[out] {csv_path}")


if __name__ == '__main__':
    main()
