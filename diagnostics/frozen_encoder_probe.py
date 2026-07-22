"""
Frozen-encoder feature probe — "is ResNet34 exonerated?"
========================================================

This diagnostic answers one question: *do the raw, frozen ResNet34 features that
the MambaAD teacher produces already separate normal from anomalous MIMII clips?*

If they do, the encoder is not the bottleneck — the ceiling is downstream
(MFF/OCE fusion, the Mamba decoder, the Hilbert scan, or the sp-score readout),
and that is where to spend effort. If they do NOT, then no scan geometry or
decoder change can recover discriminative structure the frozen encoder already
threw away, and the next moves are BN-stat adaptation, a shallower tap, or an
audio-pretrained backbone (e.g. CNN14).

How it works
------------
It bypasses the student/decoder entirely. Using *exactly* the teacher encoder
(same weights from the config's `checkpoint_path`, same taps `out_indices=[1,2,3]`
= layer1/2/3, same ImageNet-normalised 256x256 input, frozen BN in eval mode), it
extracts features on the normal-only training split, fits classic frozen-feature
anomaly scorers per machine class, and scores the test split (normal + abnormal).
It reports image-level AUROC / AP — directly comparable to MambaAD's
`mAUROC_sp_*` numbers.

Three scorers, deliberately spanning the global/local axis of the encoder critique:

  * maha   — Mahalanobis distance on globally-average-pooled features
             (Rippel et al. / PaDiM-image). Global; tests whether the *summary*
             of a clip's features shifts under anomaly.
  * knn    — mean cosine distance to k nearest normal clips on GAP features
             (DN2 / SPADE-image). Global, non-parametric.
  * patch  — PatchCore-style: per-location patch embeddings (layer2 âŠ• upsampled
             layer3, 3x3 local aggregation), min-distance to a normal patch
             memory bank, image score = max over patches. LOCAL — this is the
             one that can still see fine, localised spectral detail that global
             pooling averages away, so it is the most faithful test of "did the
             encoder keep the anomaly signal at all".

Reading the result
-------------------
Pass your trained MambaAD number via --trained_auroc to get an automatic verdict.
Rough guide (mean image-level AUROC across classes):

  frozen best >= trained          -> encoder EXONERATED. Frozen features already
                                     separate as well as / better than the trained
                                     pipeline; the problem is downstream.
  frozen best clearly < trained    -> decoder is adding real value; encoder is a
                                     plausible-but-not-sole ceiling.
  frozen best near chance (~0.5)    -> encoder features do NOT separate normal vs
                                     anomalous. Look at BN adaptation / shallower
                                     tap / audio backbone before touching the scan.

Compare `patch` vs `maha`/`knn`: if patch >> pooled, the signal is *local* and
global pooling (or an over-pooled tap) is destroying it — an argument for a
shallower tap, exactly Mismatch #2 in the encoder critique.

Usage
-----
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/anaconda3/envs/mamba-ad/lib
    CUDA_VISIBLE_DEVICES=0 python diagnostics/frozen_encoder_probe.py \
        -c MambaAD/configs/mambaad/mambaad_mimii_toy.py \
        --trained_auroc 0.62 --plot

    # other spectrogram variants use their own config + data root automatically:
    python diagnostics/frozen_encoder_probe.py -c MambaAD/configs/mambaad/mambaad_mimii_stgram.py

Outputs a printed table plus:
    runs/frozen_probe/<cfg_name>/auroc.csv        (per-class + mean, every method)
    runs/frozen_probe/<cfg_name>/scores_<cls>.png (optional, --plot)
"""

import os
import sys
import argparse
from argparse import Namespace
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

# make repo root importable regardless of CWD
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import timm
from timm.models._helpers import load_checkpoint
from sklearn.metrics import roc_auc_score, average_precision_score
import tabulate

from configs import get_cfg
from data import get_loader


# --------------------------------------------------------------------------- #
# config / data plumbing (mirror run.py's single-GPU, no-DDP setup)
# --------------------------------------------------------------------------- #
def build_cfg(cfg_path, batch, workers):
    opt = Namespace(
        cfg_path=cfg_path, mode='test', sleep=-1, memory=-1,
        dist_url='env://', logger_rank=0, opts=[],
    )
    cfg = get_cfg(opt)
    # minimal single-process runtime fields get_loader/get_dataset need
    cfg.dist = False
    cfg.world_size, cfg.rank, cfg.local_rank, cfg.master = 1, 0, 0, True
    cfg.trainer.data.batch_size_per_gpu = batch
    cfg.trainer.data.batch_size_per_gpu_test = batch
    cfg.trainer.data.num_workers_per_gpu = workers
    cfg.trainer.data.pin_memory = True
    cfg.trainer.data.drop_last = False       # keep every normal train clip
    cfg.trainer.data.persistent_workers = False
    return cfg


def build_encoder(cfg, device):
    """Rebuild the *exact* teacher encoder: timm resnet34, features_only,
    out_indices from the config, weights from the config checkpoint, frozen+eval."""
    mt = cfg.model_t
    assert mt.name == 'timm_resnet34', \
        f"probe assumes a timm_resnet34 teacher; config has {mt.name}"
    kw = dict(mt.kwargs)
    out_indices = kw.get('out_indices', [1, 2, 3])
    ckpt = kw.get('checkpoint_path', '')
    model = timm.create_model(
        'resnet34', pretrained=False, features_only=True, out_indices=out_indices,
    )
    if ckpt and os.path.isfile(ckpt):
        load_checkpoint(model, ckpt, strict=kw.get('strict', False))
        print(f"[encoder] loaded teacher weights: {ckpt}")
    else:
        print(f"[encoder] WARNING: checkpoint '{ckpt}' not found; using random init "
              f"(results will be meaningless).")
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    n_taps = len(out_indices)
    return model, n_taps


# --------------------------------------------------------------------------- #
# feature extraction
# --------------------------------------------------------------------------- #
@torch.no_grad()
def extract(model, imgs):
    """Return list of feature maps [layer1, layer2, layer3] for a batch."""
    return model(imgs)


def gap_vectors(feats):
    """Global-average-pool each tap -> list of (B, C) tensors, plus their concat."""
    pooled = [f.mean(dim=(2, 3)) for f in feats]
    concat = torch.cat(pooled, dim=1)
    return pooled, concat


def patch_embeddings(feats, layer_a=1, layer_b=2, local_k=3):
    """PatchCore-style patch tensor: locally-aggregated layer_a âŠ• upsampled layer_b,
    returned as (B, N, D) with N = spatial positions of layer_a."""
    fa, fb = feats[layer_a], feats[layer_b]
    pad = local_k // 2
    fa = F.avg_pool2d(fa, kernel_size=local_k, stride=1, padding=pad)
    fb = F.avg_pool2d(fb, kernel_size=local_k, stride=1, padding=pad)
    fb = F.interpolate(fb, size=fa.shape[-2:], mode='bilinear', align_corners=False)
    x = torch.cat([fa, fb], dim=1)                       # (B, D, H, W)
    B, D, H, W = x.shape
    return x.permute(0, 2, 3, 1).reshape(B, H * W, D)    # (B, N, D)


# --------------------------------------------------------------------------- #
# scorers
# --------------------------------------------------------------------------- #
def fit_mahalanobis(train_vecs):
    """train_vecs: (N, D) numpy. Returns (mean, inv_cov) with shrinkage."""
    mu = train_vecs.mean(axis=0)
    xc = train_vecs - mu
    cov = (xc.T @ xc) / max(len(train_vecs) - 1, 1)
    d = cov.shape[0]
    # Ledoit-Wolf-style shrinkage toward a scaled identity for a stable inverse
    tr = np.trace(cov) / d
    alpha = 0.01
    cov = (1 - alpha) * cov + alpha * tr * np.eye(d) + 1e-6 * np.eye(d)
    inv_cov = np.linalg.pinv(cov).astype(np.float32)
    return mu.astype(np.float32), inv_cov


def maha_score(vecs, mu, inv_cov):
    xc = vecs - mu
    return np.sqrt(np.einsum('ni,ij,nj->n', xc, inv_cov, xc, optimize=True))


def knn_score(train_vecs, test_vecs, k, device):
    """Mean cosine distance to k nearest normal training vectors. Higher = anomalous."""
    tr = F.normalize(torch.from_numpy(train_vecs).to(device), dim=1)
    te = F.normalize(torch.from_numpy(test_vecs).to(device), dim=1)
    scores = np.empty(len(test_vecs), dtype=np.float32)
    kk = min(k, tr.shape[0])
    for i in range(0, te.shape[0], 256):
        chunk = te[i:i + 256]
        sim = chunk @ tr.T                       # cosine similarity
        dist = 1.0 - sim
        topk = dist.topk(kk, dim=1, largest=False).values
        scores[i:i + chunk.shape[0]] = topk.mean(dim=1).cpu().numpy()
    return scores


def patchcore_image_scores(bank, test_patches_per_img, device, qchunk=4096):
    """bank: (M, D) tensor on device. test_patches_per_img: list of (N, D) tensors.
    Returns per-image score = max over patches of min-distance-to-bank."""
    scores = np.empty(len(test_patches_per_img), dtype=np.float32)
    for i, q in enumerate(test_patches_per_img):
        q = q.to(device)
        mins = torch.empty(q.shape[0], device=device)
        for s in range(0, q.shape[0], qchunk):
            qc = q[s:s + qchunk]
            d = torch.cdist(qc, bank)            # (chunk, M)
            mins[s:s + qc.shape[0]] = d.min(dim=1).values
        scores[i] = mins.max().item()
    return scores


# --------------------------------------------------------------------------- #
# main probe
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-c', '--cfg_path', required=True,
                    help='ADer config (e.g. MambaAD/configs/mambaad/mambaad_mimii_toy.py)')
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--knn_k', type=int, default=5)
    ap.add_argument('--patches_per_image', type=int, default=128,
                    help='normal patches sampled per train image for the memory bank')
    ap.add_argument('--bank_size', type=int, default=40000,
                    help='max patches in a per-class PatchCore memory bank')
    ap.add_argument('--patch_layers', type=int, nargs=2, default=[1, 2],
                    help='which taps to use for patch embeddings (indices into the '
                         '3 taps; default 1 2 = layer2+layer3, PatchCore convention)')
    ap.add_argument('--skip_patch', action='store_true',
                    help='skip the (heavier) PatchCore local scorer')
    ap.add_argument('--trained_auroc', type=float, default=None,
                    help='your trained MambaAD mean image AUROC (0-1) for an auto verdict')
    ap.add_argument('--plot', action='store_true',
                    help='save per-class normal-vs-abnormal score histograms')
    ap.add_argument('--out_dir', default=None)
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cfg = build_cfg(args.cfg_path, args.batch, args.workers)
    cfg_name = os.path.splitext(os.path.basename(args.cfg_path))[0]
    out_dir = args.out_dir or os.path.join('runs', 'frozen_probe', cfg_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[cfg] {args.cfg_path}  data.root={cfg.data.root}")

    model, n_taps = build_encoder(cfg, device)
    la, lb = args.patch_layers
    assert 0 <= la < n_taps and 0 <= lb < n_taps, "patch_layers out of range of taps"

    train_loader, test_loader = get_loader(cfg)

    # ----- pass 1: normal-only train split -> per-class pooled vecs + patch bank -----
    print("[pass 1] extracting normal training features ...")
    tr_pool = defaultdict(lambda: [[] for _ in range(n_taps)])   # cls -> per-tap list of (B,C)
    tr_pool_cat = defaultdict(list)                              # cls -> list of (B, sumC)
    tr_bank = defaultdict(list) if not args.skip_patch else None  # cls -> list of (m, D)
    for bi, batch in enumerate(train_loader):
        imgs = batch['img'].to(device, non_blocking=True)
        cls = np.array(batch['cls_name'])
        feats = extract(model, imgs)
        pooled, concat = gap_vectors(feats)
        pooled = [p.cpu().numpy() for p in pooled]
        concat = concat.cpu().numpy()
        if not args.skip_patch:
            patches = patch_embeddings(feats, la, lb)            # (B, N, D)
        for c in np.unique(cls):
            m = cls == c
            for t in range(n_taps):
                tr_pool[c][t].append(pooled[t][m])
            tr_pool_cat[c].append(concat[m])
            if not args.skip_patch:
                pc = patches[torch.from_numpy(m)]                # (b, N, D)
                b, N, D = pc.shape
                take = min(args.patches_per_image, N)
                for j in range(b):
                    idx = torch.randperm(N)[:take]
                    tr_bank[c].append(pc[j, idx].cpu())
        print(f"\r  train batch {bi + 1}", end='')
    print()

    classes = sorted(tr_pool_cat.keys())
    # consolidate + fit per class
    maha_params, knn_banks, patch_banks = {}, {}, {}
    for c in classes:
        per_tap = [np.concatenate(tr_pool[c][t], axis=0) for t in range(n_taps)]
        cat = np.concatenate(tr_pool_cat[c], axis=0)
        maha_params[c] = {'concat': fit_mahalanobis(cat)}
        for t in range(n_taps):
            maha_params[c][t] = fit_mahalanobis(per_tap[t])
        knn_banks[c] = {'concat': cat, **{t: per_tap[t] for t in range(n_taps)}}
        if not args.skip_patch:
            bank = torch.cat(tr_bank[c], dim=0)
            if bank.shape[0] > args.bank_size:
                sel = torch.randperm(bank.shape[0])[:args.bank_size]
                bank = bank[sel]
            patch_banks[c] = bank.to(device)
            print(f"  [{c}] normal clips={len(cat)}  patch bank={bank.shape[0]}x{bank.shape[1]}")
        else:
            print(f"  [{c}] normal clips={len(cat)}")

    # ----- pass 2: test split -> per-class test vecs, labels, patch scores -----
    print("[pass 2] extracting + scoring test features ...")
    te_pool = defaultdict(lambda: [[] for _ in range(n_taps)])
    te_pool_cat = defaultdict(list)
    te_label = defaultdict(list)
    patch_scores = defaultdict(list)                             # cls -> list of float
    for bi, batch in enumerate(test_loader):
        imgs = batch['img'].to(device, non_blocking=True)
        cls = np.array(batch['cls_name'])
        anom = batch['anomaly'].numpy() if torch.is_tensor(batch['anomaly']) else np.array(batch['anomaly'])
        feats = extract(model, imgs)
        pooled, concat = gap_vectors(feats)
        pooled = [p.cpu().numpy() for p in pooled]
        concat = concat.cpu().numpy()
        if not args.skip_patch:
            patches = patch_embeddings(feats, la, lb)
        for c in np.unique(cls):
            m = cls == c
            for t in range(n_taps):
                te_pool[c][t].append(pooled[t][m])
            te_pool_cat[c].append(concat[m])
            te_label[c].append(anom[m])
            if not args.skip_patch and c in patch_banks:
                pc = patches[torch.from_numpy(m)]
                per_img = [pc[j] for j in range(pc.shape[0])]
                s = patchcore_image_scores(patch_banks[c], per_img, device)
                patch_scores[c].extend(s.tolist())
        print(f"\r  test batch {bi + 1}", end='')
    print()

    # ----- score + AUROC per class -----
    tap_names = {0: 'layer1', 1: 'layer2', 2: 'layer3'}
    methods = []
    for t in range(n_taps):
        methods += [f'maha_{tap_names.get(t, t)}', f'knn_{tap_names.get(t, t)}']
    methods += ['maha_concat', 'knn_concat']
    if not args.skip_patch:
        methods.append(f'patch_{tap_names.get(la, la)}+{tap_names.get(lb, lb)}')

    auroc = defaultdict(dict)   # method -> cls -> auroc
    ap_ = defaultdict(dict)
    all_scores = defaultdict(dict)  # for plotting: method -> cls -> (scores, labels)
    for c in classes:
        y = np.concatenate(te_label[c]).astype(int)
        cat = np.concatenate(te_pool_cat[c], axis=0)
        per_tap = [np.concatenate(te_pool[c][t], axis=0) for t in range(n_taps)]
        if y.min() == y.max():
            print(f"  [WARN] class {c}: test labels all == {y[0]}, AUROC undefined; skipping")
            continue

        def record(name, s):
            auroc[name][c] = roc_auc_score(y, s)
            ap_[name][c] = average_precision_score(y, s)
            all_scores[name][c] = (s, y)

        for t in range(n_taps):
            mu, ic = maha_params[c][t]
            record(f'maha_{tap_names.get(t, t)}', maha_score(per_tap[t], mu, ic))
            record(f'knn_{tap_names.get(t, t)}',
                   knn_score(knn_banks[c][t], per_tap[t], args.knn_k, device))
        mu, ic = maha_params[c]['concat']
        record('maha_concat', maha_score(cat, mu, ic))
        record('knn_concat', knn_score(knn_banks[c]['concat'], cat, args.knn_k, device))
        if not args.skip_patch and c in patch_banks:
            record(f'patch_{tap_names.get(la, la)}+{tap_names.get(lb, lb)}',
                   np.asarray(patch_scores[c], dtype=np.float32))

    # ----- report -----
    header = ['method'] + classes + ['mean_AUROC', 'mean_AP']
    rows = []
    method_mean = {}
    for name in methods:
        if name not in auroc:
            continue
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

    print("\n=== Frozen ResNet34 feature separability (image-level AUROC %, higher=better) ===")
    print(tabulate.tabulate(rows, headers=header, tablefmt='github'))

    # CSV
    csv_path = os.path.join(out_dir, 'auroc.csv')
    with open(csv_path, 'w') as f:
        f.write(','.join(header) + '\n')
        for row in rows:
            f.write(','.join(str(x).strip() for x in row) + '\n')
    print(f"\n[out] {csv_path}")

    # ----- verdict -----
    if method_mean:
        best_name = max(method_mean, key=method_mean.get)
        best = method_mean[best_name]
        print(f"\nBest frozen-feature scorer: {best_name} = {best * 100:.2f}% mean AUROC")
        if args.trained_auroc is not None:
            tr = args.trained_auroc
            print(f"Trained MambaAD (you supplied):     {tr * 100:.2f}% mean AUROC")
            if best >= tr - 0.01:
                print("VERDICT: encoder EXONERATED — frozen features separate as well as / "
                      "better than the trained pipeline. The ceiling is DOWNSTREAM "
                      "(MFF/OCE, Mamba decoder, Hilbert scan, or sp-score readout).")
            elif best <= 0.55:
                print("VERDICT: frozen features are ~chance. The ENCODER does not carry the "
                      "anomaly signal — try BN-stat adaptation, a shallower tap, or an "
                      "audio-pretrained backbone before touching the scan/decoder.")
            else:
                print("VERDICT: mixed — the decoder adds real value (trained > frozen), but "
                      "frozen features are well above chance, so the encoder is a plausible "
                      "but not sole ceiling.")
        # local-vs-global hint
        patch_key = next((m for m in method_mean if m.startswith('patch_')), None)
        pooled_best = max((method_mean[m] for m in method_mean if not m.startswith('patch_')),
                          default=None)
        if patch_key is not None and pooled_best is not None:
            if method_mean[patch_key] - pooled_best > 0.05:
                print(f"NOTE: local patch scorer ({method_mean[patch_key]*100:.2f}%) >> best pooled "
                      f"({pooled_best*100:.2f}%): the anomaly signal is LOCAL — global pooling / an "
                      f"over-pooled deep tap is discarding it (encoder-critique Mismatch #2).")

    # ----- optional plots -----
    if args.plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"[plot] matplotlib unavailable ({e}); skipping plots")
            return
        best_name = max(method_mean, key=method_mean.get)
        for c in classes:
            if c not in all_scores.get(best_name, {}):
                continue
            s, y = all_scores[best_name][c]
            fig, axp = plt.subplots(figsize=(5, 3))
            axp.hist(s[y == 0], bins=40, alpha=0.6, label='normal', density=True)
            axp.hist(s[y == 1], bins=40, alpha=0.6, label='abnormal', density=True)
            axp.set_title(f'{c}  {best_name}  AUROC={auroc[best_name][c]*100:.1f}%')
            axp.set_xlabel('anomaly score'); axp.legend()
            fig.tight_layout()
            p = os.path.join(out_dir, f'scores_{c}.png')
            fig.savefig(p, dpi=120); plt.close(fig)
        print(f"[out] score histograms -> {out_dir}/scores_*.png")


if __name__ == '__main__':
    main()
