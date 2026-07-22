"""
Student-feature probe — "is the readout bad, or does the decoder distort the manifold?"
=======================================================================================

Companion to `frozen_encoder_probe.py`. That probe proved the *teacher* (frozen ResNet34)
features already separate MIMII normal/anomalous better than the trained MambaAD pipeline
scores them — so the ceiling is downstream. This probe bisects "downstream" into two culprits:

  1. the SCORE READOUT  (per-pixel `1-cos(ft,fs)` residual, then sp_max/sp_mean pool), or
  2. the DECODER itself distorting the feature manifold.

It loads a *trained* MambaAD checkpoint, runs the full teacher->student forward to get the
student's reconstructed features `fs`, and scores those with the SAME global Maha/kNN scorers
the frozen probe applied to the teacher `ft`. Interpretation, per representation:

  student+Maha  ~=  teacher+Maha (frozen probe)   -> decoder PRESERVES the normal manifold;
                                                     the cosine-residual/sp_max READOUT is what
                                                     throws the signal away  -> cheap fix.
  student+Maha  <<  teacher+Maha                   -> decoder DISTORTS the manifold; the loss is
                                                     architectural (MFF/OCE, Hilbert scan, or the
                                                     distillation objective over-generalising).

For reference it also (a) re-scores the teacher features here (should reproduce the frozen probe,
a sanity check) and (b) reports the native cosine-residual sp_max/sp_mean AUROC for the same
checkpoint, so all three scoring regimes sit in one table.

Usage
-----
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/anaconda3/envs/mamba-ad/lib
    CUDA_VISIBLE_DEVICES=0 python diagnostics/student_feature_probe.py \
        -c MambaAD/configs/mambaad/mambaad_mimii_toy.py \
        --ckpt runs/MAMBAADTrainer_..._mimii_toy_20260720-020017/net_20.pth

Outputs runs/student_probe/<cfg_name>/auroc.csv (per-class + mean, every method/regime).
"""

import os
import sys
import argparse
from argparse import Namespace
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
import tabulate

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from configs import get_cfg
from data import get_loader
from model import get_model
# reuse the exact scorers/pooling the frozen probe uses, so numbers are comparable
from diagnostics.frozen_encoder_probe import (
    build_cfg, gap_vectors, fit_mahalanobis, maha_score, knn_score,
)


@torch.no_grad()
def forward_feats(net, imgs):
    """Full MambaAD forward -> (teacher feats list, student feats list)."""
    feats_t, feats_s = net(imgs)
    return feats_t, feats_s


def cos_residual_scores(feats_t, feats_s):
    """Native MambaAD readout: per-pixel 1-cos summed over taps -> (sp_max, sp_mean) per image."""
    bs = feats_t[0].shape[0]
    out_size = feats_t[0].shape[-2:]
    amap = torch.zeros(bs, *out_size, device=feats_t[0].device)
    for ft, fs in zip(feats_t, feats_s):
        a = 1 - F.cosine_similarity(ft, fs, dim=1)          # (B, h, w)
        a = F.interpolate(a.unsqueeze(1), size=out_size, mode='bilinear',
                          align_corners=True).squeeze(1)
        amap += a
    amap = amap.reshape(bs, -1)
    return amap.max(dim=1).values.cpu().numpy(), amap.mean(dim=1).cpu().numpy()


def collect(net, loader, n_taps, device, want_label):
    """One pass: per-class GAP vectors for teacher & student, plus native residual scores."""
    t_pool = defaultdict(lambda: [[] for _ in range(n_taps)]); t_cat = defaultdict(list)
    s_pool = defaultdict(lambda: [[] for _ in range(n_taps)]); s_cat = defaultdict(list)
    res_max = defaultdict(list); res_mean = defaultdict(list)
    labels = defaultdict(list)
    for bi, batch in enumerate(loader):
        imgs = batch['img'].to(device, non_blocking=True)
        cls = np.array(batch['cls_name'])
        feats_t, feats_s = forward_feats(net, imgs)
        tp, tc = gap_vectors(feats_t); sp, sc = gap_vectors(feats_s)
        tp = [p.cpu().numpy() for p in tp]; tc = tc.cpu().numpy()
        sp = [p.cpu().numpy() for p in sp]; sc = sc.cpu().numpy()
        rmax, rmean = cos_residual_scores(feats_t, feats_s)
        if want_label:
            anom = batch['anomaly']
            anom = anom.numpy() if torch.is_tensor(anom) else np.array(anom)
        for c in np.unique(cls):
            m = cls == c
            for t in range(n_taps):
                t_pool[c][t].append(tp[t][m]); s_pool[c][t].append(sp[t][m])
            t_cat[c].append(tc[m]); s_cat[c].append(sc[m])
            res_max[c].extend(rmax[m].tolist()); res_mean[c].extend(rmean[m].tolist())
            if want_label:
                labels[c].append(anom[m])
        print(f"\r  batch {bi + 1}", end='')
    print()
    return dict(t_pool=t_pool, t_cat=t_cat, s_pool=s_pool, s_cat=s_cat,
                res_max=res_max, res_mean=res_mean, labels=labels)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-c', '--cfg_path', required=True)
    ap.add_argument('--ckpt', required=True, help='trained net_<E>.pth to probe')
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--knn_k', type=int, default=5)
    ap.add_argument('--out_dir', default=None)
    args = ap.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cfg = build_cfg(args.cfg_path, args.batch, args.workers)
    cfg_name = os.path.splitext(os.path.basename(args.cfg_path))[0]
    out_dir = args.out_dir or os.path.join('runs', 'student_probe', cfg_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[cfg] {args.cfg_path}  data.root={cfg.data.root}")

    # build full teacher->student MambaAD, load trained weights
    net = get_model(cfg.model).to(device).eval()
    sd = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and 'net' in sd:
        sd = sd['net']
    missing, unexpected = net.load_state_dict(sd, strict=False)
    print(f"[ckpt] {args.ckpt}  (missing={len(missing)} unexpected={len(unexpected)})")
    n_taps = len(cfg.model_t.kwargs.get('out_indices', [1, 2, 3]))

    train_loader, test_loader = get_loader(cfg)
    print("[pass 1] normal training features (teacher + student) ...")
    tr = collect(net, train_loader, n_taps, device, want_label=False)
    print("[pass 2] test features + scoring ...")
    te = collect(net, test_loader, n_taps, device, want_label=True)

    classes = sorted(tr['t_cat'].keys())
    tap_names = {0: 'layer1', 1: 'layer2', 2: 'layer3'}

    # fit per-class Maha/kNN banks for both teacher and student concat features
    def fit(pool_key, cat_key):
        maha, banks = {}, {}
        for c in classes:
            cat = np.concatenate(tr[cat_key][c], axis=0)
            maha[c] = fit_mahalanobis(cat)
            banks[c] = cat
        return maha, banks
    t_maha, t_bank = fit('t_pool', 't_cat')
    s_maha, s_bank = fit('s_pool', 's_cat')

    # methods: teacher(sanity) + student maha/knn on concat, plus native residual readouts
    auroc = defaultdict(dict); ap_ = defaultdict(dict)
    for c in classes:
        y = np.concatenate(te['labels'][c]).astype(int)
        if y.min() == y.max():
            print(f"  [WARN] class {c}: single-label test set, skipping"); continue
        t_cat = np.concatenate(te['t_cat'][c], axis=0)
        s_cat = np.concatenate(te['s_cat'][c], axis=0)

        def rec(name, s):
            auroc[name][c] = roc_auc_score(y, s); ap_[name][c] = average_precision_score(y, s)

        mu, ic = t_maha[c]; rec('teacher_maha_concat', maha_score(t_cat, mu, ic))
        rec('teacher_knn_concat', knn_score(t_bank[c], t_cat, args.knn_k, device))
        mu, ic = s_maha[c]; rec('student_maha_concat', maha_score(s_cat, mu, ic))
        rec('student_knn_concat', knn_score(s_bank[c], s_cat, args.knn_k, device))
        rec('residual_sp_max', np.asarray(te['res_max'][c], dtype=np.float32))
        rec('residual_sp_mean', np.asarray(te['res_mean'][c], dtype=np.float32))

    methods = ['teacher_maha_concat', 'teacher_knn_concat',
               'student_maha_concat', 'student_knn_concat',
               'residual_sp_max', 'residual_sp_mean']
    header = ['method'] + classes + ['mean_AUROC', 'mean_AP']
    rows = []
    for name in methods:
        if name not in auroc:
            continue
        row = [name]; aus, aps = [], []
        for c in classes:
            v = auroc[name].get(c)
            row.append(f'{v * 100:.2f}' if v is not None else '  -  ')
            if v is not None:
                aus.append(v); aps.append(ap_[name][c])
        row += [f'{np.mean(aus) * 100:.2f}' if aus else '  -  ',
                f'{np.mean(aps) * 100:.2f}' if aps else '  -  ']
        rows.append(row)

    print("\n=== Where does the signal die? (image-level AUROC %, higher=better) ===")
    print(tabulate.tabulate(rows, headers=header, tablefmt='github'))
    csv_path = os.path.join(out_dir, 'auroc.csv')
    with open(csv_path, 'w') as f:
        f.write(','.join(header) + '\n')
        for row in rows:
            f.write(','.join(str(x).strip() for x in row) + '\n')
    print(f"\n[out] {csv_path}")

    # automatic verdict
    def mean_of(name):
        vs = list(auroc[name].values()); return float(np.mean(vs)) * 100 if vs else float('nan')
    tm, sm, rx = mean_of('teacher_maha_concat'), mean_of('student_maha_concat'), mean_of('residual_sp_max')
    print(f"\nteacher_maha={tm:.2f}  student_maha={sm:.2f}  residual_sp_max={rx:.2f}")
    if sm >= tm - 2.0:
        print("VERDICT: decoder PRESERVES the manifold (student~=teacher). The cosine-residual/"
              "sp_max READOUT is the culprit -> cheap fix (Maha/kNN readout or sp_mean).")
    elif sm <= (tm + rx) / 2:
        print("VERDICT: decoder DISTORTS the manifold (student << teacher). Loss is architectural "
              "(MFF/OCE fusion, Hilbert scan, or the distillation objective).")
    else:
        print("VERDICT: mixed — decoder loses some manifold structure AND the readout costs more.")


if __name__ == '__main__':
    main()
