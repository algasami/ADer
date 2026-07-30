"""
Joint distill + classify — Rung D of the "labels close the gap?" ladder
=======================================================================

The faithful "MambaAD + labels" model. Rung C trained the Mamba student to classify sections
*instead of* reconstructing teacher features. Rung D does BOTH at once:

    loss = CosLoss(feats_t, feats_s)            # native MambaAD reconstruction/distillation
         + lambda_cls * CE(ArcFace(GAP(feats_s)), section)   # the section-classification head

so the student learns a feature space that simultaneously reconstructs the frozen teacher
(the generative UAD objective) and discriminates the 23 machine sections (the self-supervised
labels). At test time three score families are available and can be FUSED:

  * recon_spmax / recon_spmean  — the NATIVE MambaAD readout: per-pixel 1-cos(ft,fs) anomaly
    map (Evaluator.cal_anomaly_map, CosResidualScorer defaults), max / mean pooled. This is the
    project's original ~72-AUROC score — Rung D asks whether ADDING the label objective during
    training improves it.
  * neg_cos / logit_nll / maha_embed — the classification readouts (as in Rung C).
  * fusion_*  — z-normalised recon_spmean + z-normalised classification score (per class). The
    actual "MambaAD + labels" anomaly score.

Read against: native MambaAD recon-only (~72, no labels), Rung C (84.4, labels-only, no recon),
Rung B (85.9, fine-tuned encoder), STgram-MFN (90.75). Teacher frozen; student+head from scratch
(single LR); non-finite guard on (MambaAD lr-5e-3 divergence history). Per-epoch eval logs the
whole curve (peaks-early rule).

Usage
-----
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/anaconda3/envs/mamba-ad/lib
    CUDA_VISIBLE_DEVICES=0 python diagnostics/section_joint_rungD.py \
        -c MambaAD/configs/mambaad/mimii/e50/log-Mel.py \
        --epochs 50 --sub 2 --lambda_cls 1.0 --eval_maha

Outputs (runs/section_rungD/<cfg_name>/): metric_curve.csv, best_summary.csv, train_log.csv.
"""

import os
import sys
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import tabulate
from sklearn.metrics import roc_auc_score

from data import get_loader
from loss.base_loss import CosLoss
from util.metric import Evaluator
from diagnostics.frozen_encoder_probe import build_cfg, fit_mahalanobis, maha_score
from diagnostics.section_classifier_probe import parse_section
from diagnostics.section_finetune_rungB import CLASSES
from diagnostics.section_mamba_rungC import MambaSectionNet


def emb_from_featss(model, feats_s):
    """Pool the student's multi-scale features through the head to the embedding."""
    gap = torch.cat([f.mean(dim=(2, 3)) for f in feats_s], dim=1)
    return model.emb_bn(model.proj(model.in_bn(gap)))


def _z(v):
    v = np.asarray(v, dtype=np.float64)
    return (v - v.mean()) / (v.std() + 1e-6)


@torch.no_grad()
def collect_D(model, loader, device, need_anom, recon_size, gauss):
    """One forward per batch capturing: embedding, logits, recon sp_max/sp_mean, meta."""
    model.eval()
    E, L, SPMAX, SPMEAN, cls, sec, anom = [], [], [], [], [], [], []
    for batch in loader:
        imgs = batch['img'].to(device, non_blocking=True)
        feats_t, feats_s = model.net(imgs)
        emb = emb_from_featss(model, feats_s)
        E.append(emb.cpu().numpy())
        L.append(model.logits(emb).cpu().numpy())
        amap, _ = Evaluator.cal_anomaly_map(feats_t, feats_s, [recon_size, recon_size],
                                            use_cos=True, amap_mode='add', gaussian_sigma=gauss)
        flat = amap.reshape(amap.shape[0], -1)
        SPMAX.append(flat.max(axis=1))
        SPMEAN.append(flat.mean(axis=1))
        c = np.array(batch['cls_name'])
        cls.append(c)
        sec.append(np.array([parse_section(cc, p) for cc, p in zip(c, batch['img_path'])]))
        if need_anom:
            a = batch['anomaly']
            anom.append(a.numpy() if torch.is_tensor(a) else np.array(a))
    d = dict(E=np.concatenate(E).astype(np.float32), L=np.concatenate(L).astype(np.float32),
             spmax=np.concatenate(SPMAX), spmean=np.concatenate(SPMEAN),
             cls=np.concatenate(cls), sec=np.concatenate(sec))
    d['anom'] = np.concatenate(anom).astype(int) if need_anom else None
    return d


def score_D(test, sec2idx, s_scale, train=None, eval_maha=False):
    """All readouts + fusions -> {readout: {class: auroc, 'mean': mean}} (higher=anomalous)."""
    L, E = test['L'], test['E']
    logp = L - np.log(np.exp(L - L.max(1, keepdims=True)).sum(1, keepdims=True)) - L.max(1, keepdims=True)
    assigned = np.array([sec2idx.get(s, -1) for s in test['sec']])
    maha_banks = None
    if eval_maha and train is not None:
        maha_banks = {c: fit_mahalanobis(train['E'][train['cls'] == c])
                      for c in CLASSES if (train['cls'] == c).any()}
    out = defaultdict(dict)
    for c in CLASSES:
        m = test['cls'] == c
        y = test['anom'][m]
        if y.min() == y.max():
            continue
        a = assigned[m]
        recon = test['spmean'][m]
        negc = -(L[m][np.arange(len(a)), a] / s_scale)
        nll = -logp[m][np.arange(len(a)), a]
        out['recon_spmax'][c] = roc_auc_score(y, test['spmax'][m])
        out['recon_spmean'][c] = roc_auc_score(y, recon)
        out['neg_cos'][c] = roc_auc_score(y, negc)
        out['logit_nll'][c] = roc_auc_score(y, nll)
        out['fusion_negcos'][c] = roc_auc_score(y, _z(recon) + _z(negc))
        if maha_banks is not None and c in maha_banks:
            mu, ic = maha_banks[c]
            mah = maha_score(E[m], mu, ic)
            out['maha_embed'][c] = roc_auc_score(y, mah)
            out['fusion_maha'][c] = roc_auc_score(y, _z(recon) + _z(mah))
    for r in list(out.keys()):
        out[r]['mean'] = float(np.mean([out[r][c] for c in CLASSES if c in out[r]]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-c', '--cfg_path', required=True)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--embed_dim', type=int, default=128)
    ap.add_argument('--sub', type=int, default=2)
    ap.add_argument('--arc_s', type=float, default=30.0)
    ap.add_argument('--arc_m', type=float, default=0.5)
    ap.add_argument('--lambda_cls', type=float, default=1.0, help='weight on the CE term')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--recon_size', type=int, default=64, help='anomaly-map spatial size')
    ap.add_argument('--recon_gauss', type=float, default=4.0)
    ap.add_argument('--eval_every', type=int, default=1)
    ap.add_argument('--eval_maha', action='store_true')
    ap.add_argument('--recon_baseline', type=float, default=0.72)
    ap.add_argument('--rungB_auroc', type=float, default=0.859)
    ap.add_argument('--rungC_auroc', type=float, default=0.844)
    ap.add_argument('--stgram_auroc', type=float, default=0.9075)
    ap.add_argument('--out_dir', default=None)
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cfg = build_cfg(args.cfg_path, args.batch, args.workers)
    cfg_name = os.path.splitext(os.path.basename(args.cfg_path))[0]
    out_dir = args.out_dir or os.path.join('runs', 'section_rungD', cfg_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[cfg] {args.cfg_path}  data.root={cfg.data.root}")

    train_loader, test_loader = get_loader(cfg)

    print("[setup] building section label map ...")
    all_sec = []
    for batch in train_loader:
        all_sec += [parse_section(c, p) for c, p in zip(batch['cls_name'], batch['img_path'])]
    sections = sorted(set(all_sec))
    sec2idx = {s: i for i, s in enumerate(sections)}
    print(f"  {len(sections)} sections, {len(all_sec)} normal-train clips")

    from model import get_model
    mambaad_net = get_model(cfg.model).to(device)
    mambaad_net.train()
    with torch.no_grad():
        probe = next(iter(train_loader))['img'][:2].to(device)
        _, feats_s = mambaad_net(probe)
        in_dim = sum(f.shape[1] for f in feats_s)
    model = MambaSectionNet(mambaad_net, in_dim, n_classes=len(sections), embed_dim=args.embed_dim,
                            sub=args.sub, s=args.arc_s, m=args.arc_m).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    cos_loss = CosLoss()
    ce = nn.CrossEntropyLoss()
    print(f"[train] joint CosLoss + {args.lambda_cls}*CE(ArcFace sub={args.sub})  "
          f"epochs={args.epochs} lr={args.lr}  (teacher frozen, student from scratch)")

    curve_path = os.path.join(out_dir, 'metric_curve.csv')
    train_log_path = os.path.join(out_dir, 'train_log.csv')
    with open(curve_path, 'w') as f:
        f.write('epoch,readout,' + ','.join(CLASSES) + ',mean\n')
    with open(train_log_path, 'w') as f:
        f.write('epoch,total,cos,ce,acc,skipped\n')

    best = defaultdict(lambda: (-1.0, -1, None))

    for ep in range(1, args.epochs + 1):
        model.train()
        tot, correct, l_tot, l_cos, l_ce, skipped = 0, 0, 0.0, 0.0, 0.0, 0
        for batch in train_loader:
            imgs = batch['img'].to(device, non_blocking=True)
            if imgs.size(0) == 1:
                continue
            y = torch.tensor([sec2idx[parse_section(c, p)]
                              for c, p in zip(batch['cls_name'], batch['img_path'])], device=device)
            feats_t, feats_s = model.net(imgs)
            emb = emb_from_featss(model, feats_s)
            logits = model.logits(emb, y)
            lc = cos_loss(feats_t, feats_s)
            le = ce(logits, y)
            loss = lc + args.lambda_cls * le
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                skipped += 1
                continue
            opt.zero_grad()
            loss.backward()
            opt.step()
            n = imgs.size(0)
            l_tot += loss.item() * n; l_cos += lc.item() * n; l_ce += le.item() * n
            correct += (logits.argmax(1) == y).sum().item()
            tot += n
        tr = lambda v: v / max(tot, 1)
        acc = 100.0 * correct / max(tot, 1)
        with open(train_log_path, 'a') as f:
            f.write(f'{ep},{tr(l_tot):.6f},{tr(l_cos):.6f},{tr(l_ce):.6f},{acc:.4f},{skipped}\n')

        if ep % args.eval_every != 0 and ep != args.epochs:
            print(f"  epoch {ep:3d}/{args.epochs}  cos={tr(l_cos):.4f} ce={tr(l_ce):.4f} acc={acc:.2f}% skip={skipped}")
            continue
        test = collect_D(model, test_loader, device, True, args.recon_size, args.recon_gauss)
        train = collect_D(model, train_loader, device, False, args.recon_size, args.recon_gauss) if args.eval_maha else None
        res = score_D(test, sec2idx, model.arc.s, train, args.eval_maha)
        with open(curve_path, 'a') as f:
            for r, d in res.items():
                f.write(f'{ep},{r},' + ','.join(f'{d.get(c, float("nan")):.4f}' for c in CLASSES)
                        + f',{d["mean"]:.4f}\n')
        for r, d in res.items():
            if d['mean'] > best[r][0]:
                best[r] = (d['mean'], ep, {c: d.get(c) for c in CLASSES})
        msg = '  '.join(f'{r}={res[r]["mean"] * 100:.1f}'
                        for r in ('recon_spmean', 'neg_cos', 'fusion_negcos', 'fusion_maha') if r in res)
        print(f"  epoch {ep:3d}/{args.epochs}  cos={tr(l_cos):.4f} ce={tr(l_ce):.4f} acc={acc:.2f}% skip={skipped}  {msg}")

    # ---- report ----
    order = ['recon_spmax', 'recon_spmean', 'neg_cos', 'logit_nll', 'maha_embed',
             'fusion_negcos', 'fusion_maha']
    rows, header = [], ['readout', 'best_epoch'] + CLASSES + ['mean']
    for r in order:
        if best[r][1] < 0:
            continue
        mean, epb, pc = best[r]
        rows.append([r, epb] + [f'{pc[c] * 100:.2f}' if pc[c] is not None else '  -  '
                                for c in CLASSES] + [f'{mean * 100:.2f}'])
    print("\n=== Rung D: joint distill+classify (best-epoch AUROC %) ===")
    print(tabulate.tabulate(rows, headers=header, tablefmt='github'))
    with open(os.path.join(out_dir, 'best_summary.csv'), 'w') as f:
        f.write(','.join(header) + '\n')
        for row in rows:
            f.write(','.join(str(x).strip() for x in row) + '\n')

    best_mean, best_ep, _ = max(best.values(), key=lambda t: t[0])
    best_name = max(best, key=lambda r: best[r][0])
    print(f"\n[out] {curve_path}")
    print(f"Rung D best = {best_mean * 100:.2f}% @ epoch {best_ep}  (readout: {best_name})")
    print(f"  recon-only best: {max(best['recon_spmax'][0], best['recon_spmean'][0]) * 100:.2f}% "
          f"(native MambaAD readout, but jointly trained)")
    for name, a in (('native recon-only ~', args.recon_baseline), ('Rung C labels-only ', args.rungC_auroc),
                    ('Rung B FT encoder  ', args.rungB_auroc), ('STgram-MFN         ', args.stgram_auroc)):
        print(f"  vs {name} {a * 100:.2f}%  ->  {(best_mean - a) * 100:+.2f}")
    fused_best = max(best['fusion_negcos'][0], best['fusion_maha'][0])
    single_best = max(best[r][0] for r in ('recon_spmean', 'recon_spmax', 'neg_cos', 'maha_embed'))
    if fused_best > single_best + 0.005:
        print(f"VERDICT: FUSION wins ({fused_best*100:.2f} > best single {single_best*100:.2f}) — the "
              "reconstruction and classification scores are complementary; 'MambaAD + labels' as a "
              "fused readout is the payoff of the joint objective.")
    else:
        print(f"VERDICT: fusion ({fused_best*100:.2f}) does not beat the best single readout "
              f"({single_best*100:.2f}) — the two objectives are largely redundant here.")


if __name__ == '__main__':
    main()
