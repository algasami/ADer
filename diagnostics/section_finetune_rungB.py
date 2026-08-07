"""
Section-classifier fine-tuning — Rung B of the "labels close the gap?" ladder
=============================================================================

Rung A (`diagnostics/section_classifier_probe.py`) trained a head on a FROZEN ResNet34
and found the discriminative objective helps only as a test-time *readout* (+8 AUROC),
not as a representation — Maha on the trained embedding actually dropped. That is the
setup for Rung B:

    UNFREEZE ResNet34 and fine-tune it end-to-end under the SAME ArcFace section-
    classification loss. Everything else is held fixed vs Rung A (same log-Mel input,
    same concat-GAP feature, same 23 sections parsed from img_path, same neg_cos /
    logit_nll readouts). The ONLY change is frozen -> trainable encoder, so the delta
    over Rung A's converged ceiling (~80 AUROC) measures how much of the remaining ~+11
    gap to STgram-MFN (90.75) is *representation adaptation*.

This is essentially STgram-MFN with a ResNet/log-Mel front-end instead of MFN+TgramNet:
a pure classification model, so the MambaAD teacher->student decoder plays no role here
(the Mamba architecture re-enters at Rung C/D). See docs/plots/mimii_section_rungA/CONCLUSION.md.

Why per-epoch eval matters
--------------------------
This repo's hard-won rule: MIMII AUROC PEAKS EARLY and there is no best-checkpoint
selection, so the final epoch understates the model. This script therefore scores the
test split with the STgram-style readouts EVERY epoch and logs the whole curve, so you
compare Rung A vs Rung B at their *peaks*, not at the last epoch.

Anomaly readout (metadata known at test time): score = poor fit to a clip's OWN section.
  * neg_cos    = -cos(embedding, assigned-section center)   (margin-free)
  * logit_nll  = -log softmax p(assigned section)
  * maha_embed = Mahalanobis on the fine-tuned embedding (optional, --eval_maha; the
                 Rung A negative — watch whether fine-tuning finally makes it help)

Usage
-----
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/anaconda3/envs/mamba-ad/lib
    CUDA_VISIBLE_DEVICES=0 python diagnostics/section_finetune_rungB.py \
        -c MambaAD/configs/mambaad/mimii/e50/log-Mel.py \
        --epochs 50 --sub 2 --stgram_auroc 0.9075 --rungA_auroc 0.80

Outputs (runs/section_rungB/<cfg_name>/):
    metric_curve.csv    epoch,readout,<6 classes>,mean   (long form, one eval per row)
    best_summary.csv    best epoch + per-class AUROC per readout
    train_log.csv       epoch,train_loss,train_acc
"""

import os
import sys
import json
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import timm
from timm.models._helpers import load_checkpoint
from sklearn.metrics import roc_auc_score, average_precision_score
import tabulate

from data import get_loader
# reuse Rung A pieces so the ONLY change vs Rung A is the trainable encoder
from diagnostics.frozen_encoder_probe import build_cfg, fit_mahalanobis, maha_score
from diagnostics.section_classifier_probe import parse_section, ArcMarginProduct
from diagnostics.spec_augment import BatchAug, mixup_batch, mixup_arcface_loss

CLASSES = ["fan", "pump", "slider", "valve", "ToyCar", "ToyConveyor"]
_FROZEN_PREFIXES = ("conv1", "bn1", "layer1")  # what --freeze_stem pins

# Augmentation presets. Mask widths are POST-RESIZE rows/cols on the 256x256 tensor,
# so 1 mel bin = 2 rows. Frequency masking is kept narrow on purpose: machine identity
# is a spectral signature and the training objective discriminates machine sections.
AUG_PRESETS = {
    'none':     dict(time_crop_min=0.0, time_shift=0.0,
                     freq_mask=(0, 0), time_mask=(0, 0), mixup=0.0),
    'mild':     dict(time_crop_min=0.7, time_shift=0.10,
                     freq_mask=(1, 16), time_mask=(1, 32), mixup=0.0),
    'standard': dict(time_crop_min=0.5, time_shift=0.20,
                     freq_mask=(2, 24), time_mask=(2, 48), mixup=0.2),
}
AUG_KEYS = tuple(AUG_PRESETS['none'])


def resolve_aug_args(args):
    """Fill unset (None) augmentation fields from the chosen preset, in place.

    The aug flags default to None rather than 0 so that an EXPLICIT `--mixup 0`
    is distinguishable from "not passed". With 0 defaults there is no way to tell
    them apart, and the preset silently overwrites the explicit value -- which is
    exactly what turned the first mixup ablation into a duplicate of the preset run.
    """
    preset = AUG_PRESETS[getattr(args, 'aug_preset', None) or 'none']
    for k in AUG_KEYS:
        if getattr(args, k, None) is None:
            setattr(args, k, preset[k])
    return args


# --------------------------------------------------------------------------- #
# model: trainable ResNet34 backbone -> concat-GAP -> embedding -> ArcFace
# --------------------------------------------------------------------------- #
class SectionNet(nn.Module):
    def __init__(self, cfg, n_classes, embed_dim=128, sub=1, s=30.0, m=0.5,
                 use_arcface=True):
        super().__init__()
        mt = cfg.model_t
        assert mt.name == 'timm_resnet34', f"Rung B assumes resnet34; config has {mt.name}"
        kw = dict(mt.kwargs)
        self.out_indices = kw.get('out_indices', [1, 2, 3])
        self.backbone = timm.create_model(
            'resnet34', pretrained=False, features_only=True, out_indices=self.out_indices)
        ckpt = kw.get('checkpoint_path', '')
        if ckpt and os.path.isfile(ckpt):
            load_checkpoint(self.backbone, ckpt, strict=kw.get('strict', False))
            print(f"[backbone] loaded ImageNet teacher weights: {ckpt}")
        else:
            print(f"[backbone] WARNING: checkpoint '{ckpt}' missing; ImageNet init absent.")
        in_dim = sum(self.backbone.feature_info.channels())
        self.in_bn = nn.BatchNorm1d(in_dim)          # adapts feature scale during FT
        self.proj = nn.Linear(in_dim, embed_dim)
        self.emb_bn = nn.BatchNorm1d(embed_dim)
        self.use_arcface = use_arcface
        if use_arcface:
            self.arc = ArcMarginProduct(embed_dim, n_classes, s=s, m=m, sub=sub)
        else:
            self.fc = nn.Linear(embed_dim, n_classes)

    def embed(self, x):
        feats = self.backbone(x)
        gap = torch.cat([f.mean(dim=(2, 3)) for f in feats], dim=1)
        return self.emb_bn(self.proj(self.in_bn(gap)))

    def logits(self, emb, label=None):
        if self.use_arcface:
            return self.arc(emb, label) if label is not None else self.arc.s * self.arc.cosine(emb)
        return self.fc(emb)

    def forward(self, x, label=None):
        emb = self.embed(x)
        return self.logits(emb, label), emb

    # ---- partial-freeze helpers (optional stabiliser) ----
    def freeze_stem(self):
        for n, p in self.backbone.named_parameters():
            if n.startswith(_FROZEN_PREFIXES):
                p.requires_grad = False

    def set_train_mode(self, frozen_stem):
        self.train()
        if frozen_stem:  # keep frozen BN stats fixed
            for n, mod in self.backbone.named_modules():
                if n.startswith(_FROZEN_PREFIXES):
                    mod.eval()


# --------------------------------------------------------------------------- #
# eval: score the test split with the STgram-style readouts
# --------------------------------------------------------------------------- #
@torch.no_grad()
def collect(model, loader, device, need_anom):
    embs, logs, clss, secs, anoms = [], [], [], [], []
    model.eval()
    for batch in loader:
        imgs = batch['img'].to(device, non_blocking=True)
        cls = np.array(batch['cls_name'])
        emb = model.embed(imgs)
        embs.append(emb.cpu().numpy())
        logs.append(model.logits(emb).cpu().numpy())
        clss.append(cls)
        secs.append(np.array([parse_section(c, p) for c, p in zip(cls, batch['img_path'])]))
        if need_anom:
            a = batch['anomaly']
            anoms.append(a.numpy() if torch.is_tensor(a) else np.array(a))
    E = np.concatenate(embs).astype(np.float32)
    L = np.concatenate(logs).astype(np.float32)
    cls = np.concatenate(clss)
    sec = np.concatenate(secs)
    anom = np.concatenate(anoms).astype(int) if need_anom else None
    return E, L, cls, sec, anom


def score_epoch(model, test_pack, sec2idx, s_scale, train_pack=None, eval_maha=False):
    """Return {readout: {class: auroc, 'mean': mean}} for one eval."""
    Ete, Lte, cls_te, sec_te, y_anom = test_pack
    logp = Lte - torch.logsumexp(torch.from_numpy(Lte), dim=1, keepdim=True).numpy()
    assigned = np.array([sec2idx.get(s, -1) for s in sec_te])

    maha_banks = None
    if eval_maha and train_pack is not None:
        Etr, cls_tr = train_pack
        maha_banks = {c: fit_mahalanobis(Etr[cls_tr == c]) for c in CLASSES if (cls_tr == c).any()}

    out = defaultdict(dict)
    for c in CLASSES:
        m = cls_te == c
        y = y_anom[m]
        if y.min() == y.max():
            continue
        a = assigned[m]
        out['neg_cos'][c] = roc_auc_score(y, -(Lte[m][np.arange(len(a)), a] / s_scale))
        out['logit_nll'][c] = roc_auc_score(y, -logp[m][np.arange(len(a)), a])
        if maha_banks is not None and c in maha_banks:
            mu, ic = maha_banks[c]
            out['maha_embed'][c] = roc_auc_score(y, maha_score(Ete[m], mu, ic))
    for r in list(out.keys()):
        out[r]['mean'] = float(np.mean([out[r][c] for c in CLASSES if c in out[r]]))
    return out


def id_level_auroc(test_pack, sec2idx, s_scale, train_pack=None, eval_maha=False,
                    target_classes=('ToyCar', 'ToyConveyor')):
    """Same as score_epoch but broken down by machine SECTION (type/id) instead of class,
    restricted to target_classes. Used for the ToyCar/ToyConveyor frontier analysis — the
    class-level ladder numbers pool all machine IDs together, which hides whether the
    residual gap to STgram-MFN is spread evenly across IDs or concentrated in one (as it is
    for STgram-MFN itself, e.g. ToyConveyor id_02 / ToyCar id_01)."""
    Ete, Lte, cls_te, sec_te, y_anom = test_pack
    assigned = np.array([sec2idx.get(s, -1) for s in sec_te])

    maha_banks = None
    if eval_maha and train_pack is not None:
        Etr, cls_tr = train_pack
        maha_banks = {c: fit_mahalanobis(Etr[cls_tr == c]) for c in target_classes
                      if (cls_tr == c).any()}

    out = {}
    for c in target_classes:
        cm = cls_te == c
        for sec in sorted(set(sec_te[cm])):
            m = sec_te == sec
            y = y_anom[m]
            if y.min() == y.max():
                continue
            a = assigned[m]
            row = {'n': int(m.sum()),
                   'neg_cos': roc_auc_score(y, -(Lte[m][np.arange(len(a)), a] / s_scale))}
            if maha_banks is not None and c in maha_banks:
                mu, ic = maha_banks[c]
                row['maha_embed'] = roc_auc_score(y, maha_score(Ete[m], mu, ic))
            out[sec] = row
    return out


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-c', '--cfg_path', required=True)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--workers', type=int, default=8)
    # objective / head (mirror Rung A)
    ap.add_argument('--embed_dim', type=int, default=128)
    ap.add_argument('--no_arcface', action='store_true')
    ap.add_argument('--arc_s', type=float, default=30.0)
    ap.add_argument('--arc_m', type=float, default=0.5)
    ap.add_argument('--sub', type=int, default=2, help='ArcFace sub-clusters (Rung A best)')
    # fine-tuning schedule
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--lr_backbone', type=float, default=1e-4, help='low LR to avoid forgetting')
    ap.add_argument('--lr_head', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--cosine', action='store_true', help='cosine-anneal both LRs to 0')
    ap.add_argument('--freeze_stem', action='store_true',
                    help='keep conv1/bn1/layer1 frozen (stabiliser; only fine-tune layer2/3)')
    # --- augmentation (Phase 0: all default OFF, so the existing 3-seed no-aug runs
    #     stay a valid control and this script reproduces them bit-for-bit-ish) ---
    #     NOTE: these default to None, not 0, so that an EXPLICIT `--mixup 0` is
    #     distinguishable from "left unset". With a 0 default the preset resolution
    #     below cannot tell the two apart and silently overwrites the explicit value --
    #     which turned a mixup ablation into a duplicate of the preset run.
    ap.add_argument('--time_crop_min', type=float, default=None,
                    help='random time crop: min window fraction (e.g. 0.5); 0 = off')
    ap.add_argument('--time_shift', type=float, default=None,
                    help='random circular time shift, max fraction of width; 0 = off')
    ap.add_argument('--freq_mask', type=int, nargs=2, default=None, metavar=('N', 'W'),
                    help='SpecAugment frequency masks: count and max width in ROWS '
                         '(post-resize; 1 mel bin = 2 rows). Keep narrow: machine identity '
                         'IS a spectral signature and the loss discriminates machines.')
    ap.add_argument('--time_mask', type=int, nargs=2, default=None, metavar=('N', 'W'),
                    help='SpecAugment time masks: count and max width in columns')
    ap.add_argument('--mixup', type=float, default=None,
                    help='mixup Beta(a,a) alpha; 0 = off. Uses the two-label ArcFace form.')
    ap.add_argument('--aug_preset', choices=['none', 'mild', 'standard'], default=None,
                    help='shorthand for the flags above (explicit flags win)')
    ap.add_argument('--eval_every', type=int, default=1)
    ap.add_argument('--eval_maha', action='store_true',
                    help='also fit+score maha_embed each eval (extra train pass; slower)')
    # anchors for the printed verdict
    ap.add_argument('--rungA_auroc', type=float, default=0.80, help='Rung A ceiling to beat')
    ap.add_argument('--stgram_auroc', type=float, default=0.9075)
    ap.add_argument('--seed', type=int, default=0, help='seed repeat (CUDA is not bit-exact regardless)')
    ap.add_argument('--out_dir', default=None)
    args = ap.parse_args()

    resolve_aug_args(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cfg = build_cfg(args.cfg_path, args.batch, args.workers)
    cfg_name = os.path.splitext(os.path.basename(args.cfg_path))[0]
    out_dir = args.out_dir or os.path.join('runs', 'section_rungB', f'{cfg_name}_seed{args.seed}')
    os.makedirs(out_dir, exist_ok=True)
    print(f"[cfg] {args.cfg_path}  data.root={cfg.data.root}")

    train_loader, test_loader = get_loader(cfg)

    # build the global section map from a quick pass over the (normal-only) train split
    print("[setup] building section label map ...")
    all_sec = []
    for batch in train_loader:
        all_sec += [parse_section(c, p) for c, p in zip(batch['cls_name'], batch['img_path'])]
    sections = sorted(set(all_sec))
    sec2idx = {s: i for i, s in enumerate(sections)}
    print(f"  {len(sections)} sections, {len(all_sec)} normal-train clips")

    model = SectionNet(cfg, n_classes=len(sections), embed_dim=args.embed_dim,
                       sub=args.sub, s=args.arc_s, m=args.arc_m,
                       use_arcface=not args.no_arcface).to(device)
    if args.freeze_stem:
        model.freeze_stem()
        print("[backbone] stem frozen (conv1/bn1/layer1); fine-tuning layer2/3 + head")

    # two-group optimiser: tiny LR for pretrained backbone, larger for the fresh head
    bb_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and not n.startswith('backbone.')]
    opt = torch.optim.Adam([
        {'params': bb_params, 'lr': args.lr_backbone},
        {'params': head_params, 'lr': args.lr_head},
    ], weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs) if args.cosine else None
    ce = nn.CrossEntropyLoss()
    obj = 'plain-CE' if args.no_arcface else f'ArcFace(s={args.arc_s},m={args.arc_m},sub={args.sub})'
    print(f"[train] {obj}  epochs={args.epochs}  lr_bb={args.lr_backbone} lr_head={args.lr_head} "
          f"wd={args.wd}  cosine={args.cosine}")

    # augmentation is TRAIN-ONLY and applied on the GPU batch, so the eval path
    # (collect()) is byte-identical to every previous rung -- the readouts stay comparable
    aug = BatchAug(time_crop_min=args.time_crop_min, time_shift=args.time_shift,
                   n_freq=args.freq_mask[0], freq_w=args.freq_mask[1],
                   n_time=args.time_mask[0], time_w=args.time_mask[1])
    print(f"[aug]   preset={args.aug_preset or 'off'}  {aug.describe()}  mixup={args.mixup}")
    with open(os.path.join(out_dir, 'aug_config.json'), 'w') as f:
        json.dump(dict(preset=args.aug_preset, mixup=args.mixup, seed=args.seed,
                       **aug.as_dict()), f, indent=2)

    curve_path = os.path.join(out_dir, 'metric_curve.csv')
    train_log_path = os.path.join(out_dir, 'train_log.csv')
    id_breakdown_path = os.path.join(out_dir, 'id_breakdown.csv')
    with open(curve_path, 'w') as f:
        f.write('epoch,readout,' + ','.join(CLASSES) + ',mean\n')
    with open(train_log_path, 'w') as f:
        f.write('epoch,train_loss,train_acc\n')
    with open(id_breakdown_path, 'w') as f:
        f.write('epoch,section,n,neg_cos,maha_embed\n')

    best = defaultdict(lambda: (-1.0, -1, None))  # readout -> (mean, epoch, per-class dict)

    for ep in range(1, args.epochs + 1):
        # ---- train one epoch ----
        model.set_train_mode(args.freeze_stem)
        tot, correct, loss_sum = 0, 0, 0.0
        for batch in train_loader:
            imgs = batch['img'].to(device, non_blocking=True)
            if imgs.size(0) == 1:
                continue  # BatchNorm needs >1 sample
            y = torch.tensor([sec2idx[parse_section(c, p)]
                              for c, p in zip(batch['cls_name'], batch['img_path'])],
                             device=device)
            imgs = aug(imgs)
            imgs, perm, lam = mixup_batch(imgs, args.mixup)
            y_b = y[perm] if perm is not None else None
            emb = model.embed(imgs)
            loss = mixup_arcface_loss(model, emb, y, y_b, lam, ce)
            with torch.no_grad():               # margin-free logits, train-acc readout only
                logits = model.logits(emb.detach())   # under mixup, acc is vs the dominant label
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            tot += imgs.size(0)
        if sched is not None:
            sched.step()
        tr_loss, tr_acc = loss_sum / max(tot, 1), 100.0 * correct / max(tot, 1)
        with open(train_log_path, 'a') as f:
            f.write(f'{ep},{tr_loss:.6f},{tr_acc:.4f}\n')

        # ---- eval ----
        if ep % args.eval_every != 0 and ep != args.epochs:
            print(f"  epoch {ep:3d}/{args.epochs}  loss={tr_loss:.4f} acc={tr_acc:.2f}%")
            continue
        s_scale = model.arc.s if not args.no_arcface else 1.0
        test_pack = collect(model, test_loader, device, need_anom=True)
        train_pack = None
        if args.eval_maha:
            Etr, _, cls_tr, _, _ = collect(model, train_loader, device, need_anom=False)
            train_pack = (Etr, cls_tr)
        res = score_epoch(model, test_pack, sec2idx, s_scale, train_pack, args.eval_maha)
        id_res = id_level_auroc(test_pack, sec2idx, s_scale, train_pack, args.eval_maha)

        with open(curve_path, 'a') as f:
            for r, d in res.items():
                f.write(f'{ep},{r},' + ','.join(f'{d.get(c, float("nan")):.4f}' for c in CLASSES)
                        + f',{d["mean"]:.4f}\n')
        with open(id_breakdown_path, 'a') as f:
            for sec, row in id_res.items():
                f.write(f'{ep},{sec},{row["n"]},{row.get("neg_cos", float("nan")):.4f},'
                        f'{row.get("maha_embed", float("nan")):.4f}\n')
        for r, d in res.items():
            if d['mean'] > best[r][0]:
                best[r] = (d['mean'], ep, {c: d.get(c) for c in CLASSES})
        msg = '  '.join(f'{r}={res[r]["mean"] * 100:.2f}' for r in ('neg_cos', 'logit_nll') if r in res)
        star = ' *' if best['neg_cos'][1] == ep else ''
        print(f"  epoch {ep:3d}/{args.epochs}  loss={tr_loss:.4f} acc={tr_acc:.2f}%  {msg}{star}")

    # ---- final report ----
    rows, header = [], ['readout', 'best_epoch'] + CLASSES + ['mean']
    for r in ('neg_cos', 'logit_nll', 'maha_embed'):
        if best[r][1] < 0:
            continue
        mean, epb, pc = best[r]
        rows.append([r, epb] + [f'{pc[c] * 100:.2f}' if pc[c] is not None else '  -  '
                                for c in CLASSES] + [f'{mean * 100:.2f}'])
    print("\n=== Rung B: fine-tuned ResNet34 + section classifier (best-epoch AUROC %) ===")
    print(tabulate.tabulate(rows, headers=header, tablefmt='github'))
    with open(os.path.join(out_dir, 'best_summary.csv'), 'w') as f:
        f.write(','.join(header) + '\n')
        for row in rows:
            f.write(','.join(str(x).strip() for x in row) + '\n')

    best_mean, best_ep, _ = max(best.values(), key=lambda t: t[0])
    print(f"\n[out] {curve_path}")
    print(f"Rung B best = {best_mean * 100:.2f}% @ epoch {best_ep}")
    print(f"  vs Rung A ceiling {args.rungA_auroc * 100:.2f}%  ->  {(best_mean - args.rungA_auroc) * 100:+.2f} "
          f"(representation adaptation)")
    print(f"  vs STgram-MFN     {args.stgram_auroc * 100:.2f}%  ->  {(best_mean - args.stgram_auroc) * 100:+.2f} "
          f"(remaining gap)")
    if best_mean >= args.stgram_auroc - 0.01:
        print("VERDICT: fine-tuning CLOSES the gap — objective + representation was the whole story.")
    elif best_mean > args.rungA_auroc + 0.01:
        print("VERDICT: fine-tuning helps beyond the readout — representation adaptation is real; "
              "whatever remains is input (Tgram/raw wave) + architecture (MFN).")
    else:
        print("VERDICT: fine-tuning does NOT beat the frozen readout — the lever is not the "
              "ResNet representation; look to the audio-native input / architecture.")


if __name__ == '__main__':
    main()
