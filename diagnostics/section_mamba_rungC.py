"""
Mamba-student section classifier — Rung C of the "labels close the gap?" ladder
===============================================================================

Rungs A/B established that the section-classification *objective* is the dominant lever
(A: frozen ResNet + head, readout-only +8.2 -> 80.0; B: fine-tune ResNet, representation
+5.9 -> 85.9). Both left MambaAD's Mamba decoder out entirely. Rung C puts it back:

    Take the actual MambaAD student (frozen ResNet34 teacher -> MFF/OCE fusion -> MambaUPNet
    decoder) and train it to CLASSIFY the 23 machine sections (ArcFace on GAP-pooled student
    features) instead of to RECONSTRUCT teacher features (the native CosLoss). Everything else
    matches A/B: log-Mel input, concat-GAP feature, neg_cos / logit_nll readouts, per-epoch
    test-AUROC logging (peaks-early rule).

This is "MambaAD with the objective swapped from generative to discriminative" — the first
rung that actually exercises the Mamba architecture. Read it against:
  * Rung A (frozen ResNet + head, 80.0) : C - A = value of the Mamba decoder as a trainable
    discriminative transform on the SAME frozen teacher features.
  * Rung B (fine-tuned ResNet + head, 85.9) : does a from-scratch Mamba student on frozen
    features match/beat simply fine-tuning the encoder?
  * STgram-MFN (90.75) : the supervised target.

The teacher stays frozen (as in real MambaAD); the trainable parts (MFF/OCE + MambaUPNet +
head) are randomly initialised, so — unlike Rung B's pretrained backbone — they train from
scratch at a single higher LR. A non-finite loss guard skips bad steps (MambaAD's known
lr-5e-3 divergence history; classification is milder but the guard is cheap insurance).

Usage
-----
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/anaconda3/envs/mamba-ad/lib
    CUDA_VISIBLE_DEVICES=0 python diagnostics/section_mamba_rungC.py \
        -c MambaAD/configs/mambaad/mimii/e50/log-Mel.py \
        --epochs 50 --sub 2 --lr 1e-3 --stgram_auroc 0.9075

Outputs (runs/section_rungC/<cfg_name>/): metric_curve.csv, best_summary.csv, train_log.csv
(identical schema to Rung B, so docs/plot_section_rungB.py-style plotting transfers).
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

from model import get_model
from data import get_loader
from diagnostics.frozen_encoder_probe import build_cfg
from diagnostics.section_classifier_probe import parse_section, ArcMarginProduct
# reuse Rung B's model-agnostic eval + scoring so the ONLY change is the model
from diagnostics.section_finetune_rungB import collect, clip_scores, auroc_by_class, CLASSES
from diagnostics.heldout_eval import ScoreDump, clip_keys


# --------------------------------------------------------------------------- #
# model: frozen ResNet34 teacher -> MFF/OCE -> MambaUPNet student -> GAP -> ArcFace
# --------------------------------------------------------------------------- #
class MambaSectionNet(nn.Module):
    def __init__(self, mambaad_net, in_dim, n_classes, embed_dim=128, sub=2,
                 s=30.0, m=0.5, use_arcface=True):
        super().__init__()
        self.net = mambaad_net                       # MAMBAAD; teacher frozen internally
        self.in_bn = nn.BatchNorm1d(in_dim)
        self.proj = nn.Linear(in_dim, embed_dim)
        self.emb_bn = nn.BatchNorm1d(embed_dim)
        self.use_arcface = use_arcface
        if use_arcface:
            self.arc = ArcMarginProduct(embed_dim, n_classes, s=s, m=m, sub=sub)
        else:
            self.fc = nn.Linear(embed_dim, n_classes)

    def embed(self, x):
        _, feats_s = self.net(x)                     # student's multi-scale reconstruction
        gap = torch.cat([f.mean(dim=(2, 3)) for f in feats_s], dim=1)
        return self.emb_bn(self.proj(self.in_bn(gap)))

    def logits(self, emb, label=None):
        if self.use_arcface:
            return self.arc(emb, label) if label is not None else self.arc.s * self.arc.cosine(emb)
        return self.fc(emb)

    def forward(self, x, label=None):
        emb = self.embed(x)
        return self.logits(emb, label), emb


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-c', '--cfg_path', required=True)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--embed_dim', type=int, default=128)
    ap.add_argument('--no_arcface', action='store_true')
    ap.add_argument('--arc_s', type=float, default=30.0)
    ap.add_argument('--arc_m', type=float, default=0.5)
    ap.add_argument('--sub', type=int, default=2)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--lr', type=float, default=1e-3, help='single LR (student+head are fresh)')
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--cosine', action='store_true')
    ap.add_argument('--eval_every', type=int, default=1)
    ap.add_argument('--eval_maha', action='store_true')
    ap.add_argument('--no_dump_scores', dest='dump_scores', action='store_false',
                    help='skip scores_by_epoch.npz (needed for honest held-out re-scoring)')
    ap.add_argument('--rungA_auroc', type=float, default=0.80)
    ap.add_argument('--rungB_auroc', type=float, default=0.859)
    ap.add_argument('--stgram_auroc', type=float, default=0.9075)
    ap.add_argument('--seed', type=int, default=0,
                    help='seed repeat (CUDA is not bit-exact regardless)')
    ap.add_argument('--out_dir', default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cfg = build_cfg(args.cfg_path, args.batch, args.workers)
    cfg_name = os.path.splitext(os.path.basename(args.cfg_path))[0]
    out_dir = args.out_dir or os.path.join('runs', 'section_rungC', f'{cfg_name}_seed{args.seed}')
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

    # build the full MambaAD net (teacher frozen inside), infer student feature dim
    mambaad_net = get_model(cfg.model).to(device)
    mambaad_net.train()
    with torch.no_grad():
        probe = next(iter(train_loader))['img'][:2].to(device)
        _, feats_s = mambaad_net(probe)
        in_dim = sum(f.shape[1] for f in feats_s)
    print(f"[model] MambaAD student feature dim (concat-GAP) = {in_dim}")

    model = MambaSectionNet(mambaad_net, in_dim, n_classes=len(sections),
                            embed_dim=args.embed_dim, sub=args.sub, s=args.arc_s,
                            m=args.arc_m, use_arcface=not args.no_arcface).to(device)

    params = [p for p in model.parameters() if p.requires_grad]  # teacher already excluded
    opt = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs) if args.cosine else None
    ce = nn.CrossEntropyLoss()
    obj = 'plain-CE' if args.no_arcface else f'ArcFace(s={args.arc_s},m={args.arc_m},sub={args.sub})'
    print(f"[train] Mamba student + {obj}  epochs={args.epochs} lr={args.lr} wd={args.wd} "
          f"cosine={args.cosine}  (teacher frozen, student from scratch)")

    curve_path = os.path.join(out_dir, 'metric_curve.csv')
    train_log_path = os.path.join(out_dir, 'train_log.csv')
    with open(curve_path, 'w') as f:
        f.write('epoch,readout,' + ','.join(CLASSES) + ',mean\n')
    with open(train_log_path, 'w') as f:
        f.write('epoch,train_loss,train_acc,skipped\n')

    best = defaultdict(lambda: (-1.0, -1, None))
    dump = None                                   # built at the first eval (needs clip meta)

    for ep in range(1, args.epochs + 1):
        model.train()                                # MAMBAAD.train() re-freezes teacher
        tot, correct, loss_sum, skipped = 0, 0, 0.0, 0
        for batch in train_loader:
            imgs = batch['img'].to(device, non_blocking=True)
            if imgs.size(0) == 1:
                continue
            y = torch.tensor([sec2idx[parse_section(c, p)]
                              for c, p in zip(batch['cls_name'], batch['img_path'])], device=device)
            logits, _ = model(imgs, y)
            loss = ce(logits, y)
            if not torch.isfinite(loss):             # non-finite guard (divergence insurance)
                opt.zero_grad(set_to_none=True)
                skipped += 1
                continue
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
            f.write(f'{ep},{tr_loss:.6f},{tr_acc:.4f},{skipped}\n')

        if ep % args.eval_every != 0 and ep != args.epochs:
            print(f"  epoch {ep:3d}/{args.epochs}  loss={tr_loss:.4f} acc={tr_acc:.2f}% skip={skipped}")
            continue
        s_scale = model.arc.s if not args.no_arcface else 1.0
        test_pack, te_paths = collect(model, test_loader, device, need_anom=True,
                                      return_paths=True)
        train_pack = None
        if args.eval_maha:
            Etr, _, cls_tr, _, _ = collect(model, train_loader, device, need_anom=False)
            train_pack = (Etr, cls_tr)
        sc = clip_scores(test_pack, sec2idx, s_scale, train_pack, args.eval_maha)
        res = auroc_by_class(sc, test_pack[2], test_pack[4])
        if dump is None and args.dump_scores:
            dump = ScoreDump(out_dir, clip_keys(te_paths), test_pack[2], test_pack[3],
                             test_pack[4])
        if dump is not None:
            dump.add(ep, sc)

        with open(curve_path, 'a') as f:
            for r, d in res.items():
                f.write(f'{ep},{r},' + ','.join(f'{d.get(c, float("nan")):.4f}' for c in CLASSES)
                        + f',{d["mean"]:.4f}\n')
        for r, d in res.items():
            if d['mean'] > best[r][0]:
                best[r] = (d['mean'], ep, {c: d.get(c) for c in CLASSES})
        msg = '  '.join(f'{r}={res[r]["mean"] * 100:.2f}' for r in ('neg_cos', 'logit_nll') if r in res)
        star = ' *' if best['neg_cos'][1] == ep else ''
        print(f"  epoch {ep:3d}/{args.epochs}  loss={tr_loss:.4f} acc={tr_acc:.2f}% skip={skipped}  {msg}{star}")

    # ---- final report ----
    rows, header = [], ['readout', 'best_epoch'] + CLASSES + ['mean']
    for r in ('neg_cos', 'logit_nll', 'maha_embed'):
        if best[r][1] < 0:
            continue
        mean, epb, pc = best[r]
        rows.append([r, epb] + [f'{pc[c] * 100:.2f}' if pc[c] is not None else '  -  '
                                for c in CLASSES] + [f'{mean * 100:.2f}'])
    print("\n=== Rung C: Mamba student + section classifier (best-epoch AUROC %) ===")
    print(tabulate.tabulate(rows, headers=header, tablefmt='github'))
    with open(os.path.join(out_dir, 'best_summary.csv'), 'w') as f:
        f.write(','.join(header) + '\n')
        for row in rows:
            f.write(','.join(str(x).strip() for x in row) + '\n')

    best_mean, best_ep, _ = max(best.values(), key=lambda t: t[0])
    print(f"\n[out] {curve_path}")
    print(f"Rung C best = {best_mean * 100:.2f}% @ epoch {best_ep}")
    for name, a in (('Rung A frozen ', args.rungA_auroc), ('Rung B FT enc ', args.rungB_auroc),
                    ('STgram-MFN    ', args.stgram_auroc)):
        print(f"  vs {name} {a * 100:.2f}%  ->  {(best_mean - a) * 100:+.2f}")
    if best_mean >= args.rungB_auroc - 0.005:
        print("VERDICT: the Mamba student matches/beats fine-tuning the encoder — the decoder "
              "architecture carries the discriminative objective at least as well.")
    elif best_mean > args.rungA_auroc + 0.01:
        print("VERDICT: the Mamba decoder adds value over a frozen head but trails fine-tuning "
              "the encoder — architecture helps, representation (encoder) helps more.")
    else:
        print("VERDICT: the Mamba student does NOT beat the frozen-head readout — the decoder "
              "architecture is not the lever; the encoder representation is.")


if __name__ == '__main__':
    main()
