"""
B+C combined — Rung E of the "labels close the gap?" ladder
===========================================================

Rung B fine-tuned the ResNet ENCODER under the section loss (representation, 85.9). Rung C
trained the Mamba DECODER on a *frozen* encoder (architecture, 84.4). Rung E combines them:

    fine-tune the ResNet teacher AND the Mamba student TOGETHER under the same ArcFace section
    loss (classification only — the reconstruction objective was Rung D's dead end).

Question: does encoder adaptation (B, the single biggest lever) STACK on top of the Mamba
decoder (C)? Or do they overlap, so the combination just matches the better of the two?

Architecture = Rung C's (ResNet34 -> MFF/OCE -> MambaUPNet -> GAP -> ArcFace), but now the
teacher is ALSO trainable. Two subtleties vs C:
  * MAMBAAD.forward *detaches* the teacher features (teacher is frozen by design). Rung E
    therefore bypasses MAMBAAD.forward and calls net_t -> mff_oce -> net_s directly, so
    gradients flow into the encoder. (`MambaSectionNetE.embed` below.)
  * two LR groups: low LR for the pretrained teacher (like B), higher for the from-scratch
    MFF/OCE + Mamba student + head (like C).

Same input/feature/readouts/per-epoch eval as B/C. Read against Rung B (85.9), Rung C (84.4),
STgram-MFN (90.75). Non-finite guard on (now training encoder + Mamba: divergence insurance).

Usage
-----
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/anaconda3/envs/mamba-ad/lib
    CUDA_VISIBLE_DEVICES=0 python diagnostics/section_combined_rungE.py \
        -c MambaAD/configs/mambaad/mimii/e50/log-Mel.py \
        --epochs 50 --sub 2 --lr_teacher 1e-4 --lr_rest 1e-3 --eval_maha

Outputs (runs/section_rungE/<cfg_name>/): metric_curve.csv, best_summary.csv, train_log.csv.
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
from diagnostics.section_classifier_probe import parse_section
from diagnostics.section_finetune_rungB import collect, score_epoch, CLASSES
from diagnostics.section_mamba_rungC import MambaSectionNet


class MambaSectionNetE(MambaSectionNet):
    """Rung C's model, but embed() bypasses MAMBAAD.forward's teacher-feature detach so the
    encoder receives gradients (B+C: teacher AND student are trainable)."""
    def embed(self, x):
        feats_t = self.net.net_t(x)               # trainable teacher, NOT detached
        feats_s = self.net.net_s(self.net.mff_oce(feats_t))
        gap = torch.cat([f.mean(dim=(2, 3)) for f in feats_s], dim=1)
        return self.emb_bn(self.proj(self.in_bn(gap)))


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
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--lr_teacher', type=float, default=1e-4, help='pretrained encoder (like B)')
    ap.add_argument('--lr_rest', type=float, default=1e-3, help='MFF/OCE + Mamba + head (like C)')
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--eval_every', type=int, default=1)
    ap.add_argument('--eval_maha', action='store_true')
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
    out_dir = args.out_dir or os.path.join('runs', 'section_rungE', cfg_name)
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

    # build MambaAD, then UNFREEZE the teacher (B+C: everything trains)
    mambaad_net = get_model(cfg.model).to(device)
    mambaad_net.frozen_layers = []                 # stop MAMBAAD.train() re-freezing net_t
    for p in mambaad_net.net_t.parameters():
        p.requires_grad = True
    mambaad_net.train()
    with torch.no_grad():
        feats_t = mambaad_net.net_t(next(iter(train_loader))['img'][:2].to(device))
        feats_s = mambaad_net.net_s(mambaad_net.mff_oce(feats_t))
        in_dim = sum(f.shape[1] for f in feats_s)
    model = MambaSectionNetE(mambaad_net, in_dim, n_classes=len(sections), embed_dim=args.embed_dim,
                             sub=args.sub, s=args.arc_s, m=args.arc_m).to(device)

    teacher_params = list(model.net.net_t.parameters())
    tids = {id(p) for p in teacher_params}
    rest_params = [p for p in model.parameters() if p.requires_grad and id(p) not in tids]
    opt = torch.optim.Adam([
        {'params': teacher_params, 'lr': args.lr_teacher},
        {'params': rest_params, 'lr': args.lr_rest},
    ], weight_decay=args.wd)
    ce = nn.CrossEntropyLoss()
    print(f"[train] B+C: fine-tune teacher (lr={args.lr_teacher}) + Mamba student/head "
          f"(lr={args.lr_rest})  ArcFace sub={args.sub}  epochs={args.epochs}")

    curve_path = os.path.join(out_dir, 'metric_curve.csv')
    train_log_path = os.path.join(out_dir, 'train_log.csv')
    with open(curve_path, 'w') as f:
        f.write('epoch,readout,' + ','.join(CLASSES) + ',mean\n')
    with open(train_log_path, 'w') as f:
        f.write('epoch,train_loss,train_acc,skipped\n')

    best = defaultdict(lambda: (-1.0, -1, None))

    for ep in range(1, args.epochs + 1):
        model.train()
        tot, correct, loss_sum, skipped = 0, 0, 0.0, 0
        for batch in train_loader:
            imgs = batch['img'].to(device, non_blocking=True)
            if imgs.size(0) == 1:
                continue
            y = torch.tensor([sec2idx[parse_section(c, p)]
                              for c, p in zip(batch['cls_name'], batch['img_path'])], device=device)
            logits, _ = model(imgs, y)
            loss = ce(logits, y)
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                skipped += 1
                continue
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            tot += imgs.size(0)
        tr_loss, tr_acc = loss_sum / max(tot, 1), 100.0 * correct / max(tot, 1)
        with open(train_log_path, 'a') as f:
            f.write(f'{ep},{tr_loss:.6f},{tr_acc:.4f},{skipped}\n')

        if ep % args.eval_every != 0 and ep != args.epochs:
            print(f"  epoch {ep:3d}/{args.epochs}  loss={tr_loss:.4f} acc={tr_acc:.2f}% skip={skipped}")
            continue
        s_scale = model.arc.s
        test_pack = collect(model, test_loader, device, need_anom=True)
        train_pack = None
        if args.eval_maha:
            Etr, _, cls_tr, _, _ = collect(model, train_loader, device, need_anom=False)
            train_pack = (Etr, cls_tr)
        res = score_epoch(model, test_pack, sec2idx, s_scale, train_pack, args.eval_maha)

        with open(curve_path, 'a') as f:
            for r, d in res.items():
                f.write(f'{ep},{r},' + ','.join(f'{d.get(c, float("nan")):.4f}' for c in CLASSES)
                        + f',{d["mean"]:.4f}\n')
        for r, d in res.items():
            if d['mean'] > best[r][0]:
                best[r] = (d['mean'], ep, {c: d.get(c) for c in CLASSES})
        msg = '  '.join(f'{r}={res[r]["mean"] * 100:.2f}' for r in ('neg_cos', 'maha_embed') if r in res)
        star = ' *' if max(best, key=lambda k: best[k][0]) and best[max(best, key=lambda k: best[k][0])][1] == ep else ''
        print(f"  epoch {ep:3d}/{args.epochs}  loss={tr_loss:.4f} acc={tr_acc:.2f}% skip={skipped}  {msg}{star}")

    # ---- report ----
    rows, header = [], ['readout', 'best_epoch'] + CLASSES + ['mean']
    for r in ('neg_cos', 'logit_nll', 'maha_embed'):
        if best[r][1] < 0:
            continue
        mean, epb, pc = best[r]
        rows.append([r, epb] + [f'{pc[c] * 100:.2f}' if pc[c] is not None else '  -  '
                                for c in CLASSES] + [f'{mean * 100:.2f}'])
    print("\n=== Rung E: B+C combined (fine-tune teacher + Mamba student) best-epoch AUROC % ===")
    print(tabulate.tabulate(rows, headers=header, tablefmt='github'))
    with open(os.path.join(out_dir, 'best_summary.csv'), 'w') as f:
        f.write(','.join(header) + '\n')
        for row in rows:
            f.write(','.join(str(x).strip() for x in row) + '\n')

    best_mean, best_ep, _ = max(best.values(), key=lambda t: t[0])
    best_name = max(best, key=lambda r: best[r][0])
    print(f"\n[out] {curve_path}")
    print(f"Rung E best = {best_mean * 100:.2f}% @ epoch {best_ep}  (readout: {best_name})")
    for name, a in (('Rung B FT encoder ', args.rungB_auroc), ('Rung C Mamba only ', args.rungC_auroc),
                    ('STgram-MFN        ', args.stgram_auroc)):
        print(f"  vs {name} {a * 100:.2f}%  ->  {(best_mean - a) * 100:+.2f}")
    better = max(args.rungB_auroc, args.rungC_auroc)
    if best_mean >= better + 0.01:
        print("VERDICT: B+C STACK — combining encoder adaptation with the Mamba decoder beats "
              "either alone. The levers are complementary.")
    elif best_mean >= better - 0.01:
        print("VERDICT: B+C do NOT stack — the combination matches the better single lever "
              "(encoder adaptation), so the Mamba decoder adds nothing on top of fine-tuning.")
    else:
        print("VERDICT: B+C combined UNDERperforms the better single lever — jointly training "
              "encoder + from-scratch Mamba is harder to optimise than fine-tuning the encoder alone.")


if __name__ == '__main__':
    main()
