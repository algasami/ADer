"""
Rung G — AST encoder in the Rung B slot (Phase 1 of branch `aug-ast-phases`)
===========================================================================

Rung B fine-tuned a trainable ResNet34 on log-Mel PNGs under an ArcFace loss over the 23
MIMII machine sections, and settled at 85.89 +/- 0.26 mean image-AUROC (3 seeds) -- the
biggest reproducible lever on the labels/objective ladder. The last untested feature-level
lever is the ENCODER: `diagnostics/audio_backbone_probe.py` measured AST (AudioSet) at 76.8
vs ResNet34's 71.8 under an identical frozen Maha readout, +5.0. That probe was frozen-only;
training it end-to-end was skipped as a time decision, NOT refuted.

Rung G is exactly Rung B with the encoder swapped:

    trainable AST (mean patch tokens) -> Linear -> BN -> ArcFace(23 sections)

Everything else is held fixed against Rung B on purpose -- same 23 sections, same
`neg_cos` / `logit_nll` / `maha_embed` readouts, same per-epoch eval, same CSV layout -- so
the delta is attributable to the encoder alone.

Two deliberate differences, both forced by the encoder, both flagged in the output:
  1. INPUT. AST wants its own 128-mel kaldi fbank at 16 kHz with AudioSet normalization, so
     Rung G reads the raw wavs under `data/dcase-2020/` rather than the 8-bit PNGs. The
     split is the same (verified identical counts: 20119 normal-train / 10868 test clips,
     23 sections); this also sidesteps the PNG quantization ceiling, which is a confound
     worth naming rather than hiding.
  2. NO SPATIAL PYRAMID. Rung B concatenates GAP over 3 ResNet stages; AST emits one token
     sequence, so the embedding is the mean of the patch tokens (`ast_meanpatch`, the
     winning tap in the frozen probe -- it beat the pooler by 3.3, about the size of the
     headline effect).

Augmentation: MIXUP ONLY, per the Phase 0 result (`docs/plots/phase0_aug/CONCLUSION.md`).
Phase 0 decomposed spectrogram augmentation on Rung B into crop+masking (maha +0.32,
neg_cos -2.70 -- harm, no benefit) and mixup on top (maha +2.51, neg_cos -1.45). Carrying
the masking/crop half into Rung G would import a known negative.

Fbank cache
-----------
Recomputing kaldi fbanks every epoch is the bottleneck (CPU-bound, ~31k clips/epoch), so
they are computed ONCE into a float16 memmap under --cache_dir and reused by every seed.
~8 GB total. Delete the cache dir to force a rebuild.

Usage
-----
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/anaconda3/envs/mamba-ad/lib
    CUDA_VISIBLE_DEVICES=0 python diagnostics/section_ast_rungG.py --epochs 30 --seed 0

Outputs (runs/section_rungG/<tag>_seed<N>/): metric_curve.csv, best_summary.csv,
train_log.csv, id_breakdown.csv, aug_config.json -- same schema as Rung B, so
`docs/plot_phase0_aug.py --arms ...` and the ladder plotters read them unchanged.
"""

import os
import re
import sys
import glob
import json
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from sklearn.metrics import roc_auc_score
import tabulate

from diagnostics.frozen_encoder_probe import fit_mahalanobis, maha_score
from diagnostics.section_classifier_probe import ArcMarginProduct
from diagnostics.section_finetune_rungB import CLASSES, score_epoch, id_level_auroc
from diagnostics.spec_augment import mixup_batch, mixup_arcface_loss

SR = 16000
CLIP_LEN = SR * 10
AST_CKPT = 'MIT/ast-finetuned-audioset-10-10-0.4593'
_ID_RE = re.compile(r'id_(\d+)')


# --------------------------------------------------------------------------- #
# data: wav list -> cached AST fbanks
# --------------------------------------------------------------------------- #
def list_items(data_root, classes):
    """-> [(path, cls, section, anomaly, split)]. Train dirs are normal-only on disk
    (verified), but the anomaly flag is parsed from the filename regardless so an
    unexpected file cannot silently leak into the normal-only train split."""
    items = []
    for cls in classes:
        for split in ('train', 'test'):
            d = os.path.join(data_root, f'data_{cls}', cls, split)
            for p in sorted(glob.glob(os.path.join(d, '*.wav'))):
                base = os.path.basename(p)
                anom = 1 if base.startswith('anomaly') else 0
                m = _ID_RE.search(base)
                sec = f'{cls}/id_{m.group(1) if m else "??"}'
                if split == 'train' and anom:
                    continue                      # UAD: train stays normal-only
                items.append((p, cls, sec, anom, split))
    return items


def load_wav(path):
    import soundfile as sf
    w, sr = sf.read(path, dtype='float32')
    if w.ndim > 1:
        w = w.mean(axis=1)
    if sr != SR:
        import librosa
        w = librosa.resample(w, orig_sr=sr, target_sr=SR)
    return w[:CLIP_LEN] if len(w) >= CLIP_LEN else np.pad(w, (0, CLIP_LEN - len(w)))


_W = {}      # per-worker state: feature extractor + the memmap opened read-write


def _cache_init(feat_path, shape):
    # Pin each worker to ONE thread. torchaudio's kaldi fbank goes through torch, which
    # defaults to one intra-op thread per core -- with N worker processes that is N*cores
    # threads fighting over the same cores. Unpinned, 12 workers ran at exactly the serial
    # rate (3.7 clips/s) and drove load average past 380.
    torch.set_num_threads(1)
    from transformers import ASTFeatureExtractor
    _W['fe'] = ASTFeatureExtractor.from_pretrained(AST_CKPT)
    _W['mm'] = np.memmap(feat_path, dtype=np.float16, mode='r+', shape=shape)


def _cache_chunk(job):
    """Fill rows [lo, hi) of the memmap. Workers write disjoint slices of one file."""
    lo, paths = job
    wavs = [load_wav(p) for p in paths]
    out = _W['fe'](wavs, sampling_rate=SR, return_tensors='np')['input_values']
    _W['mm'][lo:lo + len(paths)] = out.astype(np.float16)
    return len(paths)


def build_cache(items, cache_dir, batch=32, procs=12):
    """Compute AST fbanks once into a float16 memmap. Returns (memmap, shape).

    Keyed by a manifest of the exact file list, so a changed split forces a rebuild
    instead of silently reusing stale features. Kaldi fbank is CPU-bound and perfectly
    parallel; serially this is ~2.7 h for 31k clips, which would dominate the phase.
    """
    import multiprocessing as mp
    # belt-and-braces: the env vars must be set before the workers import their BLAS
    for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ.setdefault(v, '1')
    from transformers import ASTFeatureExtractor
    os.makedirs(cache_dir, exist_ok=True)
    feat_path = os.path.join(cache_dir, 'fbank_f16.npy')
    man_path = os.path.join(cache_dir, 'manifest.json')
    fe = ASTFeatureExtractor.from_pretrained(AST_CKPT)
    shape = (len(items), fe.max_length, fe.num_mel_bins)
    manifest = dict(n=len(items), max_length=fe.max_length, num_mel_bins=fe.num_mel_bins,
                    ckpt=AST_CKPT, first=items[0][0], last=items[-1][0])

    if os.path.isfile(feat_path) and os.path.isfile(man_path):
        if json.load(open(man_path)) == manifest:
            print(f'[cache] reusing {feat_path}  {shape}')
            return np.memmap(feat_path, dtype=np.float16, mode='r', shape=shape), shape
        print('[cache] manifest mismatch -> rebuilding')

    print(f'[cache] building {feat_path}  {shape}  '
          f'(~{np.prod(shape) * 2 / 1e9:.1f} GB, {procs} procs)', flush=True)
    np.memmap(feat_path, dtype=np.float16, mode='w+', shape=shape).flush()  # allocate
    jobs = [(i, [p for p, *_ in items[i:i + batch]]) for i in range(0, len(items), batch)]
    done = 0
    with mp.Pool(procs, initializer=_cache_init, initargs=(feat_path, shape)) as pool:
        for k in pool.imap_unordered(_cache_chunk, jobs):
            done += k
            print(f'\r  {done}/{len(items)}', end='', flush=True)
    print()
    # the manifest is written only after every chunk lands, so an interrupted build
    # is never mistaken for a complete one on the next run
    json.dump(manifest, open(man_path, 'w'))
    return np.memmap(feat_path, dtype=np.float16, mode='r', shape=shape), shape


class FbankSet(torch.utils.data.Dataset):
    def __init__(self, mm, idx, sec_ids, cls_names, anoms):
        self.mm, self.idx = mm, idx
        self.sec_ids, self.cls_names, self.anoms = sec_ids, cls_names, anoms

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        return (torch.from_numpy(np.asarray(self.mm[j], dtype=np.float32)),
                self.sec_ids[j], j)


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
class ResNetOnFbankNet(nn.Module):
    """CONTROL for the PNG-vs-wav confound.

    Rung B read 8-bit PNGs; Rung G reads raw wavs. So a Rung G gain could be the encoder
    (AST) OR just the removal of PNG quantization / the resize to 256x256. This model is
    ResNet34 -- Rung B's exact encoder and head -- fed the SAME cached AST fbanks that
    Rung G sees. Rung G minus this control is attributable to the encoder alone; this
    control minus Rung B is attributable to the input pipeline.

    The fbank is 1-channel (1024 frames x 128 mels), replicated to 3 channels. It is
    normalized with AudioSet statistics rather than ImageNet ones, which is the point --
    the input is held identical to Rung G's.
    """

    def __init__(self, n_classes, embed_dim=128, sub=2, s=30.0, m=0.5,
                 ckpt='model/pretrain/resnet34-43635321.pth'):
        super().__init__()
        import timm
        from timm.models._helpers import load_checkpoint
        self.backbone = timm.create_model('resnet34', pretrained=False,
                                          features_only=True, out_indices=[1, 2, 3])
        if os.path.isfile(ckpt):
            load_checkpoint(self.backbone, ckpt, strict=False)
            print(f'[backbone] loaded ImageNet weights: {ckpt}')
        else:
            print(f'[backbone] WARNING: {ckpt} missing; ImageNet init absent.')
        d = sum(self.backbone.feature_info.channels())
        self.in_bn = nn.BatchNorm1d(d)
        self.proj = nn.Linear(d, embed_dim)
        self.emb_bn = nn.BatchNorm1d(embed_dim)
        self.arc = ArcMarginProduct(embed_dim, n_classes, s=s, m=m, sub=sub)

    def embed(self, x):
        x = x.unsqueeze(1).repeat(1, 3, 1, 1)        # (B,1024,128) -> (B,3,1024,128)
        feats = self.backbone(x)
        gap = torch.cat([f.mean(dim=(2, 3)) for f in feats], dim=1)
        return self.emb_bn(self.proj(self.in_bn(gap)))

    def logits(self, emb, label=None):
        return self.arc(emb, label) if label is not None else self.arc.s * self.arc.cosine(emb)


class ASTSectionNet(nn.Module):
    """Trainable AST -> mean patch tokens -> Linear -> BN -> ArcFace. Mirrors Rung B's
    SectionNet, minus the multi-stage GAP concat (AST has no spatial pyramid)."""

    def __init__(self, n_classes, embed_dim=128, sub=2, s=30.0, m=0.5):
        super().__init__()
        from transformers import ASTModel
        self.ast = ASTModel.from_pretrained(AST_CKPT)
        d = self.ast.config.hidden_size
        self.in_bn = nn.BatchNorm1d(d)
        self.proj = nn.Linear(d, embed_dim)
        self.emb_bn = nn.BatchNorm1d(embed_dim)
        self.arc = ArcMarginProduct(embed_dim, n_classes, s=s, m=m, sub=sub)

    def embed(self, x):
        lhs = self.ast(x).last_hidden_state      # (B, 2+P, D): [CLS, dist, patches...]
        pooled = lhs[:, 2:, :].mean(dim=1)       # ast_meanpatch -- the frozen probe's best tap
        return self.emb_bn(self.proj(self.in_bn(pooled)))

    def logits(self, emb, label=None):
        return self.arc(emb, label) if label is not None else self.arc.s * self.arc.cosine(emb)


def param_groups(model, lr_ast, lr_head, llrd):
    """Layer-wise lr decay over the transformer blocks: a ViT fine-tuned on ~20k clips
    wants the early blocks moving far slower than the head, or it forgets AudioSet."""
    if not hasattr(model, 'ast'):        # ResNet control: Rung B's two-group split
        bb = [p for p in model.backbone.parameters() if p.requires_grad]
        head = [p for n, p in model.named_parameters()
                if p.requires_grad and not n.startswith('backbone.')]
        return [{'params': bb, 'lr': lr_ast}, {'params': head, 'lr': lr_head}]
    blocks = model.ast.encoder.layer
    n = len(blocks)
    groups, seen = [], set()

    def take(mod, lr):
        ps = [p for p in mod.parameters() if p.requires_grad and id(p) not in seen]
        seen.update(id(p) for p in ps)
        if ps:
            groups.append({'params': ps, 'lr': lr})

    take(model.ast.embeddings, lr_ast * (llrd ** (n + 1)))
    for i, blk in enumerate(blocks):
        take(blk, lr_ast * (llrd ** (n - i)))     # deepest block gets the full lr_ast
    take(model.ast, lr_ast)                       # final layernorm / leftovers
    for mod in (model.in_bn, model.proj, model.emb_bn, model.arc):
        take(mod, lr_head)
    return groups


# --------------------------------------------------------------------------- #
@torch.no_grad()
def collect(model, loader, device, meta, need_anom, amp_dtype):
    """-> (E, L, cls, sec, anom) in the exact tuple layout Rung B's score_epoch expects."""
    model.eval()
    embs, logs, order = [], [], []
    for x, _, j in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=amp_dtype):
            e = model.embed(x)
            l = model.logits(e)
        embs.append(e.float().cpu().numpy())
        logs.append(l.float().cpu().numpy())
        order.append(j.numpy())
    order = np.concatenate(order)
    E = np.concatenate(embs).astype(np.float32)
    L = np.concatenate(logs).astype(np.float32)
    cls = np.array([meta['cls'][k] for k in order])
    sec = np.array([meta['sec'][k] for k in order])
    anom = np.array([meta['anom'][k] for k in order], dtype=int) if need_anom else None
    return E, L, cls, sec, anom


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data_root', default='data/dcase-2020')
    ap.add_argument('--cache_dir', default='runs/ast_fbank_cache')
    ap.add_argument('--batch', type=int, default=12)
    ap.add_argument('--batch_eval', type=int, default=32)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--embed_dim', type=int, default=128)
    ap.add_argument('--sub', type=int, default=2)
    ap.add_argument('--arc_s', type=float, default=30.0)
    ap.add_argument('--arc_m', type=float, default=0.5)
    # NOTE: lr_ast is the DEEPEST block's lr; block i gets lr_ast * llrd**(n-i). With
    # 1e-5/0.75 the bottom blocks land at ~3e-7, i.e. effectively frozen -- which would
    # gut the experiment, since Rung B's whole finding is that UNfreezing the encoder is
    # the lever. 5e-5/0.85 keeps block 0 at ~7e-6, actually trainable. Probed, not guessed.
    ap.add_argument('--lr_ast', type=float, default=5e-5, help='ViT fine-tune lr (deepest block)')
    ap.add_argument('--lr_head', type=float, default=1e-3)
    ap.add_argument('--llrd', type=float, default=0.85, help='layer-wise lr decay factor')
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--clip', type=float, default=5.0, help='grad-norm clip (0 disables)')
    ap.add_argument('--mixup', type=float, default=0.2,
                    help='Phase 0 verdict: mixup is the useful half of augmentation; the '
                         'crop/mask half is a net negative and is deliberately absent')
    ap.add_argument('--eval_every', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--backbone', default='ast', choices=['ast', 'resnet34'],
                    help="'resnet34' is the PNG-vs-wav control: Rung B's encoder on the "
                         'SAME fbanks AST sees, so Rung G minus this isolates the encoder')
    ap.add_argument('--tag', default='ast')
    ap.add_argument('--out_dir', default=None)
    ap.add_argument('--build_cache_only', action='store_true')
    ap.add_argument('--cache_procs', type=int, default=12,
                    help='parallel fbank workers (shared machine: do not take all cores)')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    items = list_items(args.data_root, CLASSES)
    sections = sorted({s for _, _, s, _, _ in items})
    sec2idx = {s: i for i, s in enumerate(sections)}
    tr_idx = np.array([i for i, it in enumerate(items) if it[4] == 'train'])
    te_idx = np.array([i for i, it in enumerate(items) if it[4] == 'test'])
    print(f'[data] {len(items)} clips | train(normal) {len(tr_idx)} | test {len(te_idx)} | '
          f'{len(sections)} sections')

    mm, _ = build_cache(items, args.cache_dir, procs=args.cache_procs)
    if args.build_cache_only:
        print('[done] cache only')
        return

    meta = dict(cls={i: it[1] for i, it in enumerate(items)},
                sec={i: it[2] for i, it in enumerate(items)},
                anom={i: it[3] for i, it in enumerate(items)})
    sec_ids = np.array([sec2idx[it[2]] for it in items], dtype=np.int64)
    cls_names = np.array([it[1] for it in items])
    anoms = np.array([it[3] for it in items], dtype=np.int64)

    mk = lambda idx, bs, sh: torch.utils.data.DataLoader(
        FbankSet(mm, idx, sec_ids, cls_names, anoms), batch_size=bs, shuffle=sh,
        num_workers=args.workers, pin_memory=True, drop_last=False)
    train_loader = mk(tr_idx, args.batch, True)
    train_eval_loader = mk(tr_idx, args.batch_eval, False)
    test_loader = mk(te_idx, args.batch_eval, False)

    if args.backbone == 'ast':
        model = ASTSectionNet(len(sections), args.embed_dim, args.sub,
                              args.arc_s, args.arc_m).to(device)
    else:
        model = ResNetOnFbankNet(len(sections), args.embed_dim, args.sub,
                                 args.arc_s, args.arc_m).to(device)
    opt = torch.optim.AdamW(param_groups(model, args.lr_ast, args.lr_head, args.llrd),
                            weight_decay=args.wd)
    ce = nn.CrossEntropyLoss()

    out_dir = args.out_dir or os.path.join('runs', 'section_rungG', f'{args.tag}_seed{args.seed}')
    os.makedirs(out_dir, exist_ok=True)
    print(f'[train] backbone={args.backbone.upper()} + '
          f'ArcFace(s={args.arc_s},m={args.arc_m},sub={args.sub})  '
          f'epochs={args.epochs} lr_bb={args.lr_ast} llrd={args.llrd} lr_head={args.lr_head} '
          f'amp={amp_dtype}'
          + ('   [PNG-vs-wav CONTROL: Rung B encoder on Rung G input]'
             if args.backbone != 'ast' else ''))
    print(f'[aug]   mixup={args.mixup}  (crop/masking deliberately OFF -- Phase 0)')
    json.dump(dict(mixup=args.mixup, crop_mask='off (Phase 0 negative)', seed=args.seed,
                   lr_ast=args.lr_ast, llrd=args.llrd, tap='ast_meanpatch'),
              open(os.path.join(out_dir, 'aug_config.json'), 'w'), indent=2)

    curve_path = os.path.join(out_dir, 'metric_curve.csv')
    train_log_path = os.path.join(out_dir, 'train_log.csv')
    id_path = os.path.join(out_dir, 'id_breakdown.csv')
    open(curve_path, 'w').write('epoch,readout,' + ','.join(CLASSES) + ',mean\n')
    open(train_log_path, 'w').write('epoch,train_loss,train_acc\n')
    open(id_path, 'w').write('epoch,section,n,neg_cos,maha_embed\n')

    best = defaultdict(lambda: (-1.0, -1, None))
    for ep in range(1, args.epochs + 1):
        model.train()
        tot, correct, loss_sum, skipped = 0, 0, 0.0, 0
        for x, y, _ in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if x.size(0) == 1:
                continue                                  # BatchNorm needs >1 sample
            x, perm, lam = mixup_batch(x, args.mixup)
            y_b = y[perm] if perm is not None else None
            with torch.autocast('cuda', dtype=amp_dtype):
                emb = model.embed(x)
                loss = mixup_arcface_loss(model, emb, y, y_b, lam, ce)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            # Non-finite guard, mirroring trainer/_base_trainer.py. Without it the
            # ResNet control diverged at epoch 15 (loss->nan, train acc 98->72) and the
            # run died much later inside the Mahalanobis fit with "SVD did not converge",
            # losing every epoch of work. Skip the bad step instead.
            if not (torch.isfinite(loss) and torch.isfinite(gnorm)):
                skipped += 1
                opt.zero_grad(set_to_none=True)
                continue
            opt.step()
            with torch.no_grad():
                logits = model.logits(emb.detach().float())
            loss_sum += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            tot += x.size(0)
        tr_loss, tr_acc = loss_sum / max(tot, 1), 100.0 * correct / max(tot, 1)
        open(train_log_path, 'a').write(f'{ep},{tr_loss:.6f},{tr_acc:.4f}\n')

        if ep % args.eval_every != 0 and ep != args.epochs:
            print(f'  epoch {ep:3d}/{args.epochs}  loss={tr_loss:.4f} acc={tr_acc:.2f}%')
            continue
        test_pack = collect(model, test_loader, device, meta, True, amp_dtype)
        Etr, _, cls_tr, _, _ = collect(model, train_eval_loader, device, meta, False, amp_dtype)
        # A non-finite embedding makes fit_mahalanobis raise "SVD did not converge",
        # which killed the whole run. Degrade to the finite readouts and keep going --
        # a diverged epoch should cost one row, not every epoch already computed.
        finite = np.isfinite(Etr).all() and np.isfinite(test_pack[0]).all()
        if not finite:
            print(f'  [warn] epoch {ep}: non-finite embeddings -- skipping maha this epoch')
        train_pack = (Etr, cls_tr) if finite else None
        res = score_epoch(model, test_pack, sec2idx, model.arc.s, train_pack, finite)
        id_res = id_level_auroc(test_pack, sec2idx, model.arc.s, train_pack, finite)

        with open(curve_path, 'a') as f:
            for r, d in res.items():
                f.write(f'{ep},{r},' + ','.join(f'{d.get(c, float("nan")):.4f}' for c in CLASSES)
                        + f',{d["mean"]:.4f}\n')
        with open(id_path, 'a') as f:
            for sec, row in id_res.items():
                f.write(f'{ep},{sec},{row["n"]},{row.get("neg_cos", float("nan")):.4f},'
                        f'{row.get("maha_embed", float("nan")):.4f}\n')
        for r, d in res.items():
            if d['mean'] > best[r][0]:
                best[r] = (d['mean'], ep, {c: d.get(c) for c in CLASSES})
        msg = '  '.join(f'{r}={res[r]["mean"] * 100:.2f}' for r in
                        ('neg_cos', 'logit_nll', 'maha_embed') if r in res)
        print(f'  epoch {ep:3d}/{args.epochs}  loss={tr_loss:.4f} acc={tr_acc:.2f}%  {msg}'
              + (f'  [skipped {skipped} non-finite steps]' if skipped else ''), flush=True)

    rows, header = [], ['readout', 'best_epoch'] + CLASSES + ['mean']
    for r in ('neg_cos', 'logit_nll', 'maha_embed'):
        if best[r][1] < 0:
            continue
        mean, epb, pc = best[r]
        rows.append([r, epb] + [f'{pc[c] * 100:.2f}' if pc[c] is not None else '  -  '
                                for c in CLASSES] + [f'{mean * 100:.2f}'])
    print('\n=== Rung G: fine-tuned AST + section classifier (best-epoch AUROC %) ===')
    print(tabulate.tabulate(rows, headers=header, tablefmt='github'))
    with open(os.path.join(out_dir, 'best_summary.csv'), 'w') as f:
        f.write(','.join(header) + '\n')
        for row in rows:
            f.write(','.join(str(x).strip() for x in row) + '\n')
    bm, bep, _ = max(best.values(), key=lambda t: t[0])
    print(f'\nRung G best = {bm * 100:.2f}% @ epoch {bep}')
    print(f'  vs Rung B  85.89% (3-seed)  ->  {bm * 100 - 85.89:+.2f}  (the encoder swap)')
    print(f'  vs STgram  90.75%           ->  {bm * 100 - 90.75:+.2f}  (remaining gap)')


if __name__ == '__main__':
    main()
