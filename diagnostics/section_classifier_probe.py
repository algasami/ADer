"""
Section-classifier probe — Rung A of the "labels close the gap?" ablation ladder
================================================================================

The question
------------
Your gap decomposition (`docs/ABLATION_SUMMARY.md`) blames the dominant -14 AUROC
gap to STgram-MFN on the *learning objective*, not the encoder or the input. This
probe is the **first, cleanest** test of that claim:

    Keep the encoder frozen. Change ONLY the objective — train a small head to
    classify the free **machine-section metadata** (type x id, 23 sections in
    MIMII/DCASE-2020) with the same ArcFace loss STgram-MFN uses. Then read out an
    anomaly score the STgram way. Does a *discriminative* objective, applied to the
    very same frozen ResNet34 features, separate normal from anomalous better than
    the generative cosine-residual / Mahalanobis readouts?

This is "Rung A" in the ladder: head-only, encoder frozen, so it isolates the
**objective + classification readout** with *zero* representation change. Read its
result against two anchors you already have:

  * frozen ResNet34 + Mahalanobis  (this script recomputes it as `maha_concat_raw`
    in the same run, and you can pass your logged number via --frozen_maha_auroc)
  * STgram-MFN                       (--stgram_auroc)

Interpreting the deltas (mean image-level AUROC across the 6 machine classes):

    maha_concat_raw            = frozen features + generative readout (the anchor)
    maha_embed                 = frozen features + discriminative-trained embedding
                                 + generative readout
    logit_nll / neg_cos        = frozen features + discriminative readout (STgram-style)

  maha_embed  > maha_concat_raw   -> the discriminative objective reshapes even a
                                     *frozen* feature space into something more
                                     separable. Objective is a real lever; Rung B
                                     (fine-tune the encoder) should push further.
  logit_nll  >> maha_*            -> it is specifically the *classification readout*
                                     (fit-to-your-own-section) that carries the
                                     signal — the STgram scoring rule matters.
  all ~= maha_concat_raw          -> head-only is not enough; the lever needs the
                                     encoder to move (go straight to Rung B), or the
                                     gap is not the objective after all (a finding).

Design notes
------------
* Encoder: the *exact* frozen teacher from the config (timm resnet34, taps
  layer1/2/3, ImageNet norm) — reuses `frozen_encoder_probe.build_encoder`, so its
  raw-Maha anchor is byte-comparable to that script.
* Features are frozen, so we extract concat-GAP vectors ONCE and train the head on
  the cached vectors for many cheap epochs (no encoder forward in the loop).
* Head = standardize -> Linear(D->embed) -> BN -> ArcFace(embed, 23, sub-cluster).
  Ported from `STgram-MFN/net.py`. `--no_arcface` swaps in a plain linear+CE head
  (the STgram `ASDLoss` floor).
* Sections come from `img_path` (`id_XX`), so NO change to the shared dataset
  loader is needed — Rung A stays a self-contained diagnostic.
* Readouts are image-level -> AUROC computed per machine class then averaged,
  matching every number in the ablation summary.

Usage
-----
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/anaconda3/envs/mamba-ad/lib
    CUDA_VISIBLE_DEVICES=0 python diagnostics/section_classifier_probe.py \
        -c MambaAD/configs/mambaad/mimii/e50/log-Mel.py \
        --frozen_maha_auroc 0.79 --stgram_auroc 0.86 --plot

Outputs:
    runs/section_probe/<cfg_name>/auroc.csv        (per-class + mean, every readout)
    runs/section_probe/<cfg_name>/train_log.csv    (head loss/acc per epoch)
    runs/section_probe/<cfg_name>/scores_<cls>.png (optional, --plot)
"""

import os
import re
import sys
import math
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# make repo root importable regardless of CWD
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from sklearn.metrics import roc_auc_score, average_precision_score
import tabulate

from data import get_loader
# reuse the frozen-probe plumbing so the encoder + raw-Maha anchor stay identical
from diagnostics.frozen_encoder_probe import (
    build_cfg, build_encoder, extract, gap_vectors, fit_mahalanobis, maha_score,
)

_ID_RE = re.compile(r'id_(\d+)')


# --------------------------------------------------------------------------- #
# section labels (parsed from img_path; no dataset change)
# --------------------------------------------------------------------------- #
def parse_section(cls_name, img_path):
    """Global section key = '<machine_type>/id_<XX>' parsed from the filename.
    Falls back to '<cls>/id_??' if no id_ token is present (should not happen for MIMII)."""
    m = _ID_RE.search(os.path.basename(img_path))
    sec_id = m.group(1) if m else '??'
    return f'{cls_name}/id_{sec_id}'


# --------------------------------------------------------------------------- #
# ArcFace head (ported from STgram-MFN/net.py, sub-cluster capable)
# --------------------------------------------------------------------------- #
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, sub=1, easy_margin=False):
        super().__init__()
        self.out_features = out_features
        self.s, self.m, self.sub = s, m, sub
        self.weight = nn.Parameter(torch.Tensor(out_features * sub, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m, self.sin_m = math.cos(m), math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def cosine(self, x):
        """Margin-free cosine logits (test-time): (N, out_features), sub-cluster max-pooled."""
        cos = F.linear(F.normalize(x), F.normalize(self.weight))
        if self.sub > 1:
            cos = cos.view(-1, self.out_features, self.sub).max(dim=2).values
        return cos

    def forward(self, x, label):
        cos = self.cosine(x)
        sine = torch.sqrt(torch.clamp(1.0 - cos ** 2, min=0.0))
        phi = cos * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cos > 0, phi, cos)
        else:
            phi = torch.where((cos - self.th) > 0, phi, cos - self.mm)
        one_hot = torch.zeros_like(cos)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        out = (one_hot * phi) + ((1.0 - one_hot) * cos)
        return out * self.s


class SectionHead(nn.Module):
    """Frozen-feature classification head: standardize -> Linear -> BN -> embedding,
    then ArcFace (or a plain linear classifier). Only this module is trained."""
    def __init__(self, in_dim, n_classes, embed_dim=128, use_arcface=True,
                 s=30.0, m=0.5, sub=1, feat_mean=None, feat_std=None):
        super().__init__()
        self.use_arcface = use_arcface
        # fixed input standardization (fit on normal-train features)
        self.register_buffer('feat_mean', feat_mean if feat_mean is not None else torch.zeros(in_dim))
        self.register_buffer('feat_std', feat_std if feat_std is not None else torch.ones(in_dim))
        self.proj = nn.Linear(in_dim, embed_dim)
        self.bn = nn.BatchNorm1d(embed_dim)
        if use_arcface:
            self.arc = ArcMarginProduct(embed_dim, n_classes, s=s, m=m, sub=sub)
        else:
            self.fc = nn.Linear(embed_dim, n_classes)

    def embed(self, x):
        x = (x - self.feat_mean) / self.feat_std
        x = self.bn(self.proj(x))
        return x

    def logits(self, emb, label=None):
        """Training logits (ArcFace needs the label); test logits are margin-free."""
        if self.use_arcface:
            return self.arc(emb, label) if label is not None else self.arc.s * self.arc.cosine(emb)
        return self.fc(emb)

    def forward(self, x, label=None):
        return self.logits(self.embed(x), label)


# --------------------------------------------------------------------------- #
# feature extraction (concat GAP over taps) with section + class + anomaly
# --------------------------------------------------------------------------- #
@torch.no_grad()
def extract_split(model, loader, device, need_anom):
    """Return X (N,D) concat-GAP, cls (N,) str, sec (N,) str section-key,
    anom (N,) int or None. One frozen forward per batch."""
    Xs, clss, secs, anoms = [], [], [], []
    for bi, batch in enumerate(loader):
        imgs = batch['img'].to(device, non_blocking=True)
        cls = np.array(batch['cls_name'])
        paths = batch['img_path']
        feats = extract(model, imgs)
        _, concat = gap_vectors(feats)
        Xs.append(concat.cpu().numpy())
        clss.append(cls)
        secs.append(np.array([parse_section(c, p) for c, p in zip(cls, paths)]))
        if need_anom:
            a = batch['anomaly']
            anoms.append(a.numpy() if torch.is_tensor(a) else np.array(a))
        print(f"\r  batch {bi + 1}", end='')
    print()
    X = np.concatenate(Xs, axis=0).astype(np.float32)
    cls = np.concatenate(clss)
    sec = np.concatenate(secs)
    anom = np.concatenate(anoms).astype(int) if need_anom else None
    return X, cls, sec, anom


# --------------------------------------------------------------------------- #
# head training on cached frozen vectors
# --------------------------------------------------------------------------- #
def train_head(head, X, y, device, epochs, lr, wd, batch, log_path):
    head = head.to(device).train()
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=wd)
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y).long()
    n = len(Xt)
    ce = nn.CrossEntropyLoss()
    rows = []
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot, correct, loss_sum = 0, 0, 0.0
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            xb = Xt[idx].to(device)
            yb = yt[idx].to(device)
            logits = head(xb, yb)
            loss = ce(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(idx)
            correct += (logits.argmax(1) == yb).sum().item()
            tot += len(idx)
        rows.append((ep + 1, loss_sum / tot, 100.0 * correct / tot))
        if (ep + 1) % max(1, epochs // 10) == 0 or ep == 0:
            print(f"  [head] epoch {ep + 1:4d}/{epochs}  loss={loss_sum / tot:.4f}  "
                  f"train_acc={100.0 * correct / tot:.2f}%")
    with open(log_path, 'w') as f:
        f.write("epoch,loss,train_acc\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]:.6f},{r[2]:.4f}\n")
    head.eval()
    return head


@torch.no_grad()
def head_embeddings(head, X, device, batch=1024):
    embs = []
    Xt = torch.from_numpy(X)
    for s in range(0, len(Xt), batch):
        embs.append(head.embed(Xt[s:s + batch].to(device)).cpu().numpy())
    return np.concatenate(embs, axis=0).astype(np.float32)


@torch.no_grad()
def head_logits(head, X, device, batch=1024):
    """Margin-free classification logits (test-time)."""
    out = []
    Xt = torch.from_numpy(X)
    for s in range(0, len(Xt), batch):
        out.append(head(Xt[s:s + batch].to(device)).cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-c', '--cfg_path', required=True,
                    help='ADer config (e.g. MambaAD/configs/mambaad/mimii/e50/log-Mel.py)')
    ap.add_argument('--batch', type=int, default=32, help='encoder forward batch')
    ap.add_argument('--workers', type=int, default=8)
    # head / objective
    ap.add_argument('--embed_dim', type=int, default=128)
    ap.add_argument('--no_arcface', action='store_true',
                    help='use a plain linear+CE head (STgram ASDLoss floor) instead of ArcFace')
    ap.add_argument('--arc_s', type=float, default=30.0)
    ap.add_argument('--arc_m', type=float, default=0.5)
    ap.add_argument('--sub', type=int, default=1, help='ArcFace sub-clusters per section')
    ap.add_argument('--head_epochs', type=int, default=100)
    ap.add_argument('--head_lr', type=float, default=1e-3)
    ap.add_argument('--head_wd', type=float, default=1e-4)
    ap.add_argument('--head_batch', type=int, default=256)
    # readout
    ap.add_argument('--maha_scope', choices=['class', 'section'], default='class',
                    help="fit the maha_embed bank per machine class (matches frozen probe) "
                         "or per section (tighter, one bank per id)")
    # anchors for the verdict
    ap.add_argument('--frozen_maha_auroc', type=float, default=None,
                    help='your logged frozen ResNet34 + Maha mean AUROC (0-1) anchor')
    ap.add_argument('--stgram_auroc', type=float, default=None,
                    help='STgram-MFN mean AUROC (0-1) target')
    ap.add_argument('--plot', action='store_true')
    ap.add_argument('--out_dir', default=None)
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cfg = build_cfg(args.cfg_path, args.batch, args.workers)
    cfg_name = os.path.splitext(os.path.basename(args.cfg_path))[0]
    out_dir = args.out_dir or os.path.join('runs', 'section_probe', cfg_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[cfg] {args.cfg_path}  data.root={cfg.data.root}")

    model, _ = build_encoder(cfg, device)
    train_loader, test_loader = get_loader(cfg)

    # ----- pass 1: frozen features on normal-only train split -----
    print("[pass 1] extracting normal-train frozen features ...")
    Xtr, cls_tr, sec_tr, _ = extract_split(model, train_loader, device, need_anom=False)

    # global section label map (sorted for determinism)
    sections = sorted(set(sec_tr.tolist()))
    sec2idx = {s: i for i, s in enumerate(sections)}
    y_tr = np.array([sec2idx[s] for s in sec_tr], dtype=np.int64)
    print(f"  {len(Xtr)} normal-train clips, D={Xtr.shape[1]}, {len(sections)} sections:")
    for s in sections:
        print(f"    {s:20s} n={int((sec_tr == s).sum())}")

    # ----- train the head on cached frozen vectors -----
    feat_mean = torch.from_numpy(Xtr.mean(0))
    feat_std = torch.from_numpy(Xtr.std(0) + 1e-6)
    head = SectionHead(
        in_dim=Xtr.shape[1], n_classes=len(sections), embed_dim=args.embed_dim,
        use_arcface=not args.no_arcface, s=args.arc_s, m=args.arc_m, sub=args.sub,
        feat_mean=feat_mean, feat_std=feat_std,
    )
    obj = 'plain-CE' if args.no_arcface else f'ArcFace(s={args.arc_s},m={args.arc_m},sub={args.sub})'
    print(f"[head] objective={obj}  embed_dim={args.embed_dim}  "
          f"epochs={args.head_epochs} lr={args.head_lr} wd={args.head_wd}")
    head = train_head(head, Xtr, y_tr, device, args.head_epochs, args.head_lr,
                      args.head_wd, args.head_batch, os.path.join(out_dir, 'train_log.csv'))

    # ----- fit generative banks on normal train: raw-concat + learned embedding -----
    Etr = head_embeddings(head, Xtr, device)
    classes = sorted(set(cls_tr.tolist()))
    maha_raw, maha_emb = {}, {}
    for c in classes:
        m = cls_tr == c
        maha_raw[c] = fit_mahalanobis(Xtr[m])          # frozen-feature anchor, per class
    if args.maha_scope == 'class':
        for c in classes:
            maha_emb[c] = fit_mahalanobis(Etr[cls_tr == c])
    else:  # per-section banks, keyed by section string
        for s in sections:
            maha_emb[s] = fit_mahalanobis(Etr[sec_tr == s])

    # ----- pass 2: frozen features on test split (normal + abnormal) -----
    print("[pass 2] extracting + scoring test features ...")
    Xte, cls_te, sec_te, y_anom = extract_split(model, test_loader, device, need_anom=True)
    Ete = head_embeddings(head, Xte, device)
    Lte = head_logits(head, Xte, device)               # (N, n_sections) margin-free logits
    logp = Lte - torch.logsumexp(torch.from_numpy(Lte), dim=1, keepdim=True).numpy()  # log-softmax
    # assigned-section index for each test clip (metadata is known at test time)
    assigned = np.array([sec2idx.get(s, -1) for s in sec_te])

    # ----- per-class readouts + AUROC -----
    readouts = ['maha_concat_raw', 'maha_embed', 'logit_nll', 'neg_cos']
    auroc = defaultdict(dict)
    ap_ = defaultdict(dict)
    all_scores = defaultdict(dict)
    for c in classes:
        m = cls_te == c
        y = y_anom[m]
        if y.min() == y.max():
            print(f"  [WARN] class {c}: test labels all == {y[0]}, AUROC undefined; skipping")
            continue

        def record(name, s):
            auroc[name][c] = roc_auc_score(y, s)
            ap_[name][c] = average_precision_score(y, s)
            all_scores[name][c] = (s, y)

        # generative anchors (higher dist = more anomalous)
        mu, ic = maha_raw[c]
        record('maha_concat_raw', maha_score(Xte[m], mu, ic))
        if args.maha_scope == 'class':
            mu, ic = maha_emb[c]
            record('maha_embed', maha_score(Ete[m], mu, ic))
        else:
            # per-section bank: score each clip against its own section's bank
            s_emb = np.empty(int(m.sum()), dtype=np.float32)
            idxs = np.where(m)[0]
            for j, gi in enumerate(idxs):
                mu, ic = maha_emb[sec_te[gi]]
                s_emb[j] = maha_score(Ete[gi:gi + 1], mu, ic)[0]
            record('maha_embed', s_emb)

        # discriminative readouts (STgram-style: poor fit to your OWN section = anomalous)
        a = assigned[m]
        record('logit_nll', -logp[m][np.arange(len(a)), a])       # -log p(assigned section)
        # neg cosine to assigned center: recover cosine from margin-free logits (/s)
        s_scale = head.arc.s if not args.no_arcface else 1.0
        record('neg_cos', -(Lte[m][np.arange(len(a)), a] / s_scale))

    # ----- report -----
    header = ['readout'] + classes + ['mean_AUROC', 'mean_AP']
    rows, method_mean = [], {}
    for name in readouts:
        if name not in auroc:
            continue
        row = [name]
        aus, aps = [], []
        for c in classes:
            v = auroc[name].get(c)
            row.append(f'{v * 100:.2f}' if v is not None else '  -  ')
            if v is not None:
                aus.append(v); aps.append(ap_[name][c])
        mean_au = float(np.mean(aus)) if aus else float('nan')
        method_mean[name] = mean_au
        row += [f'{mean_au * 100:.2f}', f'{np.mean(aps) * 100:.2f}' if aps else '  -  ']
        rows.append(row)

    print("\n=== Rung A: frozen ResNet34 + section-classifier head "
          "(image-level AUROC %, higher=better) ===")
    print(tabulate.tabulate(rows, headers=header, tablefmt='github'))

    csv_path = os.path.join(out_dir, 'auroc.csv')
    with open(csv_path, 'w') as f:
        f.write(','.join(header) + '\n')
        for row in rows:
            f.write(','.join(str(x).strip() for x in row) + '\n')
    print(f"\n[out] {csv_path}")
    print(f"[out] {os.path.join(out_dir, 'train_log.csv')}")

    # ----- verdict -----
    anchor = method_mean.get('maha_concat_raw')
    if anchor is not None:
        print(f"\nAnchor  frozen+Maha (this run) : {anchor * 100:.2f}%")
        if args.frozen_maha_auroc is not None:
            print(f"        frozen+Maha (--supplied): {args.frozen_maha_auroc * 100:.2f}%")
        disc_best = max((method_mean[m] for m in ('logit_nll', 'neg_cos') if m in method_mean),
                        default=None)
        emb = method_mean.get('maha_embed')
        if emb is not None:
            d = (emb - anchor) * 100
            print(f"Δ maha_embed  − frozen+Maha : {d:+.2f}  "
                  f"({'objective reshapes the frozen space' if d > 1 else 'no reshape from head alone'})")
        if disc_best is not None:
            d = (disc_best - anchor) * 100
            best_name = max(('logit_nll', 'neg_cos'),
                            key=lambda m: method_mean.get(m, -1))
            print(f"Δ {best_name:10s} − frozen+Maha : {d:+.2f}  "
                  f"({'classification readout carries signal' if d > 1 else 'classification readout no better'})")
        if args.stgram_auroc is not None:
            best = max(method_mean.values())
            gap = (args.stgram_auroc - best) * 100
            print(f"Remaining gap to STgram-MFN ({args.stgram_auroc * 100:.2f}%): {gap:+.2f} "
                  f"(best Rung-A readout = {best * 100:.2f}%)")
            if best >= args.stgram_auroc - 0.01:
                print("VERDICT: head-only classification ALONE closes the gap — the objective was "
                      "the whole story; the encoder never needed to move.")
            elif (method_mean.get('maha_embed', 0) - anchor) > 0.01 or (disc_best or 0) - anchor > 0.01:
                print("VERDICT: head-only helps but does not close it — objective is a real lever; "
                      "run Rung B (fine-tune the encoder under the same loss) to see how much of "
                      "the rest is representation adaptation.")
            else:
                print("VERDICT: head-only does NOT move the needle. Either the lever needs the "
                      "encoder to move (go to Rung B), or the gap is not the objective — a finding.")

    # ----- optional plots -----
    if args.plot and method_mean:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"[plot] matplotlib unavailable ({e}); skipping")
            return
        best_name = max(method_mean, key=method_mean.get)
        for c in classes:
            if c not in all_scores.get(best_name, {}):
                continue
            s, y = all_scores[best_name][c]
            fig, axp = plt.subplots(figsize=(5, 3))
            axp.hist(s[y == 0], bins=40, alpha=0.6, label='normal', density=True)
            axp.hist(s[y == 1], bins=40, alpha=0.6, label='abnormal', density=True)
            axp.set_title(f'{c}  {best_name}  AUROC={auroc[best_name][c] * 100:.1f}%')
            axp.set_xlabel('anomaly score'); axp.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f'scores_{c}.png'), dpi=120)
            plt.close(fig)
        print(f"[out] score histograms -> {out_dir}/scores_*.png ({best_name})")


if __name__ == '__main__':
    main()
