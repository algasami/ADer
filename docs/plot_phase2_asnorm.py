"""
Phase 2 figures — per-section Maha banks, the STgram-MFN comparison, and Rung H
==============================================================================

`docs/plots/phase2_asnorm/` was written as prose only (CONCLUSION.md, RUNG_H.md);
this script regenerates the figures those two documents describe, entirely from the
CSVs the runs already wrote. Nothing here retrains anything.

Sources (all under `runs/section_rungG/`):
  rn34fold_seed{0,1,2}   ResNet34 / fbank 1024x128 / 70 ep  -- the headline 91.15 runs
  baseH2_seed{0,1,2}     Rung H control  (no decoder)       / fbank 512x128 / 30 ep
  rungH2_seed{0,1,2}     Rung H treatment (Mamba student)   / fbank 512x128 / 30 ep
  meanid_folds.csv       mean-of-per-ID AUROC, full test + each half of the fixed split
  asnorm_by_section.csv  per-section AUROC per epoch per readout mode
  train_log.csv          loss / 23-way section accuracy
plus the baseline's own training log, `STgram-MFN/runs/STgram-MFN(m=0.7,s=30)/running.log`,
which evaluates every 10 epochs and is what makes an epoch-vs-epoch comparison possible.

Three figures, one claim each:

  bank_lever.png   The BANK is the lever and AS-norm is worth zero. The class_raw and
                   class_asnorm curves are numerically identical (as are the two section
                   curves) because AUROC within a section is invariant to any strictly
                   increasing per-section transform -- so the visible gap is the bank alone.
  vs_stgram.png    Epoch-for-epoch against STgram-MFN, on STgram's own metric (mean of
                   per-ID AUROC). Both arms' headline numbers are best-epoch-on-test;
                   ours is also shown under held-out selection, which STgram never applies
                   to itself.
  rung_h.png       The Mamba decoder against its true control (identical except the
                   decoder), plus the train-accuracy panel showing Rung H trained cleanly.

Usage
-----
    python docs/plot_phase2_asnorm.py                 # all three
    python docs/plot_phase2_asnorm.py --figs rungh
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CLASSES = ['fan', 'pump', 'slider', 'valve', 'ToyCar', 'ToyConveyor']
STGRAM = 90.75                       # STgram-MFN(m=0.7,s=30), best-epoch mean-of-per-ID
MODES = ['class_raw', 'class_asnorm', 'section_raw', 'section_asnorm']

# Categorical slots taken in fixed order from the validated reference palette
# (blue / orange / aqua / violet). Never cycled, never reordered per figure.
C_BLUE, C_ORANGE, C_AQUA, C_VIOLET = '#2a78d6', '#eb6834', '#1baf7a', '#4a3aa7'
MODE_COLOR = {'class_raw': C_BLUE, 'class_asnorm': C_BLUE,
              'section_raw': C_ORANGE, 'section_asnorm': C_ORANGE}
INK, MUTED = '#0b0b0b', '#52514e'


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_arm(pattern):
    """-> [(name, folds_df, section_df, train_df), ...] sorted by run name."""
    out = []
    for d in sorted(glob.glob(pattern)):
        f = os.path.join(d, 'meanid_folds.csv')
        if not os.path.isdir(d) or not os.path.isfile(f):
            continue
        sec = os.path.join(d, 'asnorm_by_section.csv')
        tr = os.path.join(d, 'train_log.csv')
        out.append((os.path.basename(d), pd.read_csv(f),
                    pd.read_csv(sec) if os.path.isfile(sec) else None,
                    pd.read_csv(tr) if os.path.isfile(tr) else None))
    return out


def selection_rules(folds, mode):
    """The three ways this campaign has quoted a number, for one run and one readout.

    held-out: fold A picks the epoch and fold B scores it, and vice versa -- neither
    reported number took part in choosing itself. Same rule as docs/select_heldout_epoch.py.
    """
    df = folds[folds['mode'] == mode].sort_values('epoch')
    epA = int(df.loc[df.meanid_foldA.idxmax(), 'epoch'])
    epB = int(df.loc[df.meanid_foldB.idxmax(), 'epoch'])
    return dict(
        test_selected=float(df.meanid_full.max()) * 100,
        heldout=0.5 * (float(df.loc[df.epoch == epA, 'meanid_foldB'].iloc[0]) +
                       float(df.loc[df.epoch == epB, 'meanid_foldA'].iloc[0])) * 100,
        final=float(df.meanid_full.iloc[-1]) * 100,
        ep_test=int(df.loc[df.meanid_full.idxmax(), 'epoch']), ep_A=epA, ep_B=epB)


def arm_table(arm, modes=('section_asnorm',)):
    rows = []
    for name, folds, _, _ in arm:
        for m in modes:
            rows.append(dict(run=name, mode=m, **selection_rules(folds, m)))
    return pd.DataFrame(rows)


def curve_band(arm, mode):
    """Seed mean and min/max envelope of the full-test mean-of-ID curve."""
    wide = pd.concat([f[f['mode'] == mode].set_index('epoch').meanid_full.rename(n)
                      for n, f, _, _ in arm], axis=1) * 100
    return wide.index.values, wide.mean(1).values, wide.min(1).values, wide.max(1).values


def parse_stgram_log(path):
    """-> (per-epoch DataFrame, best-model per-class Series).

    The log prints a 6-class + 'Total average' eval block every 10 epochs, then re-evaluates
    the reloaded best checkpoint at the end (blocks with no preceding Epoch- line).
    """
    ep, cur, rows, tail = None, {}, [], []
    for line in open(path):
        m = re.search(r'Epoch-(\d+)\s', line)
        if m:
            ep = int(m.group(1))
            continue
        m = re.search(r'(\w+)\s+AUC:\s*([\d.]+)', line)
        if m and m.group(1) in CLASSES:
            cur[m.group(1)] = float(m.group(2))
            continue
        m = re.search(r'Total average:\s+AUC:\s*([\d.]+)', line)
        if m:
            rec = dict(epoch=ep, mean=float(m.group(1)), **cur)
            (rows if ep is not None else tail).append(rec)
            cur, ep = {}, None            # a fresh block must re-announce its epoch
    df = pd.DataFrame(rows).drop_duplicates('epoch', keep='last').sort_values('epoch')
    best = pd.Series(tail[-1]).drop('epoch') if tail else df.loc[df['mean'].idxmax()]
    return df, best.astype(float)


def parse_stgram_result(path):
    """result.csv -> {'ToyCar/id_01': auroc%, ...} for the 23 sections."""
    out, cls = {}, None
    for line in open(path):
        p = line.strip().split(',')
        if len(p) == 1 and p[0] in CLASSES:
            cls = p[0]
        elif len(p) == 3 and cls and re.fullmatch(r'\d+', p[0]):
            out[f'{cls}/id_{p[0]}'] = float(p[1]) * 100
    return out


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=10.5, color=INK)
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.grid(alpha=.25, lw=.6)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8.5, colors=MUTED)


# --------------------------------------------------------------------------- #
# figure 1 — the bank is the lever, AS-norm is zero
# --------------------------------------------------------------------------- #
def fig_bank(arm, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5),
                             gridspec_kw=dict(width_ratios=[1.15, .85, 1.25]))

    # (a) the four readout curves. class_asnorm sits exactly on class_raw and
    # section_asnorm on section_raw -- drawn dashed on top so the overlap is the point.
    ax = axes[0]
    for mode in MODES:
        ep, mu, lo, hi = curve_band(arm, mode)
        if mode.endswith('asnorm'):      # thin dashes ride exactly on the fat raw line
            ax.plot(ep, mu, color=MODE_COLOR[mode], lw=1.4, dashes=(3, 3), label=mode)
        else:
            ax.fill_between(ep, lo, hi, color=MODE_COLOR[mode], alpha=.15, lw=0)
            ax.plot(ep, mu, color=MODE_COLOR[mode], lw=4, alpha=.45, label=mode)
    ax.axhline(STGRAM, ls=':', c=INK, lw=1.4)
    ax.text(1, STGRAM + .25, f'STgram-MFN {STGRAM}', fontsize=8.5, color=INK)
    ax.legend(fontsize=8.5, frameon=False, loc='lower right')
    style(ax, '(a) per-section bank vs per-class bank\n(3 seeds, band = min-max)',
          'epoch', 'mean-of-per-ID AUROC (%)')
    top, bot = curve_band(arm, 'section_asnorm')[1][44], curve_band(arm, 'class_raw')[1][44]
    ax.annotate('', xy=(45, top), xytext=(45, bot),
                arrowprops=dict(arrowstyle='<->', color=MUTED, lw=1.2))
    ax.text(46.5, (top + bot) / 2, 'bank\nlever', fontsize=8.5, color=MUTED, va='center')
    ax.text(3, 76.5, 'AS-norm curves lie exactly on the raw curves:\nAUROC within a section is '
            'invariant to a\nmonotone per-section rescaling', fontsize=8, color=MUTED)

    # (b) the same two readouts under each selection rule, with the seeds shown
    ax = axes[1]
    tbl = arm_table(arm, modes=('class_raw', 'section_asnorm'))
    rules = [('test_selected', 'test-selected\n(optimistic)'),
             ('heldout', 'held-out\n(HONEST)'), ('final', 'final epoch\n(floor)')]
    w = .36
    for k, (mode, col) in enumerate([('class_raw', C_BLUE), ('section_asnorm', C_ORANGE)]):
        sub = tbl[tbl['mode'] == mode]
        x = np.arange(len(rules)) + (k - .5) * (w + .02)
        vals = [sub[c].mean() for c, _ in rules]
        ax.bar(x, vals, w, color=col, label=mode, zorder=2)
        ax.text(x[0], 87.75, mode, rotation=90, ha='center', va='bottom',
                fontsize=8.5, color='white', zorder=4)   # direct label, no legend box
        for xi, (c, _) in zip(x, rules):
            ax.scatter(np.full(len(sub), xi), sub[c], s=14, color='white',
                       edgecolor=MUTED, lw=.8, zorder=3)
            ax.text(xi, sub[c].max() + .35, f'{sub[c].mean():.2f}', ha='center',
                    fontsize=8.5, color=INK)
    ax.axhline(STGRAM, ls=':', c=INK, lw=1.4)
    ax.text(-.55, STGRAM + .1, f'STgram-MFN {STGRAM}', fontsize=8.5, color=INK)
    ax.set_xticks(range(len(rules)))
    ax.set_xticklabels([l for _, l in rules], fontsize=8.5)
    ax.set_ylim(87.5, 92.6)
    style(ax, '(b) the bank lever survives every selection rule', '',
          'mean-of-per-ID AUROC (%)')

    # (c) where the bank change lands, per section
    ax = axes[2]
    rows = []
    for name, folds, sec, _ in arm:
        ep = selection_rules(folds, 'section_asnorm')['ep_test']
        for mode in ('class_raw', 'section_asnorm'):
            s = sec[(sec.epoch == ep) & (sec['mode'] == mode)]
            rows += [dict(run=name, mode=mode, section=r.section, auroc=r.auroc * 100)
                     for r in s.itertuples()]
    per_sec = pd.DataFrame(rows).groupby(['section', 'mode']).auroc.mean().unstack()
    per_sec = per_sec.sort_values('class_raw')
    y = np.arange(len(per_sec))
    ax.hlines(y, per_sec.class_raw, per_sec.section_asnorm, color=MUTED, lw=1.2, zorder=1)
    ax.scatter(per_sec.class_raw, y, s=34, color=C_BLUE, zorder=2, label='class_raw')
    ax.scatter(per_sec.section_asnorm, y, s=34, color=C_ORANGE, zorder=2,
               label='section_asnorm')
    ax.set_yticks(y)
    ax.set_yticklabels(per_sec.index, fontsize=7.5)
    for lbl, yi in zip(per_sec.index, y):
        if lbl.startswith('ToyConveyor'):
            ax.get_yticklabels()[yi].set_color(C_VIOLET)
            ax.get_yticklabels()[yi].set_fontweight('bold')
    ax.legend(fontsize=8.5, frameon=False, loc='upper left')
    style(ax, '(c) per-section AUROC at each seed\'s best epoch\n'
              '(ToyConveyor in violet: the "structural floor" was largely the bank)',
          'AUROC (%)', '')
    per_sec.assign(delta=per_sec.section_asnorm - per_sec.class_raw) \
           .to_csv(os.path.join(out_dir, 'bank_lever_per_section.csv'))
    tbl.to_csv(os.path.join(out_dir, 'bank_lever.csv'), index=False)

    fig.tight_layout()
    p = os.path.join(out_dir, 'bank_lever.png')
    fig.savefig(p, dpi=150, facecolor='white')
    plt.close(fig)
    print(f'[fig] {p}')
    print(tbl.groupby('mode')[['test_selected', 'heldout', 'final']].mean().round(2)
          .to_string())


# --------------------------------------------------------------------------- #
# figure 2 — epoch-for-epoch against STgram-MFN
# --------------------------------------------------------------------------- #
def fig_stgram(arm, log_path, result_path, out_dir):
    st_curve, st_best = parse_stgram_log(log_path)
    st_sections = parse_stgram_result(result_path)
    tbl = arm_table(arm)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5),
                             gridspec_kw=dict(width_ratios=[1.15, 1, 1.1]))

    # (a) both training curves on one axis, one metric
    ax = axes[0]
    ax.plot(st_curve.epoch, st_curve['mean'], color=C_AQUA, lw=2, marker='o', ms=3.5,
            label='STgram-MFN (300 ep, eval every 10)')
    ep, mu, lo, hi = curve_band(arm, 'section_asnorm')
    ax.fill_between(ep, lo, hi, color=C_ORANGE, alpha=.15, lw=0)
    ax.plot(ep, mu, color=C_ORANGE, lw=2, label='ours: ResNet34 fbank + section Maha (70 ep)')
    st_ep = int(st_curve.loc[st_curve['mean'].idxmax(), 'epoch'])
    ax.scatter([st_ep], [st_curve['mean'].max()],
               s=70, facecolor='white', edgecolor=C_AQUA, lw=1.8, zorder=4)
    ax.scatter([ep[np.argmax(mu)]], [mu.max()], s=70, facecolor='white',
               edgecolor=C_ORANGE, lw=1.8, zorder=4)
    ax.axhline(tbl.heldout.mean(), ls='--', c=C_ORANGE, lw=1.2)
    ax.text(300, tbl.heldout.mean() + .35, f'ours, held-out {tbl.heldout.mean():.2f}',
            ha='right', fontsize=8.5, color=C_ORANGE)
    ax.axhline(STGRAM, ls=':', c=INK, lw=1.4)
    # the baseline's own headline is a best-epoch-on-test number too -- worth saying out loud,
    # since ours is the one that gets de-optimized by held-out selection.
    ax.annotate(f'STgram headline {STGRAM}\nis its best epoch (@{st_ep}),\nchosen on test',
                xy=(st_ep, st_curve['mean'].max()), xytext=(196, 85.4), fontsize=8.5,
                color=C_AQUA, arrowprops=dict(arrowstyle='-', color=C_AQUA, lw=1,
                                              shrinkA=6, shrinkB=8))
    ax.set_ylim(70, 93)
    ax.legend(fontsize=8.5, frameon=False, loc='lower right')
    style(ax, '(a) same metric, same test set, per epoch',
          'training epoch', 'mean-of-per-ID AUROC (%)')

    # (b) per class at each arm's headline checkpoint
    ax = axes[1]
    ours = []
    for name, folds, sec, _ in arm:
        e = selection_rules(folds, 'section_asnorm')['ep_test']
        s = sec[(sec.epoch == e) & (sec['mode'] == 'section_asnorm')].copy()
        s['cls'] = s.section.str.split('/').str[0]
        ours.append(s.groupby('cls').auroc.mean() * 100)
    ours = pd.concat(ours, axis=1).mean(1)
    order = CLASSES + ['mean']
    ours['mean'], theirs = ours[CLASSES].mean(), st_best.reindex(order)
    theirs['mean'] = st_best[CLASSES].mean()
    x, w = np.arange(len(order)), .38
    ax.bar(x - w / 2 - .01, theirs[order], w, color=C_AQUA, label='STgram-MFN', zorder=2)
    ax.bar(x + w / 2 + .01, ours[order], w, color=C_ORANGE, label='ours', zorder=2)
    for xi, c in zip(x, order):
        d = ours[c] - theirs[c]
        ax.text(xi, max(ours[c], theirs[c]) + 1, f'{d:+.1f}', ha='center', fontsize=8,
                color=C_ORANGE if d > 0 else C_BLUE)
    ax.set_xticks(x)
    ax.set_xticklabels(order, fontsize=8.5, rotation=20, ha='right')
    ax.set_ylim(60, 108)
    ax.legend(fontsize=8.5, frameon=False, loc='upper left', ncol=2)
    style(ax, '(b) per class at each arm\'s best checkpoint', '', 'AUROC (%)')

    # (c) all 23 sections, ours vs theirs
    ax = axes[2]
    rows = []
    for name, folds, sec, _ in arm:
        e = selection_rules(folds, 'section_asnorm')['ep_test']
        s = sec[(sec.epoch == e) & (sec['mode'] == 'section_asnorm')]
        rows += [dict(run=name, section=r.section, auroc=r.auroc * 100)
                 for r in s.itertuples()]
    mine = pd.DataFrame(rows).groupby('section').auroc.mean()
    comp = pd.DataFrame({'stgram': pd.Series(st_sections), 'ours': mine}).dropna()
    comp['cls'] = [i.split('/')[0] for i in comp.index]
    cmap = {c: col for c, col in zip(CLASSES, [C_BLUE, C_ORANGE, C_AQUA, C_VIOLET,
                                               '#eda100', '#e34948'])}
    ax.plot([55, 101], [55, 101], color=MUTED, lw=1, ls='--', zorder=1)
    for c in CLASSES:
        g = comp[comp.cls == c]
        ax.scatter(g.stgram, g.ours, s=48, color=cmap[c], label=c, zorder=2,
                   edgecolor='white', lw=.8)
    for i, r in comp.iterrows():
        if abs(r.ours - r.stgram) > 8:
            ax.annotate(i, (r.stgram, r.ours), fontsize=7, color=MUTED,
                        xytext=(4, -8), textcoords='offset points')
    ax.text(56, 61, f'ours better in {(comp.ours > comp.stgram).sum()}/{len(comp)} sections',
            fontsize=8.5, color=MUTED)
    ax.legend(fontsize=7.5, frameon=False, loc='upper left', ncol=2)
    style(ax, '(c) per-section: ours vs STgram-MFN', 'STgram-MFN AUROC (%)',
          'ours AUROC (%)')

    st_curve.to_csv(os.path.join(out_dir, 'stgram_epoch_curve.csv'), index=False)
    comp.to_csv(os.path.join(out_dir, 'vs_stgram_per_section.csv'))
    fig.tight_layout()
    p = os.path.join(out_dir, 'vs_stgram.png')
    fig.savefig(p, dpi=150, facecolor='white')
    plt.close(fig)
    print(f'[fig] {p}')
    print(f'  STgram best-in-log {st_curve["mean"].max():.2f} @ep '
          f'{int(st_curve.loc[st_curve["mean"].idxmax(), "epoch"])}, final-epoch '
          f'{st_curve["mean"].iloc[-1]:.2f}, reloaded-best {st_best[CLASSES].mean():.2f}')


# --------------------------------------------------------------------------- #
# figure 3 — Rung H: does the Mamba decoder add anything?
# --------------------------------------------------------------------------- #
def fig_rungh(base, mamba, out_dir):
    arms = [('baseline (no decoder)', base, C_ORANGE), ('Rung H (Mamba student)', mamba, C_VIOLET)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    for lbl, arm, col in arms:
        ep, mu, lo, hi = curve_band(arm, 'section_asnorm')
        ax.fill_between(ep, lo, hi, color=col, alpha=.15, lw=0)
        ax.plot(ep, mu, color=col, lw=2, label=lbl)
    ax.axhline(STGRAM, ls=':', c=INK, lw=1.4)
    ax.text(ax.get_xlim()[1], STGRAM + .15, f'STgram-MFN {STGRAM}', ha='right',
            fontsize=8.5, color=INK)
    ax.legend(fontsize=8.5, frameon=False, loc='lower right')
    style(ax, '(a) identical except the decoder\n(fbank 512x128, 3 seeds, band = min-max)',
          'epoch', 'mean-of-per-ID AUROC (%)')

    ax = axes[1]
    rules = [('test_selected', 'test-selected'), ('heldout', 'held-out\n(HONEST)'),
             ('final', 'final epoch')]
    tabs = {lbl: arm_table(arm) for lbl, arm, _ in arms}
    w = .36
    for k, (lbl, _, col) in enumerate(arms):
        sub = tabs[lbl]
        x = np.arange(len(rules)) + (k - .5) * (w + .02)
        ax.bar(x, [sub[c].mean() for c, _ in rules], w, color=col, label=lbl, zorder=2)
        for xi, (c, _) in zip(x, rules):
            ax.scatter(np.full(len(sub), xi), sub[c], s=14, color='white',
                       edgecolor=MUTED, lw=.8, zorder=3)
            ax.text(xi, sub[c].max() + .25, f'{sub[c].mean():.2f}', ha='center',
                    fontsize=8.5, color=INK)
    for i, (c, _) in enumerate(rules):     # panel (a) already carries the arm legend
        d = tabs[arms[1][0]][c].mean() - tabs[arms[0][0]][c].mean()
        ax.text(i, 92.95, f'{d:+.2f}', ha='center', fontsize=9.5, color=C_VIOLET,
                fontweight='bold')
    ax.text(1, 93.5, 'decoder effect (Rung H - baseline)', ha='center', fontsize=8.5,
            color=MUTED)
    ax.axhline(STGRAM, ls=':', c=INK, lw=1.4)
    ax.text(-.62, STGRAM + .1, f'STgram-MFN {STGRAM}', fontsize=8.5, color=INK)
    ax.set_xticks(range(len(rules)))
    ax.set_xticklabels([l for _, l in rules], fontsize=8.5)
    ax.set_ylim(85.8, 94.0)
    style(ax, '(b) the gap holds under every selection rule', '',
          'mean-of-per-ID AUROC (%)')

    ax = axes[2]
    for lbl, arm, col in arms:
        for j, (_, _, _, tr) in enumerate(arm):
            if tr is not None:
                ax.plot(tr.epoch, tr.train_acc, color=col, lw=1.6, alpha=.8,
                        label=lbl if j == 0 else None)
    ax.legend(fontsize=8.5, frameon=False, loc='lower right')
    style(ax, '(c) both arms fit the 23-way task\n'
              'Rung H is a valid run that is simply worse, not a broken one',
          'epoch', 'train accuracy (%)')

    out = pd.concat([t.assign(arm=lbl) for lbl, t in tabs.items()])
    out.to_csv(os.path.join(out_dir, 'rung_h.csv'), index=False)
    fig.tight_layout()
    p = os.path.join(out_dir, 'rung_h.png')
    fig.savefig(p, dpi=150, facecolor='white')
    plt.close(fig)
    print(f'[fig] {p}')
    print(out.groupby('arm')[['test_selected', 'heldout', 'final']]
          .agg(['mean', 'std']).round(2).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold_glob', default='runs/section_rungG/rn34fold_seed*')
    ap.add_argument('--base_glob', default='runs/section_rungG/baseH2_seed*')
    ap.add_argument('--mamba_glob', default='runs/section_rungG/rungH2_seed*')
    ap.add_argument('--stgram_log',
                    default='STgram-MFN/runs/STgram-MFN(m=0.7,s=30)/running.log')
    ap.add_argument('--stgram_result',
                    default='STgram-MFN/results/STgram-MFN(m=0.7,s=30)/result.csv')
    ap.add_argument('--out_dir', default='docs/plots/phase2_asnorm')
    ap.add_argument('--figs', nargs='+', default=['bank', 'stgram', 'rungh'])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    arm = load_arm(args.fold_glob)
    if not arm:
        raise SystemExit(f'no runs with meanid_folds.csv matched {args.fold_glob}')
    print(f'headline arm: {[n for n, *_ in arm]}')
    if 'bank' in args.figs:
        fig_bank(arm, args.out_dir)
    if 'stgram' in args.figs:
        fig_stgram(arm, args.stgram_log, args.stgram_result, args.out_dir)
    if 'rungh' in args.figs:
        base, mamba = load_arm(args.base_glob), load_arm(args.mamba_glob)
        if base and mamba:
            fig_rungh(base, mamba, args.out_dir)
        else:
            print('[skip] rung_h: need both --base_glob and --mamba_glob runs')


if __name__ == '__main__':
    main()
