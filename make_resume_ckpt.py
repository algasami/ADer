"""build a resume checkpoint from a periodic net_<epoch>.pth snapshot.
python make_resume_ckpt.py --run_dir runs/<run> --epoch <epoch> --lr 5e-4
resume with:
python run.py -c <cfg> -m train trainer.resume_dir=<run> \
    model.kwargs.checkpoint_path=ckpt_resume<epoch>.pth
"""
import argparse
import os

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_dir', required=True, help='run directory containing ckpt.pth and net_<epoch>.pth')
    parser.add_argument('--epoch', type=int, required=True, help='epoch of the net_<epoch>.pth snapshot to resume from')
    parser.add_argument('--lr', type=float, default=None, help='new base lr (default: keep the lr stored in ckpt.pth)')
    parser.add_argument('--out', default=None, help='output path (default: <run_dir>/ckpt_resume<epoch>.pth)')
    args = parser.parse_args()

    net_path = os.path.join(args.run_dir, f'net_{args.epoch}.pth')
    tpl_path = os.path.join(args.run_dir, 'ckpt.pth')
    out_path = args.out or os.path.join(args.run_dir, f'ckpt_resume{args.epoch}.pth')

    net = torch.load(net_path, map_location='cpu', weights_only=False)
    nonfinite = [k for k, v in net.items()
                 if isinstance(v, torch.Tensor) and v.is_floating_point() and not torch.isfinite(v).all()]
    if nonfinite:
        raise SystemExit(f'{net_path} has {len(nonfinite)} non-finite tensors (e.g. {nonfinite[0]}); '
                         f'pick an earlier epoch')

    tpl = torch.load(tpl_path, map_location='cpu', weights_only=False)
    if tpl['epoch'] < args.epoch:
        raise SystemExit(f"ckpt.pth is at epoch {tpl['epoch']}, before requested epoch {args.epoch}")
    iters_per_epoch = tpl['iter'] // tpl['epoch']

    optimizer = {'state': {}, 'param_groups': tpl['optimizer']['param_groups']}
    scheduler = dict(tpl['scheduler'])
    if args.lr is not None:
        for group in optimizer['param_groups']:
            group['lr'] = args.lr
            group['initial_lr'] = args.lr
        scheduler['base_values'] = [args.lr] * len(scheduler['base_values'])

    metric_recorder = {k: v[:args.epoch] for k, v in tpl['metric_recorder'].items()}

    ckpt = {'net': net,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'scaler': None,
            'iter': args.epoch * iters_per_epoch,
            'epoch': args.epoch,
            'metric_recorder': metric_recorder,
            'total_time': tpl['total_time'] * args.epoch / tpl['epoch']}
    torch.save(ckpt, out_path)
    print(f"saved {out_path}: epoch={args.epoch} iter={ckpt['iter']} "
          f"base_lr={scheduler['base_values'][0]} (optimizer state reset)")


if __name__ == '__main__':
    main()
