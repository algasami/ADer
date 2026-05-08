import os
import sys
import signal
import time
from argparse import Namespace

import torchvision.transforms.functional as F

sys.path.insert(0, os.getcwd())


def alarm_handler(signum, frame):
    raise TimeoutError("sample timeout")


def build_cfg():
    cfg = Namespace()
    cfg.dist = False
    cfg.data = Namespace()
    cfg.data.type = 'DefaultAD'
    cfg.data.root = 'data/dcase-2020-spectrogram'
    cfg.data.meta = 'meta.json'
    cfg.data.cls_names = []
    cfg.data.loader_type = 'pil'
    cfg.data.loader_type_target = 'pil_L'
    cfg.data.train_transforms = [
        dict(type='Resize', size=(256, 256), interpolation=F.InterpolationMode.BILINEAR),
        dict(type='CenterCrop', size=(256, 256)),
        dict(type='ToTensor'),
        dict(type='Normalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True),
    ]
    cfg.data.test_transforms = cfg.data.train_transforms
    cfg.data.target_transforms = [
        dict(type='Resize', size=(256, 256), interpolation=F.InterpolationMode.NEAREST),
        dict(type='CenterCrop', size=(256, 256)),
        dict(type='ToTensor'),
    ]
    return cfg


def main():
    signal.signal(signal.SIGALRM, alarm_handler)
    cfg = build_cfg()
    from data import get_dataset
    train_set, _ = get_dataset(cfg)
    print(f"dataset length={len(train_set)}")

    start = time.time()
    for idx in range(len(train_set)):
        data_meta = train_set.data_all[idx]
        img_rel = data_meta['img_path']
        img_abs = f"{train_set.root}/{img_rel}"
        try:
            signal.alarm(3)
            sample = train_set[idx]
            signal.alarm(0)
        except TimeoutError:
            print(f"TIMEOUT idx={idx} img={img_abs}")
            return
        except Exception as e:
            signal.alarm(0)
            print(f"ERROR idx={idx} img={img_abs} err={type(e).__name__}: {e}")
            return

        if idx % 500 == 0:
            print(f"ok idx={idx} elapsed={time.time()-start:.1f}s")

    print(f"all samples ok in {time.time()-start:.1f}s")


if __name__ == '__main__':
    main()
