import time
import signal

def alarm_handler(signum, frame):
    raise TimeoutError("timeout")

signal.signal(signal.SIGALRM, alarm_handler)

def try_fetch(num_workers):
    print(f"\n--- Testing num_workers={num_workers} ---")
    import sys, os
    sys.path.insert(0, os.getcwd())
    from argparse import Namespace
    # minimal cfg matching mambaad_mimii_toy dataset settings
    cfg = Namespace()
    cfg.data = Namespace()
    cfg.data.type = 'DefaultAD'
    cfg.data.root = 'data/dcase-2020-spectrogram'
    cfg.data.meta = 'meta.json'
    cfg.data.cls_names = []
    cfg.data.loader_type = 'pil'
    cfg.data.loader_type_target = 'pil_L'
    import torchvision.transforms.functional as F
    # minimal transforms: Resize -> ToTensor -> Normalize
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
    cfg.trainer = Namespace()
    cfg.trainer.data = Namespace()
    cfg.trainer.data.num_workers_per_gpu = num_workers
    cfg.trainer.data.persistent_workers = False
    cfg.trainer.data.batch_size_per_gpu = 16
    cfg.trainer.data.batch_size_per_gpu_test = 16
    cfg.dist = False
    cfg.trainer.data.pin_memory = False
    cfg.trainer.data.drop_last = True
    cfg.trainer.data.timeout = 30 if num_workers > 0 else 0

    from data import get_loader
    from data import get_dataset
    start = time.time()
    try:
        if num_workers > 0:
            signal.alarm(30)
        else:
            signal.alarm(0)
        train_set, _ = get_dataset(cfg)
        print('About to call train_set[0]')
        sample0 = train_set[0]
        print('train_set[0] keys:', list(sample0.keys()))
        train_loader, test_loader = get_loader(cfg)
        it = iter(train_loader)
        batch = next(it)
        signal.alarm(0)
        if isinstance(batch, dict):
            print("Fetched batch keys:", list(batch.keys()))
        else:
            print("Fetched batch type:", type(batch))
    except TimeoutError:
        print("Timed out while fetching batch")
    except Exception as e:
        print("Exception:", e)
    finally:
        signal.alarm(0)
    print("Elapsed:", time.time() - start)

if __name__ == '__main__':
    try_fetch(4)
    try_fetch(0)
