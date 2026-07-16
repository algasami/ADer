from mimii_anomalib_datamodule import MIMIIAnomalibDataModule
from anomalib.models import Padim
from padim_adaln.lightning_model import PadimAdaLN
from anomalib.engine import Engine
from timm.data.constants import IMAGENET_DEFAULT_MEAN
from timm.data.constants import IMAGENET_DEFAULT_STD
from torchvision.transforms.v2 import Compose, Resize, ToTensor, Normalize, CenterCrop
import torch

performance_dict: dict[str, dict[str, float]] = {}

def test_against_model(model, model_name, datamodule, limit_train_batches=None):
	print(f"Testing {model_name} model")
	engine = Engine(limit_train_batches=limit_train_batches) if limit_train_batches else Engine()
	engine.fit(model=model, datamodule=datamodule)
	results = engine.test(model=model, datamodule=datamodule)
	performance_dict[model_name] = dict(results[0])

CATEGORIES = (
    "fan",
    "pump",
    "slider",
    "valve",
	"ToyConveyor",
	"ToyCar",
)
patched_dm = MIMIIAnomalibDataModule(
	root="../data/dcase-2020-spectrogram",
	train_batch_size=32,
	eval_batch_size=32,
	num_workers=8,
	categories=CATEGORIES,
	augmentations=Compose([
		Resize((256, 256)),
		CenterCrop((256, 256)),
		ToTensor(),
		Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, inplace=True),
	]),
)

# General-purpose Models
# Cap training samples: the combined 4-category train set (~4704 imgs) makes
# PaDiM's on-GPU memory bank too large for wide_resnet50_2 (39.5 GiB, doubled by
# vstack during fit). 64 batches (~2048 imgs) keep the Gaussian representative
# while bounding peak GPU memory. Both backbones use the same subset for fairness.
GENERAL_LIMIT_TRAIN_BATCHES = 64
general_resnet18_model = Padim(backbone="resnet18", n_features=100)
test_against_model(general_resnet18_model, "ResNet18 PaDiM - General", patched_dm,
				   limit_train_batches=GENERAL_LIMIT_TRAIN_BATCHES)

# Free the resnet18 run's state before the large wide_resnet50_2 allocation.
del general_resnet18_model
torch.cuda.empty_cache()

general_wide_resnet50_2_model = Padim(backbone="wide_resnet50_2", n_features=550)
test_against_model(general_wide_resnet50_2_model, "Wide ResNet50-2 PaDiM - General", patched_dm,
				   limit_train_batches=GENERAL_LIMIT_TRAIN_BATCHES)

del general_wide_resnet50_2_model
torch.cuda.empty_cache()

# Specialized Models
for category in CATEGORIES:
	print(f"Testing category: {category}")
	category_dm = MIMIIAnomalibDataModule(
		root="../data/dcase-2020-spectrogram",
		train_batch_size=32,
		eval_batch_size=32,
		num_workers=8,
		categories=(category,),
		augmentations=Compose([
			Resize((256, 256)),
			CenterCrop((256, 256)),
			ToTensor(),
			Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, inplace=True),
		]),
	)
	resnet18_model = Padim(backbone="resnet18", n_features=100)
	wide_resnet50_2_model = Padim(backbone="wide_resnet50_2", n_features=550)
	test_against_model(resnet18_model, f"ResNet18 PaDiM - {category}", category_dm)
	test_against_model(wide_resnet50_2_model, f"Wide ResNet50-2 PaDiM - {category}", category_dm)

	del resnet18_model
	del wide_resnet50_2_model
	torch.cuda.empty_cache()

headers = ["model", "image_AUROC", "image_F1Score"]

# output csv file
with open("padim_performance_mimii.csv", "w") as f:
	f.write(",".join(headers) + "\n")
	for model_name, performance in performance_dict.items():
		row = [
			model_name,
			str(performance["image_AUROC"]),
			str(performance["image_F1Score"]),
		]
		f.write(",".join(row) + "\n")

from tabulate import tabulate

print(tabulate([[model_name] + [str(performance[key]) for key in headers[1:]]
		  for model_name, performance in performance_dict.items()],
		  headers=headers,
		  tablefmt="grid"))
