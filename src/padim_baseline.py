from mvtecad_patched_datamodule import MVTecADPatched
from anomalib.data.datamodules import MVTecAD
from anomalib.models import Padim
from padim_adaln.lightning_model import PadimAdaLN
from anomalib.engine import Engine
from timm.data.constants import IMAGENET_DEFAULT_MEAN
from timm.data.constants import IMAGENET_DEFAULT_STD
from torchvision.transforms.v2 import Compose, Resize, ToTensor, Normalize, CenterCrop
import torch

performance_dict: dict[str, dict[str, float]] = {}

def test_against_model(model, model_name, datamodule):
	print(f"Testing {model_name} model")
	engine = Engine()
	engine.fit(model=model, datamodule=datamodule)
	results = engine.test(model=model, datamodule=datamodule)
	performance_dict[model_name] = dict(results[0])

categories = (
	"bottle",
	"cable",
	"capsule",
	"carpet",
	"grid",
	"hazelnut",
	"leather",
	"metal_nut",
	"pill",
	"screw",
	"tile",
	"toothbrush",
	"transistor",
	"wood",
	"zipper",
)
patched_dm = MVTecADPatched(
	root="../data/mvtec",
	train_batch_size=32,
	eval_batch_size=32,
	num_workers=8,
	categories=categories,
	augmentations=Compose([
		Resize((256, 256)),
		CenterCrop((256, 256)),
		ToTensor(),
		Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, inplace=True),
	]),
)

# adaln_resnet18_model = PadimAdaLN(backbone="resnet18", n_features=100)
# test_against_model(adaln_resnet18_model, "ResNet18 PaDiM-AdaLN", patched_dm)
# adaln_wide_resnet50_2_model = PadimAdaLN(backbone="wide_resnet50_2", n_features=550)
# test_against_model(adaln_wide_resnet50_2_model, "Wide ResNet50-2 PaDiM-AdaLN", patched_dm)

# General-purpose Models
general_resnet18_model = Padim(backbone="resnet18", n_features=100)
test_against_model(general_resnet18_model, "ResNet18 PaDiM - General", patched_dm)
general_wide_resnet50_2_model = Padim(backbone="wide_resnet50_2", n_features=550)
test_against_model(general_wide_resnet50_2_model, "Wide ResNet50-2 PaDiM - General", patched_dm)

del general_resnet18_model
del general_wide_resnet50_2_model

torch.cuda.empty_cache()

# Specialized Models
for category in categories:
	print(f"Testing category: {category}")
	category_dm = MVTecAD(
		root="../data/mvtec",
		train_batch_size=32,
		eval_batch_size=32,
		num_workers=8,
		category=category,
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

headers = ["model", "image_AUROC", "image_F1Score", "pixel_AUROC", "pixel_F1Score"]

# output csv file
with open("padim_performance.csv", "w") as f:
	f.write(",".join(headers) + "\n")
	for model_name, performance in performance_dict.items():
		row = [
			model_name,
			str(performance["image_AUROC"]),
			str(performance["image_F1Score"]),
			str(performance["pixel_AUROC"]),
			str(performance["pixel_F1Score"]),
		]
		f.write(",".join(row) + "\n")

from tabulate import tabulate

print(tabulate([[model_name] + [str(performance[key]) for key in headers[1:]]
		  for model_name, performance in performance_dict.items()],
		  headers=headers,
		  tablefmt="grid"))
