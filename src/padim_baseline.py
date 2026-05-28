from mvtecad_patched_datamodule import MVTecADPatched
from anomalib.data.datamodules import MVTecAD
from anomalib.models import Padim
from padim_adaln.lightning_model import PadimAdaLN
from anomalib.engine import Engine
from timm.data.constants import IMAGENET_DEFAULT_MEAN
from timm.data.constants import IMAGENET_DEFAULT_STD
from torchvision.transforms.v2 import Compose, Resize, ToTensor, Normalize, CenterCrop

def test_against_model(model, model_name, datamodule):
	print(f"Testing {model_name} model")
	engine = Engine()
	engine.fit(model=model, datamodule=datamodule)
	engine.test(model=model, datamodule=datamodule)

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
dm = MVTecADPatched(
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

adaln_resnet18_model = PadimAdaLN(backbone="resnet18", n_features=100)
test_against_model(adaln_resnet18_model, "ResNet18 PaDiM-AdaLN", dm)
adaln_wide_resnet50_2_model = PadimAdaLN(backbone="wide_resnet50_2", n_features=550)
test_against_model(adaln_wide_resnet50_2_model, "Wide ResNet50-2 PaDiM-AdaLN", dm)

resnet18_model = Padim(backbone="resnet18", n_features=100)
test_against_model(resnet18_model, "ResNet18 PaDiM", dm)
wide_resnet50_2_model = Padim(backbone="wide_resnet50_2", n_features=550)
test_against_model(wide_resnet50_2_model, "Wide ResNet50-2 PaDiM", dm)
