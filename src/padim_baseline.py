from mvtecad_patched_datamodule import MVTecADPatched
from anomalib.models import Padim
from anomalib.engine import Engine
from timm.data.constants import IMAGENET_DEFAULT_MEAN
from timm.data.constants import IMAGENET_DEFAULT_STD
from torchvision.transforms.v2 import Compose, Resize, ToTensor, Normalize, CenterCrop

if __name__ == "__main__":
	dm = MVTecADPatched(
		root="../data/mvtec",
		train_batch_size=32,
		eval_batch_size=32,
		num_workers=8,
		categories=(
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
		),
		augmentations=Compose([
			Resize((256, 256)),
			CenterCrop((256, 256)),
			ToTensor(),
			Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, inplace=True),
		]),
	)
	print("Initialized datamodule")
	engine = Engine()
	print("Initialized engine")
	resnet18_model = Padim(backbone="resnet18", n_features=100)
	print("Initialized resnet18 model")
	engine.fit(model=resnet18_model, datamodule=dm)
	engine.test(model=resnet18_model, datamodule=dm)
	print("Finished testing resnet18 model")

	wide_resnet50_2_model = Padim(backbone="wide_resnet50_2", n_features=550)
	print("Initialized wide_resnet50_2 model")
	engine.fit(model=wide_resnet50_2_model, datamodule=dm)
	engine.test(model=wide_resnet50_2_model, datamodule=dm)
	print("Finished testing wide_resnet50_2 model")
