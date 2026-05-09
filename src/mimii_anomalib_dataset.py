from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from pandas import DataFrame
from torchvision.transforms.v2 import Transform

from anomalib.data.dataclasses import ImageItem
from anomalib.data.datasets.base import AnomalibDataset
from anomalib.data.errors import MisMatchError
from anomalib.data.utils import LabelName, Split, validate_path
from anomalib.utils.path import get_datasets_dir

IMG_EXTENSIONS = (".png", ".PNG")
CATEGORIES = (
    "fan",
    "pump",
    "slider",
    "valve"
)
SEGMENTS = (
    "id_00",
    "id_02",
    "id_04",
	"id_06",
)


class MIMIIAnomalibDataset(AnomalibDataset):
    """MVTec AD dataset class.

    Dataset class for loading and processing MVTec AD dataset images. Supports
    both classification and segmentation tasks.

    Args:
        root (Path | str | None): Path to root directory containing the dataset.
            Defaults to ``None``.
        category (str): Category name, must be one of ``CATEGORIES``.
            Defaults to ``"bottle"``.
        augmentations (Transform, optional): Augmentations that should be applied to the input images.
            Defaults to ``None``.
        split (str | Split | None, optional): Dataset split - usually
            ``Split.TRAIN`` or ``Split.TEST``. Defaults to ``None``.

    Example:
        >>> from pathlib import Path
        >>> from anomalib.data.datasets import MIMIIADDataset
        >>> dataset = MIMIIADDataset(
        ...     root=Path("./datasets/MIMIIAD"),
        ...     category="fan",
        ...     split="train"
        ... )

        For classification tasks, each sample contains:

        >>> sample = dataset[0]
        >>> list(sample.keys())
        ['image_path', 'label', 'image']

        For segmentation tasks, samples also include mask paths and masks:

        >>> dataset.task = "segmentation"
        >>> sample = dataset[0]
        >>> list(sample.keys())
        ['image_path', 'label', 'image', 'mask_path', 'mask']

        Images are PyTorch tensors with shape ``(C, H, W)``, masks have shape
        ``(H, W)``:

        >>> sample["image"].shape, sample["mask"].shape
        (torch.Size([3, 256, 256]), torch.Size([256, 256]))
    """

    def __init__(
        self,
        root: Path | str | None = None,
        categories: Sequence[str] = CATEGORIES,
        segments: Sequence[str] = SEGMENTS,
        augmentations: Transform | None = None,
    ) -> None:
        super().__init__(augmentations=augmentations)

        root = root if root is not None else get_datasets_dir() / "dcase-2020-spectrogram"

        self.root_category = Path(root)
		# verify that category and segment names are valid
        if not all(c in CATEGORIES for c in categories):
            raise ValueError(f"Invalid category name. Must be one of {CATEGORIES}.")
        if not all(s in SEGMENTS for s in segments):
            raise ValueError(f"Invalid segment name. Must be one of {SEGMENTS}.")
        self.category = "|".join(categories)
        self.segment = "|".join(segments)
        self.samples = make_mimii_ad_dataset(
            self.root_category,
            categories=categories,
            segments=segments,
            extensions=IMG_EXTENSIONS,
        )

    def __getitem__(self, index: int) -> ImageItem:
        """Return a classification sample without touching mask paths."""
        image_path = self.samples.iloc[index].image_path
        label_index = self.samples.iloc[index].label_index

        image = self._read_image(image_path)

        if self.augmentations:
            image = self.augmentations(image)

        gt_label = None if label_index == LabelName.UNKNOWN else torch.tensor(label_index)

        return ImageItem(
            image=image,
            gt_label=gt_label,
            gt_mask=None,
            mask_path=None,
            image_path=image_path,
        )

    @staticmethod
    def _read_image(image_path: str) -> Any:
        from anomalib.data.utils import read_image

        return read_image(image_path, as_tensor=True)


def make_mimii_ad_dataset(
    root: str | Path,
    categories: Sequence[str] = CATEGORIES,
	segments: Sequence[str] = SEGMENTS,
    extensions: Sequence[str] | None = None,
) -> DataFrame:
    """Create MVTec AD samples by parsing the data directory structure.

    The files are expected to follow the structure:
        ``path/to/dataset/split/category/image_filename.png``
        ``path/to/dataset/ground_truth/category/mask_filename.png``

    Args:
        root (Path | str): Path to dataset root directory
        split (str | Split | None, optional): Dataset split (train or test)
            Defaults to ``None``.
        extensions (Sequence[str] | None, optional): Valid file extensions
            Defaults to ``None``.

    Returns:
        DataFrame: Dataset samples with columns:
            - path: Base path to dataset
            - split: Dataset split (train/test)
            - label: Class label
            - image_path: Path to image file
            - mask_path: Path to mask file (if available)
            - label_index: Numeric label (0=normal, 1=abnormal)

    Example:
        >>> root = Path("./datasets/MIMIIAD/fan")
        >>> samples = make_mimii_ad_dataset(root, split="train")
        >>> samples.head()
           path                split label image_path           mask_path label_index
        0  datasets/MIMIIAD/fan train good  [...]/good/105.png           0
        1  datasets/MIMIIAD/fan train good  [...]/good/017.png           0

    Raises:
        RuntimeError: If no valid images are found
        MisMatchError: If anomalous images and masks don't match
    """
    if extensions is None:
        extensions = IMG_EXTENSIONS

    root = validate_path(root)
    
    # samples_list = [(str(root), *f.parts[-3:]) for f in root.glob(r"**/*") if f.suffix in extensions]
    samples_list = []
    for category in categories:
        for segment in segments:
            root_path = root / category / segment
            samples_list.extend(
                [(str(root_path), *f.parts[-3:]) for f in root_path.glob(f"**/*") if f.suffix in extensions]
            )
    if not samples_list:
        msg = f"Found 0 images in {root}"
        raise RuntimeError(msg)

    samples = DataFrame(samples_list, columns=["path", "split", "label", "image_path"])

    # Modify image_path column by converting to absolute path
    samples["image_path"] = samples.path + "/" + samples.label + "/" + samples.image_path

    # There is no mask_path
    samples["mask_path"] = None

    # Create label index for normal (0) and anomalous (1) images.
    samples.loc[(samples.label == "normal"), "label_index"] = LabelName.NORMAL
    samples.loc[(samples.label != "normal"), "label_index"] = LabelName.ABNORMAL
    samples["label_index"] = samples["label_index"].astype(int)

    # infer the task type
    samples.attrs["task"] = "classification"

    return samples
