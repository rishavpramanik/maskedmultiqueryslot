import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.datasets.voc import (
    DATASET_YEAR_DICT, VisionDataset, os, verify_str_arg)


class _VOCBase(VisionDataset):
    _SPLITS_DIR: str
    _TARGET_DIR: str
    _TARGET_FILE_EXT: str

    def __init__(
        self,
        root: str,
        year: str = "2012",
        image_set: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        super().__init__(root, transforms, transform, target_transform)

        self.year = verify_str_arg(year, "year", valid_values=[
                                   str(yr) for yr in range(2007, 2013)])
        self.set = image_set
        valid_image_sets = ["train", "trainval", "val"]
        if year == "2007":
            valid_image_sets.append("test")
        self.image_set = verify_str_arg(
            image_set, "image_set", valid_image_sets)

        key = "2007-test" if year == "2007" and image_set == "test" else year
        dataset_year_dict = DATASET_YEAR_DICT[key]

        self.url = dataset_year_dict["url"]
        self.filename = dataset_year_dict["filename"]
        self.md5 = dataset_year_dict["md5"]

        base_dir = dataset_year_dict["base_dir"]
        voc_root = os.path.join(self.root, base_dir)

        if not os.path.isdir(voc_root):
            raise RuntimeError(
                "Dataset not found or corrupted. You can use download=True to download it")

        splits_dir = os.path.join(voc_root, "ImageSets", self._SPLITS_DIR)
        split_f = os.path.join(splits_dir, image_set.rstrip("\n") + ".txt")
        splitval=os.path.join(splits_dir, 'val'.rstrip("\n") + ".txt")
        with open(os.path.join(split_f)) as f:
            file_names = [x.strip() for x in f.readlines()]
        with open(os.path.join(splitval)) as f:
            file_names_val=[x.strip() for x in f.readlines()]

        image_dir = os.path.join(voc_root, "JPEGImages")
        if image_set in ('train'):
            limages = [os.path.join(image_dir, x)
                           for x in os.listdir(image_dir)]
            self.val_images=[os.path.join(image_dir, x + ".jpg")
                           for x in file_names_val]
            self.images=list(set(limages)-set(self.val_images))
        elif image_set == 'val' or 'trainval':
            self.images = [os.path.join(image_dir, x + ".jpg")
                           for x in file_names]

            target_dir = os.path.join(voc_root, self._TARGET_DIR)
            self.targets = [os.path.join(
                target_dir, x + self._TARGET_FILE_EXT) for x in file_names]

            assert len(self.images) == len(self.targets)

    def __len__(self) -> int:
        return len(self.images)


labels = [0.0,
          0.003921568859368563,
          0.007843137718737125,
          0.0117647061124444,
          0.01568627543747425,
          0.019607843831181526,
          0.0235294122248888,
          0.027450980618596077,
          0.0313725508749485,
          0.03529411926865578,
          0.03921568766236305,
          0.04313725605607033,
          0.0470588244497776,
          0.05098039284348488,
          0.054901961237192154,
          0.05882352963089943,
          0.062745101749897,
          0.06666667014360428,
          0.07058823853731155,
          0.07450980693101883,
          0.0784313753247261,
          1]


class VOCSegmentation(_VOCBase):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.

    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years ``"2007"`` to ``"2012"``.
        image_set (string, optional): Select the image_set to use, ``"train"``, ``"trainval"`` or ``"val"``. If
            ``year=="2007"``, can also be ``"test"``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    _SPLITS_DIR = "Segmentation"
    _TARGET_DIR = "SegmentationClass"
    _TARGET_FILE_EXT = ".png"

    def __init__(self, evo: bool = False, *args, transform=None, dino_transform=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.transform = transform
        self.dino_transform = dino_transform
        self.evo = evo
        self.brisk = list()

    @property
    def masks(self) -> List[str]:
        return self.targets

    def one_hot_encode(self, segmentation_map):
        # Ensure segmentation_map is a PyTorch tensor

        if not torch.is_tensor(segmentation_map):
            segmentation_map = F.to_tensor(segmentation_map)

        one_hot_map = torch.zeros(
            (len(labels), segmentation_map.shape[1], segmentation_map.shape[2]), dtype=torch.uint8)

        # Fill the one-hot encoding tensor
        for class_idx in range(len(labels)):
            one_hot_map[class_idx] = (segmentation_map[0] == labels[class_idx]).byte()
        one_hot_map[-1,:,:]=torch.zeros_like(one_hot_map[-1,:,:])

        return one_hot_map[:-1]

    def __getitem__(self, index: int) -> Tuple[Any, Any]: # type: ignore
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert("RGB")
        if self.transform is not None:

            dino_dta = self.dino_transform(img)
            img = self.transform(dino_dta)

            if self.set in ('val','trainval') and self.evo:
                target = Image.open(self.masks[index])# .convert("RGB")
                target = self.transform(target)
                target = self.one_hot_encode(TF.to_tensor(target))

                return img, dino_dta, target  # type: ignore
            else:
                return img, dino_dta # type: ignore
