import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Tuple, Union


import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from datas.load import DinoFeatExtractor


class ImagesDataset(Dataset):
    """A very simple dataset for loading images."""

    def __init__(self, filenames: str, images_root: Optional[str] = None, transform: Optional[Callable] = None,
                 prepare_filenames: bool = True) -> None:
        self.root = None if images_root is None else Path(images_root)
        self.filenames = sorted(list(set(filenames))) if prepare_filenames else filenames
        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path = self.filenames[index]
        full_path = Path(path) if self.root is None else self.root / path
        assert full_path.is_file(), f'Not a file: {full_path}'
        image = Image.open(str(full_path))
        image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, path, index

    def __len__(self) -> int:
        return len(self.filenames)


def get_model(name: str,model_name,patch_size=8,version=1):
    if 'dino' in name:
        model_name = DinoFeatExtractor(model_name,True,True,
                                       patch_size,version)
        model=model_name.load_dino_backbone()
        model.fc = torch.nn.Identity()
        val_transform = get_transform(name)
        patch_size = model_name.patch_size
        num_heads = model.blocks[0].attn.num_heads
    else:
        raise ValueError(f'Cannot get model: {name}')
    model.eval()
    return model, val_transform, patch_size, num_heads


def get_transform(name: str):
    # if any(x in name for x in ('dino', 'mocov3', 'convnext', )):
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        # transform = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(320),normalize])
    # else:
        # raise NotImplementedError()
    return transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(320),normalize])


def get_inverse_transform(name: str):
    if 'dino' in name:
        inv_normalize = transforms.Normalize(
            [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            [1 / 0.229, 1 / 0.224, 1 / 0.225])
        transform = transforms.Compose([transforms.ToTensor(), inv_normalize])
    else:
        raise NotImplementedError()
    return transform


def get_image_sizes(data_dict: dict, downsample_factor: Optional[int] = None):
    P = data_dict['patch_size'] if downsample_factor is None else downsample_factor
    B, C, H, W = data_dict['shape']
    assert B == 1, 'assumption violated :('
    H_patch, W_patch = H // P, W // P
    H_pad, W_pad = H_patch * P, W_patch * P
    return (B, C, H, W, P, H_patch, W_patch, H_pad, W_pad)


def _get_files(p: str):
    if Path(p).is_dir():
        return sorted(Path(p).iterdir())
    elif Path(p).is_file():
        return Path(p).read_text().splitlines()
    else:
        raise ValueError(p)


def get_paired_input_files(path1: str, path2: str):
    files1 = _get_files(path1)
    files2 = _get_files(path2)
    assert len(files1) == len(files2)
    return list(enumerate(zip(files1, files2)))


def make_output_dir(output_dir, check_if_empty=True):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    if check_if_empty and (len(list(output_dir.iterdir())) > 0):
        print(f'Output dir: {str(output_dir)}')
        if input(f'Output dir already contains files. Continue? (y/n) >> ') != 'y':
            sys.exit()  # skip because already generated




def get_border_fraction(segmap: np.array):
    num_border_pixels = 2 * (segmap.shape[0] + segmap.shape[1])
    counts_map = {idx: 0 for idx in np.unique(segmap)}
    np.zeros(len(np.unique(segmap)))
    for border in [segmap[:, 0], segmap[:, -1], segmap[0, :], segmap[-1, :]]:
        unique, counts = np.unique(border, return_counts=True)
        for idx, count in zip(unique.tolist(), counts.tolist()):
            counts_map[idx] += count
    # normlized_counts_map = {idx: count / num_border_pixels for idx, count in counts_map.items()}
    indices = np.array(list(counts_map.keys()))
    normlized_counts = np.array(list(counts_map.values())) / num_border_pixels
    return indices, normlized_counts


def parallel_process(inputs: Iterable, fn: Callable, multiprocessing: int = 0):
    start = time.time()
    if multiprocessing:
        print('Starting multiprocessing')
        with Pool(multiprocessing) as pool:
            for _ in tqdm(pool.imap(fn, inputs), total=len(inputs)):
                pass
    else:
        for inp in tqdm(inputs):
            fn(inp)
    print(f'Finished in {time.time() - start:.1f}s')
