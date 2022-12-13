import torch 
import numpy as np

from datasets.waterbirds import Waterbirds100, Waterbirds95, WaterbirdsDiffusion
from datasets.imagenette import Imagenette, ImagenetteWoof, NoisyImagenette, ImagenetteC, ExpandedImagenette
from datasets.cub import Cub2011, Cub2011Painting
from datasets.base import CombinedDataset, SubsetDataset
from datasets.food101 import Food101

def get_dataset(name, cfg, split='train'):
    """
    Gets a dataset by name
    """
    print(f"Get dataset {name}")
    if name == 'Imagenette':
        return Imagenette(root=cfg.data.root, cfg=cfg, split=split)
    elif name == 'ImagenetteWoof':
        return ImagenetteWoof(root=cfg.data.root, cfg=cfg, split=split)
    elif name == 'Waterbirds100':
        return Waterbirds100(root=cfg.data.root, split=split)
    elif name == 'Waterbirds95':
        if split == 'train':
            return Waterbirds95(root=cfg.data.root, split=split)
        return Waterbirds100(root=cfg.data.root, split=split)
    elif name == 'NoisyImagenette':
        return NoisyImagenette(root=cfg.data.root, cfg=cfg, split=split)
    elif name == 'ImagenetteC':
        corruption=cfg.data.imagenetc_corruption
        # datasets = [ImagenetteC(root=cfg.data.root, cfg=cfg, split=split, corruption=corruption, severity=i) for i in range(1,6)]
        return ImagenetteC(root=cfg.data.root, cfg=cfg, split=split, corruption=corruption, severity=1)
    elif name == 'Cub':
        return Cub2011(root=cfg.data.root, split=split)
    elif name == 'CubDirty':
        cub = Cub2011(root=cfg.data.root, split=split)
        if split == 'train':
            painting_dataset = Cub2011Painting(cfg.data.root, p=0.1)
            return CombinedDataset([cub, painting_dataset])
        return cub
    elif name == 'WaterbirdsDiffusion':
        if split == 'train':
            waterbirds_sim = WaterbirdsDiffusion(root="/shared/lisabdunlap/data", split='train')
            waterbirds_real = Waterbirds100(root=cfg.data.root, split='train')
            return CombinedDataset([waterbirds_sim, waterbirds_real])
        return Waterbirds100(root=cfg.data.root, split=split)
    elif name == 'Food101':
        return Food101(root=cfg.data.root, split=split)
    elif name == 'ExpandedImagenette':
        imagenette = Imagenette(root=cfg.data.root, cfg=cfg, split=split)
        imagenette_woof = ExpandedImagenette(root=cfg.data.root, cfg=cfg, split=split)
        return CombinedDataset([imagenette, imagenette_woof])
    else:
        raise ValueError(f"Dataset {name} not supported")
