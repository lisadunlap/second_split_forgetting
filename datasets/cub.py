import os
import clip
import torch
import torchvision

import numpy as np
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import random
from mpl_toolkits.axes_grid1 import ImageGrid
import os
import torchvision.transforms as transforms

from datasets.base import VisImageFolder

CUB_DOMAINS = ["photo", "painting"]

with open('/shared/lisabdunlap/data/CUB-200-Painting/classes.txt') as f:
    lines = f.readlines()
    
CUB_CLASSES = [l.replace('\n', '').split('.')[-1].replace('_', ' ') for l in lines]

class Cub2011(torch.utils.data.Dataset):
    base_folder = 'CUB_200_2011/images'

    transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __init__(self, root, split='train', transform=transform, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.split = split

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
    
        if split == 'val':
            self.data.drop([s for i, s in enumerate(self.data.index) if i % 2 == 0], inplace=True)
        elif split == 'test':
            self.data.drop([s for i, s in enumerate(self.data.index) if i % 2 == 1], inplace=True)

        self.samples = [(os.path.join(self.root, self.base_folder, s['filepath']), s['target']-1) for i, s in self.data.iterrows()]
        self.labels = [s['target']-1 for i, s in self.data.iterrows()]
        self.classes = CUB_CLASSES
        

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.split == 'train':
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img =  Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        # return img, target
        return img, target, 0, idx

class Cub2011Painting(VisImageFolder):

    transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __init__(self, root, transform=transform, p=1.0):
        super().__init__(os.path.join(root, 'CUB-200-Painting'), transform=transform)
        self.classes = CUB_CLASSES
        self.samples = [s for i,s in enumerate(self.samples) if i % int(1/p) == 0]
        self.labels = [s[1] for s in self.samples]

    def __getitem__(self, idx):
        path, target = self.samples[idx]

        img =  Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target, 1, idx