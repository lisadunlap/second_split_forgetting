import torch
import pandas as pd
from utils import get_counts
import methods
from PIL import Image
import numpy as np
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

class BaseDataset:
    """
    Wrapper for dataset class that allows for mislables.
    """
    def __init__(self, dataset, cfg, clean=False):
        self.dataset = dataset
        self.cfg = cfg
        self.clean = clean
        self.classes = dataset.classes
        # clean = True overrides the config
        self.mislabel_method = getattr(methods, 'noop' if clean else cfg.noise.method)
        self.class_weights = get_counts(self.dataset.labels)
        self.clean_labels = self.dataset.labels
        self.labels, self.noisy_idxs = self.mislabel_method(self.dataset.labels, self.cfg.noise.p)
        self.idxs = [i for i in range(len(self.labels))]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        label = self.labels[idx]

        if len(item) == 2:
            return item[0], label, 0, idx
        elif len(item) == 3:
            return item[0], label, item[2], idx
        else:
            return item[0], label, item[2], idx

    # def remove_examples(self, idxs):
    #     self.idxs = [i for i in self.idxs if i not in idxs]
    #     self.labels = [self.labels[i] for i in self.idxs]
    #     self.noisy_idxs = [i for i in self.noisy_idxs if i not in idxs]
    #     self.class_weights = get_counts(self.labels)
    #     self.dataset = torch.utils.data.Subset(self.dataset, self.idxs)
    #     self.clean_labels = [self.clean_labels[i] for i in self.idxs]

class VisImageFolder(ImageFolder):
    transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def __init__(self, root, transform=transform, split='train'):
        super().__init__(root, transform=transform)
        if split == 'val':
            self.samples = self.samples[::2]
        if split == 'test':
            self.samples = self.samples[1::2]
        self.labels = [s[1] for s in self.samples]

    def __getitem__(self, idx):
        inp, label = super().__getitem__(idx)
        return inp, label, 0, idx

    def vis_example(self, idx):
        filename, label = self.samples[idx]
        return Image.open(filename).convert('RGB')

class CombinedDataset:
    """
    Wrapper that combines a list of datasets into one
    """
    def __init__(self, datasets):
        self.datasets = datasets
        print(datasets[0].samples[0])
        print(datasets[1].samples[0])
        self.samples = list(np.concatenate([d.samples for d in self.datasets]))
        # what dataset does this sample belong to
        self.dataset_idx = list(np.concatenate([np.full(len(d), i) for i, d in enumerate(self.datasets)]))
        
        # what the index of that dataset is for this sample
        self.idxs = list(np.concatenate([np.arange(len(d)) for d in self.datasets]))
        print("HAS ATTR GROUPS: ", hasattr(datasets[0], 'groups'))
        if hasattr(datasets[0], 'groups'):
            assert all([hasattr(d, 'groups') for d in self.datasets]), 'All datasets must have groups attribute' 
            self.filenames, self.labels = [s[0] for s in self.samples], [int(s[1]) for s in self.samples]
            self.groups = np.concatenate([np.array(d.groups) for d in self.datasets])
        else:
            self.filenames, self.labels, self.groups = [s[0] for s in self.samples], [int(s[1]) for s in self.samples], [int(s[2]) for s in self.samples]
            print(len(self.samples))
        self.class_weights = get_counts(self.labels)
        self.classes = np.sort(np.unique(np.concatenate([np.array(d.classes) for d in self.datasets])))
        self.transforms = [d.transform for d in self.datasets]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        didx = self.dataset_idx[idx]
        dataset = self.datasets[didx]
        item = list(dataset[self.idxs[idx]])
        # file, label = self.filenames[idx], self.labels[idx]
        # group = self.groups[idx]

        # inp = Image.open(file).convert('RGB')
        # if self.transforms[didx]:
        #     inp = self.transforms[didx](inp)
        # print(type(item[1]))
        if item[0].shape != torch.Size([3, 224, 224]):
            print(item[0].shape)
        if len(item) == 2:
            return item[0], item[1], self.groups[idx], idx
        elif len(item) == 3:
            return item[0], item[1], item[2], idx
        else:
            return item[0], item[1], item[2], idx
        # print(inp.shape, type(label), group)
        # return inp, np.int64(label), group, idx

class SubsetDataset:
    """
    Wrapper that subsets a dataset
    """
    def __init__(self, dataset, classes, transform=None):
        if type(classes[0]) != int:
            classes = [dataset.classes.index(c) for c in classes]
        self.samples = [item for item in dataset.samples if item[1] in classes]
        self.labels = [item[1] for item in self.samples]
        self.transform = transform
        self.classes = [dataset.classes[c] for c in classes]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img = Image.open(item[0])
        if self.transform:
            img = self.transform(img)
        if len(item) == 2:
            return item[0], item[1], 0, idx
        elif len(item) == 3:
            return item[0], item[1], item[2], idx
        return item