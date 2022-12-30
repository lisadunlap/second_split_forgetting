import torch
import pandas as pd
from utils import get_counts
import methods.remove as remove
from PIL import Image
import numpy as np
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

from collections import Counter

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
        self.mislabel_method = getattr(remove, 'noop' if clean else cfg.noise.method)
        self.class_weights = get_counts(self.dataset.labels)
        self.clean_labels = self.dataset.labels
        self.labels, self.noisy_idxs = self.mislabel_method(cfg, self.dataset.labels, self.cfg.noise.p)
        self.idxs = [i for i in range(len(self.labels))]
        self.groups = dataset.groups if hasattr(dataset, 'groups') else [1 for _ in range(len(self.labels))]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        label = self.labels[idx]

        if len(item) == 2:
            return item[0], label, self.groups[idx], idx
        elif len(item) == 3:
            return item[0], label, self.groups[idx], idx
        else:
            return item[0], label, self.groups[idx], idx

class VisImageFolder(ImageFolder):
    default_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def __init__(self, root, transform=default_transform, split='train'):
        super().__init__(root, transform=transform)
        if split == 'val':
            self.samples = self.samples[::2]
        if split == 'test':
            self.samples = self.samples[1::2]
        self.labels = [s[1] for s in self.samples]
        self.groups = [0 for s in self.samples]

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
        self.samples = list(np.concatenate([d.samples for d in self.datasets]))
        # what dataset does this sample belong to
        self.dataset_idx = list(np.concatenate([np.full(len(d), i) for i, d in enumerate(self.datasets)]))
        
        # what the index of that dataset is for this sample
        self.idxs = list(np.concatenate([np.arange(len(d)) for d in self.datasets]))
        if hasattr(datasets[0], 'groups'):
            assert all([hasattr(d, 'groups') for d in self.datasets]), 'All datasets must have groups attribute' 
            self.filenames, self.labels = [s[0] for s in self.samples], [int(s[1]) for s in self.samples]
            self.groups = np.concatenate([np.array(d.groups) for d in self.datasets])
        else:
            self.filenames, self.labels, self.groups = [s[0] for s in self.samples], [int(s[1]) for s in self.samples], [int(s[2]) for s in self.samples]
        self.class_weights = get_counts(self.labels)
        self.classes = np.sort(np.unique(np.concatenate([np.array(d.classes) for d in self.datasets])))
        self.transforms = [d.transform for d in self.datasets]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        didx = self.dataset_idx[idx]
        dataset = self.datasets[didx]
        item = list(dataset[self.idxs[idx]])
        if item[0].shape != torch.Size([3, 224, 224]):
            print(item[0].shape)
        if len(item) == 2:
            return item[0], item[1], self.groups[idx], idx
        elif len(item) == 3:
            return item[0], item[1], item[2], idx
        else:
            return item[0], item[1], item[2], idx

    def vis_example(self, idx):
        didx = self.dataset_idx[idx]
        dataset = self.datasets[didx]
        return dataset.vis_example(self.idxs[idx])

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

class EmbeddingDataset:

    def __init__(self, dataset, embeddings, labels, groups, idxs):
        self.dataset = dataset
        self.embeddings = embeddings
        # self.samples = dataset.samples
        self.labels = labels
        self.groups = groups
        self.classes = dataset.classes
        self.class_weights = dataset.class_weights
        self.idxs = idxs
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx], self.groups[idx], idx

def get_sampler(cfg, dataset, samples_to_remove=[], samples_to_upweight=[], split='train'):
    """
    Returns a sampler for the given dataset. (weighted, random, etc.)
    Upweight is dictated by cfg.data.upweight_factor
    """
    assert all([r != u for r, u in zip(samples_to_remove, samples_to_upweight)]), 'samples to remove and upweight must be disjoint'
    image_ids = [i for i in range(len(dataset))]
    weights = np.ones(len(image_ids))
    weights[samples_to_remove] = 0
    weights[samples_to_upweight] *= cfg.data.upweight_factor
    # normalize by class counts
    labels = [l for l in dataset.labels if weights[l] > 0]
    groups = [g for g in dataset.dataset.groups if weights[g] > 0]
    for i, (label, group) in enumerate(zip(labels, groups)):
        label_subset = [i for i,l in enumerate(labels) if l == label]
        # group_subset = [i for i,g in enumerate(groups) if g == group]
        # subset = list(set(label_subset).union(group_subset))
        weights[i] /= np.sum(label_subset)
    weights /= np.max(weights)
    # print(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights), replacement=True)
    return sampler
