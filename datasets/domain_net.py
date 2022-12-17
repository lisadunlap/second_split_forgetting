import os
from PIL import Image
import numpy as np
from utils import get_counts
import torchvision.transforms as transforms
from datasets.base import VisImageFolder

class DomainNet(VisImageFolder):
    default_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __init__(self, root, cfg, domains, split='train', transform=default_transform):
        super().__init__(root, transform=transform)
        if type(domains) == str: # for single domain
            domains = [domains]
        self.root = root
        self.cfg = cfg
        self.split = split
        self.transform = transform
        self.domains = sorted(domains)
        if cfg.dataset.classes.file:
            with open(cfg.dataset.classes.file) as f:
                self.classes = [c.replace(' ', '_') for c in f.read().splitlines()]
        else:
            self.classes = cfg.dataset.classes.class_list
        self.samples, self.labels, self.img_paths, self.groups = [], [], [], []
        file_split = 'test' if self.split == 'val' else self.split
        for i, domain in enumerate(self.domains):
            with open(f'{root}/{domain}_{file_split}.txt') as f:
                samples = [(os.path.join(root, i.split(' ')[0]), int(i.split(' ')[1])) for i in f.read().splitlines()]
            class_names = [s[0].split('/')[6] for s in samples]
            samples = [(s[0], self.classes.index(class_names[i])) for i, s in enumerate(samples) if class_names[i] in self.classes]
            if domain != 'real' and split == 'train':
                samples = list(np.array(samples)[np.random.choice(list(range(len(samples))), 1000, replace=False)])
            elif split  != 'train':
                balanced_samples = []
                for j in range(len(self.classes)):
                    subsample = [s for s in samples if s[1] == j]
                    balanced_samples += list(np.array(subsample)[np.random.choice(list(range(len(subsample))), 31, replace=False)])
                samples = balanced_samples

            # sort out classes
            self.samples += samples
            self.groups += [i for _ in range(len(samples))]
        self.samples = [(s[0], self.classes.index(s[0].split('/')[6])) for s in self.samples]
        # if split == 'val':
        #     self.samples = self.samples[::2]
        #     self.groups = self.groups[::2]
        # if split == 'test':
        #     self.samples = self.samples[1::2]
        #     self.groups = self.groups[1::2]
        self.labels = [s[1] for s in self.samples]
        self.img_paths = [s[0] for s in self.samples]
        # self.class_weights = get_counts(self.labels)
        assert len(self.samples) == len(self.labels) == len(self.img_paths) == len(self.groups), "samples, labels, img_paths, and groups should be the same length"

    def __getitem__(self, idx):
        img, label, group, idx = super().__getitem__(idx)
        return img, label, self.groups[idx], idx
        