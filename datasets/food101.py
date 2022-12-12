import os
import torchvision.transforms as transforms

from datasets.base import VisImageFolder

class Food101(VisImageFolder):

    def __init__(self, root, transform=None, split='train'):
        s = 'train' if split == 'train' else 'test'
        super().__init__(os.path.join(root, 'food-101', s))