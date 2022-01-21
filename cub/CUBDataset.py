
import sys
import random

import torch
import torch.utils.data as data

import warnings
from PIL import Image
import os
import os.path
import gzip
import numpy as np
import torch
import random
from .CUBSingleDatasets import CUBSentences, CUBImageFt

class VisionDataset(data.Dataset):
    _repr_indent = 4

    def __init__(self, root):
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = root

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __repr__(self):
        head = "Dataset " + self.dataset
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        if hasattr(self, 'transform') and self.transform is not None:
            body += self._format_transform_repr(self.transform,
                                                "Transforms: ")
        if hasattr(self, 'target_transform') and self.target_transform is not None:
            body += self._format_transform_repr(self.target_transform,
                                                "Target transforms: ")
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def _format_transform_repr(self, transform, head):
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def extra_repr(self):
        return ""

class CUBDataset(VisionDataset):
    def __init__(self, flags, split, device, transform=None, **kwargs):
        """split: 'train' or 'test' """
        super(CUBDataset, self).__init__(flags.dir_data)
        self.dataset = 'CUB'
        self.dir_img = os.path.join(self.root, self.dataset)
        self.dir_txt = os.path.join(self.root, self.dataset)
        self.split = split
        self.CUBtxt = CUBSentences(self.dir_txt, split=split, transform=transform, **kwargs)
        self.CUBimg = CUBImageFt(self.dir_img, split=split, device=device)
        
    def __len__(self):
        return len(self.CUBtxt) - 1
    
    def __getitem__(self, idx):
        txt = self.CUBtxt.__getitem__(idx)[0]
        img = self.CUBimg.__getitem__(idx // 10)
        return {'cub_img': img, 'cub_txt': txt}, idx
        
    def extra_repr(self):
        return 'Split: ' + self.split

    @staticmethod
    def extract_gzip(gzip_path, remove_finished=False):
        print('Extracting {}'.format(gzip_path))
        with open(gzip_path.replace('.gz', ''), 'wb') as out_f, \
                gzip.GzipFile(gzip_path) as zip_f:
            out_f.write(zip_f.read())
        if remove_finished:
            os.unlink(gzip_path)

    
