
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
from torch.distributions import multivariate_normal as mv
import random

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


class MNISTGMDataset(VisionDataset):
    training_file_mnist = 'train.pt'
    test_file_mnist = 'test.pt'
    classes = ['1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight']


    def __init__(self, flags,  alphabet, train=True, transform=None, target_transform=None):
        super(MNISTGMDataset, self).__init__(flags.dir_data)
        self.flags = flags
        self.dataset = 'MNIST_GM'
        self.dataset_mnist = 'MNIST'
        self.dataset_gm = 'GM'
        self.len_sequence = flags.len_sequence
        self.sample_size = flags.GM_sample_size
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.alphabet = alphabet

        self.dir_mnist = os.path.join(self.root, self.dataset_mnist)
        
        if not self._check_exists_mnist():
            raise RuntimeError('Dataset MNIST not found.')

        if self.train:
            data_file_mnist = self.training_file_mnist
        else:
            data_file_mnist = self.test_file_mnist

        # Load the pt for MNIST
        self.data_mnist, self.labels_mnist = torch.load(os.path.join(self.dir_mnist, data_file_mnist))
        self.data_mnist = self.data_mnist[(self.labels_mnist!=0)&(self.labels_mnist!=9),:,:]
        self.labels_mnist = self.labels_mnist[(self.labels_mnist!=0)&(self.labels_mnist!=9)]
        
        # Generate Gaussian mixtures
        self.gm_var = torch.eye(2,dtype=torch.float32) * flags.GM_var
        angles = np.pi * np.arange(8) / 4
        self.gm_locs = np.array([[flags.GM_radius * np.cos(a), flags.GM_radius * np.sin(a)] for a in angles])
        self.gms = [mv.MultivariateNormal(torch.tensor(mu, dtype=torch.float32), self.gm_var) for mu in self.gm_locs]
        self.data_gm = torch.cat([d.sample((self.sample_size, )) for d in self.gms], dim=0)
        self.labels_gm = np.repeat(np.arange(1, 9, dtype=np.int32), self.sample_size)
        
        self.gauss_target_idx_mapping = self.process_gauss_labels()
        
    def process_gauss_labels(self):
        numbers_dict = { 1: [], 2: [], 3:[], 4:[], 5:[], 6:[], 7: [], 8:[]}
        for i in range(len(self.labels_gm)):
            gauss_target = self.labels_gm[i]
            numbers_dict[gauss_target].append(i)
        return numbers_dict
    

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        idx_mnist = int(np.floor(index / self.flags.data_multiplications))
        img_mnist, target_mnist = self.data_mnist[idx_mnist], int(self.labels_mnist[idx_mnist])
        
        # Randomly pick an index from the indices list
        indices_list = self.gauss_target_idx_mapping[target_mnist]
        idx = random.choice(indices_list)
        vec_gm = self.data_gm[idx]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img_mnist = Image.fromarray(img_mnist.numpy(), mode='L')

        if self.transform is not None:
            if self.transform[0] is not None:
                img_mnist = self.transform[0](img_mnist)

        if self.target_transform is not None:
            target = self.target_transform(target_mnist)
        else:
            target = target_mnist

        batch = {'mnist': img_mnist, 'gm': vec_gm}
        return batch, target

    def __len__(self):
        return len(self.data_mnist) * self.flags.data_multiplications - 1

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists_mnist(self):
        return (os.path.exists(os.path.join(self.dir_mnist,
                                            self.training_file_mnist)) and
                os.path.exists(os.path.join(self.dir_mnist,
                                            self.test_file_mnist)))
    
    @staticmethod
    def extract_gzip(gzip_path, remove_finished=False):
        print('Extracting {}'.format(gzip_path))
        with open(gzip_path.replace('.gz', ''), 'wb') as out_f, \
                gzip.GzipFile(gzip_path) as zip_f:
            out_f.write(zip_f.read())
        if remove_finished:
            os.unlink(gzip_path)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")
