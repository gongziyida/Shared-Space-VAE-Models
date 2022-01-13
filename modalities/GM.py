
import torch
import numpy as np

from modalities.Modality import Modality

from utils import utils
from utils.save_samples import write_samples_img_to_file


class GM(Modality):
    def __init__(self, name, enc, dec, class_dim, style_dim, lhood_name):
        super().__init__(name, enc, dec, class_dim, style_dim, lhood_name);
        self.data_size = torch.Size((1, 2));
        self.gen_quality_eval = True;
        self.file_suffix = '.npy';


    def save_data(self, d, fn, args):
        np.save(fn, d)


    def plot_data(self, d):
        return 0
