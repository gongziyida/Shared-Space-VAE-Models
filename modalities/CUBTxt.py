
import torch
import numpy as np

from modalities.Modality import Modality

from utils import utils
from utils.save_samples import write_samples_img_to_file


class CUBTxt(Modality):
    def __init__(self, name, enc, dec, class_dim, style_dim, lhood_name, sent_len):
        super().__init__(name, enc, dec, class_dim, style_dim, lhood_name)
        self.data_size = torch.Size([sent_len])
        self.gen_quality_eval = True
        self.file_suffix = '.npy'


    def save_data(self, d, fn, args):
        np.save(fn, d)


    def plot_data(self, d):
        return 0
    
    def calc_log_prob(self, out_dist, target, norm_value):
        idx = target.view(-1).to(torch.long)
        target = torch.nn.functional.one_hot(idx, num_classes=out_dist.logits.shape[1])
        log_prob = out_dist.log_prob(target.to(out_dist.logits.device)).sum()
        mean_val_logprob = log_prob/norm_value
        return mean_val_logprob