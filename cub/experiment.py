import os

import random
import numpy as np 

import PIL.Image as Image
from PIL import ImageFont

import torch
from torchvision import transforms
import torch.optim as optim
from sklearn.metrics import accuracy_score

#from utils.BaseExperiment import BaseExperiment

from modalities.CUBImg import CUBImg
from modalities.CUBTxt import CUBTxt

from cub.CUBDataset import CUBDataset
from cub.networks.VAEbimodalCUB import VAEbimodalCUB
from cub.networks.ClfImg import ClfImg
from cub.networks.ClfTxt import ClfTxt

from cub.networks.Img import EncImg, DecImg
from cub.networks.Txt import EncTxt, DecTxt

from utils.BaseExperiment import BaseExperiment


class CUB(BaseExperiment):
    def __init__(self, flags, *args):
        super().__init__(flags)
        self.plot_img_size = torch.Size((3, 28, 28))
        self.font = ImageFont.truetype('FreeSerif.ttf', 38)

        self.modalities = self.set_modalities()
        self.num_modalities = len(self.modalities.keys())
        self.subsets = self.set_subsets()
        self.dataset_train = None
        self.dataset_test = None
        self.set_dataset()

        self.mm_vae = self.set_model()
        self.clfs = self.set_clfs()
        self.optimizer = None
        self.rec_weights = self.set_rec_weights()
        self.style_weights = self.set_style_weights()

        self.test_samples = self.get_test_samples()
        self.eval_metric = accuracy_score 
        self.paths_fid = self.set_paths_fid()
        
        self.labels = []


    def set_model(self):
        model = VAEbimodalCUB(self.flags, self.modalities, self.subsets)
        model = model.to(self.flags.device)
        return model


    def set_modalities(self):
        mod1 = CUBImg('cub_img', EncImg(self.flags), DecImg(self.flags),
                      self.flags.class_dim, self.flags.style_m1_dim, self.flags.likelihood_m1)
        mod2 = CUBTxt('cub_txt', EncTxt(self.flags), DecTxt(self.flags),
                      self.flags.class_dim, self.flags.style_m2_dim, self.flags.likelihood_m2,
                      self.flags.txt_len)
        mods = {mod1.name: mod1, mod2.name: mod2}
        return mods

    def set_dataset(self):
        transforms = lambda data: torch.Tensor(data)
        train = CUBDataset(self.flags,
                           split='train',
                           device=self.flags.device,
                           transform=transforms)
        test = CUBDataset(self.flags, 
                          split='test', 
                          device=self.flags.device,
                          transform=transforms)
        self.dataset_train = train
        self.dataset_test = test


    def set_clfs(self):
        model_clf_m1 = None
        model_clf_m2 = None
        if self.flags.use_clf:
            model_clf_m1 = ClfImg()
            model_clf_m1.load_state_dict(torch.load(os.path.join(self.flags.dir_clf,
                                                                 self.flags.clf_save_m1)))
            model_clf_m1 = model_clf_m1.to(self.flags.device)

            model_clf_m2 = ClfTxt(self.flags.Txt_radius, self.flags.Txt_var)
            model_clf_m2 = model_clf_m2.to(self.flags.device)

        clfs = {'cub_img': model_clf_m1,
                'cub_txt': model_clf_m2}
        return clfs


    def set_optimizer(self):
        # optimizer definition
        optimizer = optim.Adam(
            list(self.mm_vae.parameters()),
            lr=self.flags.initial_learning_rate,
            betas=(self.flags.beta_1, self.flags.beta_2), 
            amsgrad=True)
        self.optimizer = optimizer


    def set_rec_weights(self):
        rec_weights = dict()
        ref_mod_d_size = self.modalities['cub_img'].data_size.numel()
        for k, m_key in enumerate(self.modalities.keys()):
            mod = self.modalities[m_key]
            numel_mod = mod.data_size.numel()
            rec_weights[mod.name] = float(ref_mod_d_size/numel_mod) if self.flags.reweight_rec else 1
        return rec_weights


    def set_style_weights(self):
        weights = dict()
        weights['cub_img'] = self.flags.beta_m1_style
        weights['cub_txt'] = self.flags.beta_m2_style
        return weights


    def get_test_samples(self):
        return []


    def mean_eval_metric(self, values):
        return np.mean(np.array(values))


    def get_prediction_from_attr(self, attr, index=None):
        pred = np.argmax(attr, axis=1).astype(int)
        return pred


    def eval_label(self, values, labels, index):
        pred = self.get_prediction_from_attr(values)
        return self.eval_metric(labels, pred)

