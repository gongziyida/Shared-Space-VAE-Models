import torch
import torch.nn as nn

class EncImg(nn.Module):
    def __init__(self, flags):
        super(EncImg, self).__init__()
        self.flags = flags 
        assert not flags.factorized_representation
        self.latent_dim = flags.class_dim
        self.input_dim = flags.img_feature_dim

        n = (self.input_dim, 1024, 512, 256)
        li = []
        for i in range(len(n)-1):
            li += [nn.Linear(n[i], n[i+1]), nn.ELU(inplace=True)]
        self.enc = nn.Sequential(*li)
        self.enc_mu = nn.Linear(256, self.latent_dim)
        self.enc_var = nn.Linear(256, self.latent_dim)
        
    def forward(self, x):
        x = self.enc(x)
        mu = self.enc_mu(x)
        log_var = self.enc_var(x)
        return None, None, mu, log_var

class DecImg(nn.Module):
    def __init__(self, flags):
        super(DecImg, self).__init__()
        self.flags = flags 
        assert not flags.factorized_representation
        self.latent_dim = flags.class_dim
        self.output_dim = flags.img_feature_dim
        
        n = (self.output_dim, 1024, 512, 256)
        li = [nn.Linear(self.latent_dim, 256)]
        for i in range(len(n)-1, 0, -1):
            li += [nn.ELU(inplace=True), nn.Linear(n[i], n[i-1])]
        self.dec = nn.Sequential(*li)
        
    def forward(self, _, z):
        x_hat = self.dec(z)
        return x_hat, torch.tensor(0.75).to(z.device)