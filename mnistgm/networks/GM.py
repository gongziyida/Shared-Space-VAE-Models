
import torch
import torch.nn as nn

data_dim = 2

class EncGM(nn.Module):
    def __init__(self, flags):
        super(EncGM, self).__init__()
        self.flags = flags
        assert not flags.factorized_representation
        self.latent_dim = flags.class_dim
        self.emb_dim = flags.emb_dim
        if self.latent_dim > data_dim:
            raise ValueError('latent_dim > {data_dim}')

        self.enc = nn.Sequential(nn.Linear(data_dim, self.emb_dim),
                                 nn.LeakyReLU(inplace=True))
        self.enc_mu = nn.Linear(self.emb_dim, self.latent_dim)
        self.enc_var = nn.Linear(self.emb_dim, self.latent_dim)

    def forward(self, x):
        x = self.enc(x)
        mu = self.enc_mu(x)
        log_var = self.enc_var(x)
        return None, None, mu, log_var

class DecGM(nn.Module):
    def __init__(self, flags):
        super(DecGM, self).__init__()  
        self.flags = flags
        assert not flags.factorized_representation
        self.latent_dim = flags.class_dim
        self.emb_dim = flags.emb_dim
        if self.latent_dim > data_dim:
            raise ValueError('latent_dim > {data_dim}')
            
        self.dec = nn.Sequential(nn.Linear(self.latent_dim, self.emb_dim), 
                                 nn.LeakyReLU(inplace=True),
                                 nn.Linear(self.emb_dim, data_dim))
        
    def forward(self, _, x):
        x_hat = self.dec(x)
        return x_hat, torch.tensor(0.75).to(z.device)