import torch
import torch.nn as nn

class EncTxt(nn.Module):
    def __init__(self, flags):
        super(EncTxt, self).__init__()
        self.flags = flags 
        assert not flags.factorized_representation
        self.vocab_size = flags.vocab_size
        self.latent_dim = flags.class_dim
        self.emb_dim = flags.txt_emb_dim
        
        # 0 is for the excluded words and does not contribute to gradient
        self.emb = nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=0)

        n_channels = (1, 32, 64, 128, 256, 512)
        kernels = (4, 4, 4, (1, 4), (1, 4))
        strides = (2, 2, 2, (1, 2), (1, 2))
        paddings = (1, 1, 1, (0, 1), (0, 1))
        li = []
        for i, (n, k, s, p) in enumerate(zip(n_channels[1:], kernels, strides, paddings), 1):
            li += [nn.Conv2d(n_channels[i-1], n, kernel_size=k, stride=s, padding=p), 
                   nn.BatchNorm2d(n), nn.ReLU(inplace=True)]
            
        self.enc = nn.Sequential(*li)
        self.enc_mu = nn.Conv2d(512, self.latent_dim, kernel_size=4, stride=1, padding=0)
        self.enc_var = nn.Conv2d(512, self.latent_dim, kernel_size=4, stride=1, padding=0)
        
    def forward(self, x):
        x = self.emb(x.long()).unsqueeze(1) # add channel dim
        x = self.enc(x)
        mu = self.enc_mu(x).squeeze()
        log_var = self.enc_var(x).squeeze()
        return None, None, mu, log_var

class DecTxt(nn.Module):
    def __init__(self, flags):
        super(DecTxt, self).__init__()
        self.flags = flags 
        assert not flags.factorized_representation
        self.vocab_size = flags.vocab_size
        self.latent_dim = flags.class_dim
        self.emb_dim = flags.txt_emb_dim
        self.txt_len = flags.txt_len
        
        n_channels = (1, 32, 64, 128, 256, 512, self.latent_dim)
        kernels = (4, 4, 4, (1, 4), (1, 4), 4)
        strides = (2, 2, 2, (1, 2), (1, 2), 1)
        paddings = (1, 1, 1, (0, 1), (0, 1), 0)
        li = []
        for i, (n, k, s, p) in enumerate(zip(n_channels[1:], kernels, strides, paddings), 1):
            li = [nn.ConvTranspose2d(n, n_channels[i-1], kernel_size=k, stride=s, padding=p), 
                  nn.BatchNorm2d(n_channels[i-1]), nn.ReLU(inplace=True)] + li
            
        # No batchnorm at the first and last block
        del li[-2]
        del li[1]
        
        self.dec = nn.Sequential(*li)
        self.anti_emb = nn.Sequential(nn.Linear(self.emb_dim, self.vocab_size),
                                      nn.Sigmoid())
        
    def forward(self, _, z):
        z = z.view(z.shape[0], z.shape[1], 1, 1)
        z = self.dec(z)
        x_hat = self.anti_emb(z.view(-1, self.emb_dim))
        # x_hat = x_hat.view(-1, self.txt_len, self.vocab_size) # batch x txt len x vocab size
        x_hat = x_hat.view(-1, self.vocab_size)
        return (x_hat, )
