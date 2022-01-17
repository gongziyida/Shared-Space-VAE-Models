
import torch
import torch.nn as nn

class EncSVHN(nn.Module):
    def __init__(self, flags):
        super(EncSVHN, self).__init__()
        self.flags = flags
        assert not flags.factorized_representation
        self.latent_dim = flags.class_dim
        
        n_channels = (3, 32, 64, 128)
        kernels = (4, 4, 4)
        strides = (2, 2, 2)
        paddings = (1, 1, 1)
        li = []
        for i, (n, k, s, p) in enumerate(zip(n_channels[1:], kernels, strides, paddings), 1):
            li += [nn.Conv2d(n_channels[i-1], n, kernel_size=k, stride=s, padding=p), 
                   nn.ReLU(inplace=True)]
            
        self.enc = nn.Sequential(*li)
        self.enc_mu = nn.Conv2d(in_channels=128, out_channels=self.latent_dim, 
                                kernel_size=4, stride=1, padding=0)
        self.enc_var = nn.Conv2d(in_channels=128, out_channels=self.latent_dim, 
                                 kernel_size=4, stride=1, padding=0)
        
    def forward(self, x):
        x = self.enc(x)
        # Be careful not to squeeze the batch dimension if batch size = 1
        mu = self.enc_mu(x).squeeze(-1).squeeze(-1)
        log_var = self.enc_var(x).squeeze(-1).squeeze(-1)
        return None, None, mu, log_var
    
class DecSVHN(nn.Module):
    def __init__(self, flags):
        super(DecSVHN, self).__init__()  
        self.flags = flags
        assert not flags.factorized_representation
        self.latent_dim = flags.class_dim
        n_channels = (self.latent_dim, 128, 64, 32, 3)
        kernels = (4, 4, 4, 4)
        strides = (1, 2, 2, 2)
        paddings = (0, 1, 1, 1)
        li = []
        for i, (n, k, s, p) in enumerate(zip(n_channels[1:], kernels, strides, paddings), 1):
            li += [nn.ConvTranspose2d(n_channels[i-1], n, kernel_size=k, stride=s, padding=p), 
                   nn.ReLU(inplace=True)]
        li[-1] = nn.Sigmoid()
        
        self.dec = nn.Sequential(*li)
        
    def forward(self, _, z):
        z = z.view(z.shape[0], z.shape[1], 1, 1)
        x_hat = self.dec(z)
        return x_hat, torch.tensor(0.75).to(z.device)
