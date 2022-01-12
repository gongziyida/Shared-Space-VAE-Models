
import torch
import torch.nn as nn

dataSize = torch.Size([1, 28, 28])

class EncMNIST(nn.Module):
    def __init__(self, flags):
        super(EncMNIST, self).__init__()
        self.flags = flags
        assert not flags.factorized_representation
        self.latent_dim = flags.class_dim
        self.dim_MNIST = 28 * 28

        self.enc1 = nn.Linear(self.dim_MNIST, 400)
        self.enc_mu = nn.Linear(400, self.latent_dim)
        self.enc_var = nn.Linear(400, self.latent_dim)

    def forward(self, x):
        x = x.view(*x.size()[:-3], -1)
        x = F.relu(self.enc1(x))
        mu = self.enc_mu(x)
        log_var = self.enc_var(x)
        return None, None, mu, log_var

class DecMNIST(nn.Module):
    def __init__(self, flags):
        super(DecMNIST, self).__init__()  
        self.flags = flags
        assert not flags.factorized_representation
        self.latent_dim = flags.class_dim
        self.dim_MNIST   = 28 * 28
        
        self.dec = nn.Sequential(nn.Linear(self.latent_dim, 400), 
                                 nn.ReLU(inplace=True), 
                                 nn.Linear(400, self.dim_MNIST),
                                 nn.Sigmoid())
        
    def forward(self, _, z):
        x_hat = self.dec(z)
        x_hat = x_hat.view(*x_hat.size()[:-1], *dataSize)
        return x_hat, torch.tensor(0.75).to(z.device)
        