import numpy as np 
import torch
import torch.nn as nn

class FCVAE(nn.Module):
    def __init__(self, batch_size=1e5, latent_dim=20):
        super(FCVAE,self).__init__()
        self.latent_dim = latent_dim
        self.batch_size = batch_size

        in_layer = torch.nn.Linear(self.batch_size,256)

        encoder = nn.Sequential(
            torch.nn.Linear(256,256),
            torch.nn.Linear(256,256),
            torch.nn.Linear(256,256),
            torch.nn.Linear(256,2*self.latent_dim)
        )

        decoder = nn.Sequential(
            torch.nn.Linear(2*self.latent_dim,256),
            torch.nn.Linear(256,256),
            torch.nn.Linear(256,256),
            torch.nn.Linear(256,256)
        )

        out_layer = torch.nn.Linear(256,self.batch_size)
    
    def forward(self,x):
        mu, log_var = self.encoder(x)[:latent_dim], encoder(x)[latent_dim:] 
        generated = 0

        