#Code for training VAEBM 
import os 
import sys

import torch
from torch.utils.data import DataLoader
import numpy as np 
import tqdm 

from vae.disvae.training import Trainer
from vae.disvae.utils.modelIO import load_model

def sample_epsilon(vae_decoder, EBM, batch_size, latent_dim, sampling_steps, step_size):
    pass
    epsilon = torch.randn(batch_size,latent_dim)
    
    epsilon.requires_grad = True
    vae_decoder.parameters.requires_grad = False
    EBM.parameters.requires_grad = False

    h_prob_dist = lambda eps: torch.exp(-EBM(vae_decoder(eps))) * torch.exp(-0.5 * (torch.linalg.norm(eps,dim=1) ** 2))

    for _ in tqdm(range(sampling_steps)):
        noise = torch.randn(batch_size,latent_dim)
        
        prob_out = h_prob_dist(epsilon)
        prob_out.sum().backward()

        epsilon.data.add_(noise,torch.sqrt(step_size))

        epsilon.grad.data.clamp_(-0.01,0.01)

        epsilon.data.add(-step_size / 2, epsilon.grad.data)
        epsilon.grad.detach_()
        epsilon.grad.zero_()
        #epsilon.data.clamp_(0, 1)

    return epsilon
        


RES_DIR = './vae/results/'

model_name = 'vae'
model_dir = os.path.join(RES_DIR,model_name)
vae_model = load_model(model_dir)
vae_model.eval()









    
    
    







