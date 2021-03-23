import torch 
import PIL 
import numpy as np
import matplotlib.pyplot as plt 

from igebm.model import EBM
from vae.disvae.utils.modelIO import load_model
from train_vaebm import langevin_sample

with open('./results/model_version.txt','r') as f:
    ebm_model_file = f.read()

dataset = 'mnist'

vae_model_name = 'VAE_'+dataset      #Choose from VAE, beta-VAE, beta-TCVAE, factor-VAE 
vae_model_dir = os.path.join(RES_DIR,model_name)
vae_model = load_model(model_dir).to(device)
vae_model.eval()

ebm_trained = EBM()
ebm_trained.load_state_dict(torch.load(ebm_model_file))
ebm.eval()

#epsilon = torch.randn(batch_size,latent_dim)



