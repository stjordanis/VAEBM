#Code for training VAEBM 
import os 
import sys

import torch
import hamiltorch
from torch.optim import Adam
from torch.utils.data import DataLoader

import tqdm 
import numpy as np 
import matplotlib.pyplot as plt

from vae.disvae.training import Trainer
from vae.disvae.utils.modelIO import load_model, load_metadata

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

RES_DIR = './vae/results/'

LD_N_STEPS = 10
LD_STEP_SIZE = 8e-5

BATCH_SIZE = 32
N_EPOCHS = 16
ADAM_LR = 3e-5

def load_ebm(model_dir):
    """
    Load the EBM model

    Parameters--->
        model_dir (str): location of model to be loaded

    Returns--->
        ebm_model (torch.nn.module): EBM model as required
    """
    pass

def load_data(dataset):
    """
    Load the specified dataset for training.

    Parameters--->
        dataset (str): dataset specification ("CIFAR-10", "MNIST")

    Returns--->
        data_loader: torch dataloader object for the dataset
    """
    pass

def langevin_sample(vae, ebm, latent_dim, batch_size=BATCH_SIZE, sampling_steps=LD_N_STEPS, step_size=LD_STEP_SIZE):
    """
    Sample epsilon using Langevin dynamics based MCMC, 
    for reparametrizing negative phase sampling in EBM

    Parameters--> 
        vae (torch.nn.module) : VAE model used in VAEBM
        ebm (torch.nn.module) : EBM model used in VAEBM
        batch_size (int): batch size of data, default: 
        latent_dim (int): latent dimension of the VAE in vae_decoder
        sampling_steps (int): number of sampling steps in MCMC
        step_size (int): step size in sampling 

    Returns-->
        epsilon (torch.Tensor): epsilon sample
    """

    epsilon = torch.randn(batch_size,latent_dim)
    
    epsilon.requires_grad = True
    vae.parameters.requires_grad = False
    ebm.parameters.requires_grad = False

    h_prob_dist = lambda eps: torch.exp(-ebm(vae.decoder(eps))) * torch.exp(-0.5 * (torch.linalg.norm(eps,dim=1) ** 2))

    for _ in range(sampling_steps):
        noise = torch.randn(batch_size,latent_dim)
        
        prob_out = h_prob_dist(epsilon)
        prob_out.sum().backward()

        epsilon.data.add_(noise,torch.sqrt(step_size))

        epsilon.grad.data.clamp_(-0.01,0.01)

        epsilon.data.add(-step_size / 2, epsilon.grad.data)
        epsilon.grad.detach_()
        epsilon.grad.zero_()
        #epsilon.data.clamp_(0, 1)

    epsilon.requires_grad = False
    return epsilon
        

def hamiltonian_sample(vae, ebm, latent_dim, batch_size=BATCH_SIZE, sampling_steps=LD_N_STEPS):
    
    """
    Not sure about this yet.    
    """

    hamiltorch.set_random_seed(123)
    epsilon = torch.randn(batch_size,latent_dim)
        
    #These constants copied from documentation
    step_size = 0.3093
    num_samples = 10
    L = 25

    epsilon.requires_grad = True
    vae.parameters.requires_grad = False
    ebm.parameters.requires_grad = False

    h_prob_dist = lambda eps: torch.exp(-ebm(vae.decoder(eps))) * torch.exp(-0.5 * (torch.linalg.norm(eps,dim=1) ** 2))

    epsilon_hmc = hamiltorch.sample(log_prob_func=h_prob_dist.log_prob(epsilon), params_init=epsilon, num_samples=num_samples,
                               step_size=step_size, num_steps_per_sample=L)
    
    return epsilon_hmc

    

def train_vaebm(vae,ebm,dataset):
    """
    Train the VAEBM model, with a pre-trained VAE.

    Parameters--->
        vae (torch.nn.module): VAE model used in the VAEBM
        ebm (torch.nn.module): EBM model used in the VAEBM
        dataset (torch.utils.DataLoader): dataset used for training

    Returns--->
        None
    """

    vae.parameters.requires_grad = False
    ebm.parameters.requires_grad = True
    
    data = load_data(dataset)

    optimizer = Adam(params=ebm.parameters,lr=ADAM_LR)

    for epoch in range(N_EPOCHS):
        epoch_losses=[]
        for _,pos_image in tqdm(enumerate(data)):
            
            pos_energy = ebm(pos_image)

            epsilon = langevin_sample(
                vae=vae,ebm=ebm,
                latent_dim=vae.latent_dim
            )

            neg_energy = ebm(vae(epsilon))

            loss = -pos_energy + neg_energy
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        epoch_losses.append(loss)
        
        torch.save(ebm.state_dict,'model'+str(epoch)+'.ckpt')
    
    plt.plot(epoch+1,epoch_losses)
    plt.savefig('vaebm_training_loss.jpg')



def main():
    model_name = 'vae'
    model_dir = os.path.join(RES_DIR,model_name)

    vae_model = load_model(model_dir)
    vae_model.eval()

    ebm_model = load_ebm(model_dir)
    ebm_model.train()

    train_vaebm(vae_model,ebm_model,dataset)














    
    
    







