#Code for training VAEBM 
import os 

import torch
import hamiltorch
import torchvision
from torch.optim import Adam
from torch.utils.data import DataLoader

import tqdm 
import numpy as np 
import matplotlib.pyplot as plt

from vae.disvae.training import Trainer
from vae.disvae.utils.modelIO import load_model, load_metadata

from igebm.model import IGEBM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

RES_DIR = './vae/results/'

LD_N_STEPS = 10
LD_STEP_SIZE = 8e-5

HMC_N_STEPS = 25
HMC_STEP_SIZE = 0.3093

BATCH_SIZE = 32
N_EPOCHS = 16
ADAM_LR = 3e-5


def load_data(dataset):
    """
    Load the specified dataset for training.                #Need to train VAE on LSUN, CIFAR10, CIFAR100

    Parameters--->
        dataset (str): dataset specification ("CIFAR-10", "MNIST")

    Returns--->
        data_loader: torch DataLoader object for the dataset
    """
    
    dataset = dataset.upper().replace(" ","")
    transform = torchvision.transforms.ToTensor()   #Define custom based on different datasets 

    if dataset in ['MNIST','CIFAR10','CIFAR100','CelebA','LSUN']:
        
        if dataset == 'MNIST':
            trainset = torchvision.datasets.MNIST(root='./data', transform=transform)
        if dataset == 'CIFAR10':
            trainset = torchvision.datasets.CIFAR10(root='./data', transform=transform)
        if dataset == 'CIFAR100':
            trainset = torchvision.datasets.CIFAR100(root='./data', transform=transform)
        if dataset == 'CelebA':
            trainset = torchvision.datasets.CelebA(root='./data',transform=transform)
        if dataset == 'LSUN':
            trainset = torchvision.datasets.LSUN(root='./data', transform=transform)
        

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

        return trainloader

    else:
        raise Exception('Dataset not available -- choose from MNIST, CIFAR10, CIFAR100, CelebA, LSUN')


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
        

def hamiltonian_sample(vae, ebm, latent_dim, batch_size=BATCH_SIZE, sampling_steps=HMC_N_STEPS, step_size=HMC_STEP_SIZE):
    
    """
    Not sure about this yet.    
    """

    hamiltorch.set_random_seed(123)
    epsilon = torch.randn(batch_size,latent_dim)
        
    #These constants copied from documentation

    epsilon.requires_grad = True
    vae.parameters.requires_grad = False
    ebm.parameters.requires_grad = False

    h_prob_dist = lambda eps: torch.exp(-ebm(vae.decoder(eps))) * torch.exp(-0.5 * (torch.linalg.norm(eps,dim=1) ** 2))

    epsilon_hmc = hamiltorch.sample(log_prob_func=h_prob_dist.log_prob(epsilon), params_init=epsilon, num_samples=latent_dim,
                               step_size=step_size, num_steps_per_sample=sampling_steps)
    
    return epsilon_hmc

    

def train_vaebm(vae,ebm,dataset):
    """
    Train the VAEBM model, with a pre-trained VAE.

    Parameters--->
        vae (torch.nn.module): VAE model used in the VAEBM
        ebm (torch.nn.module): EBM model used in the VAEBM
        dataset (torch.utils.DataLoader): dataset used for training

    Returns--->
        epoch_losses (list of ints): Losses in all epochs of training
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

    return epoch_losses


def main():
    dataset = 'cifar10'

    model_name = 'vae'      #Choose from VAE, beta-VAE, beta-TCVAE, factor-VAE 
    model_dir = os.path.join(RES_DIR,model_name)

    vae_model = load_model(model_dir)
    vae_model.eval()

    ebm_model = IGEBM()
    ebm_model.train()

    train_vaebm(vae_model,ebm_model,dataset)














    
    
    







