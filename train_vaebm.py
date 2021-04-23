#Code for training VAEBM 
import os 
import sys
import argparse

import torch
import hamiltorch
import torchvision
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CelebA
from tqdm import tqdm
import matplotlib.pyplot as plt

from vae.disvae.training import Trainer
from vae.disvae.utils.modelIO import load_model
from Langevin_dynamics.langevin_sampling.SGLD import SGLD
from igebm.model import IGEBM
from igebm.train import SampleBuffer

from datasets import Chairs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

VAE_DIR = './vae/results/'

HMC_N_STEPS = 25
HMC_STEP_SIZE = 0.3093

scaler = torch.cuda.amp.GradScaler()

def load_data(dataset, **kwargs):
    """
    Load the specified dataset for training.                #Need to train VAE on LSUN, CIFAR10, CIFAR100
    Parameters--->
        dataset (str): dataset specification ("CIFAR-10", "MNIST")
    Returns--->
        data_loader: torch DataLoader object for the dataset
    """
    
    dataset = dataset.upper().replace(" ","")
    transform = torchvision.transforms.ToTensor()   #Define custom based on different datasets 

    if dataset in ['MNIST','CELEBA','CHAIRS']:
        
        if dataset == 'MNIST':
            trainset = MNIST(root='./data', transform=transform)
        if dataset == 'CelebA':
            trainset = CelebA(root='./data',transform=transform)
        if dataset == 'CHAIRS':
            trainset = Chairs(root='./data/chairs')
        

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=kwargs['batch_size'],
                                                shuffle=True, pin_memory=True, num_workers=kwargs['num_workers'])

        return trainloader

    else:
        raise Exception('Dataset not available -- choose from MNIST, CelebA, Chairs')


def langevin_sample(vae, ebm, **kwargs):
    """
    Sample epsilon using Langevin dynamics based MCMC, 
    for reparametrizing negative phase sampling in EBM

    Parameters--> 
        vae (torch.nn.module): VAE model used in VAEBM
        ebm (torch.nn.module : EBM model used in VAEBM
        **kwargs (dict): 
            batch_size (int): batch size of data, default: 
            sampling_steps (int): number of sampling steps in MCMC
            step_size (int): step size in sampling 
    Returns-->
        epsilon (torch.Tensor): epsilon sample
    """
    vae.eval()
    ebm.eval()

    epsilon = torch.randn(kwargs['batch_size'],vae.latent_dim,device=device,requires_grad=True)
    
    log_h_eps = lambda eps: ebm(vae.decoder(eps)) + 0.5 * (torch.linalg.norm(eps,dim=1) ** 2)

    for _ in range(kwargs['sample_steps']):
        noise = torch.randn(kwargs['batch_size'],vae.latent_dim,device=device)
        loss = log_h_eps(epsilon)
        loss.sum().backward()

        epsilon.grad.data.clamp_(-0.01,0.01)

        epsilon.data.add(epsilon.grad.data, alpha=-0.5*kwargs['sample_step_size'])
        epsilon.data.add_(noise, alpha=torch.sqrt(torch.tensor(kwargs['sample_step_size'])))

        epsilon.grad.detach_()
        epsilon.grad.zero_()
        
        loss = loss.detach()
        noise = noise.detach()

    epsilon = epsilon.detach()
    return epsilon
        

def hamiltonian_sample(vae, ebm, **kwargs):
    
    """
    Uses Hamiltorch library for Hamiltonian MC sampling.
    Parameters-->
        vae (torch.nn.module) : VAE model used in VAEBM
        ebm (torch.nn.module) : EBM model used in VAEBM
        batch_size (int): batch size of data, default: 
        
    """

    hamiltorch.set_random_seed(123)
    epsilon = torch.randn(kwargs['batch_size'],vae.latent_dim,device=device)
    
    epsilon.requires_grad = True
    vae.eval()
    ebm.eval()

    h_prob_dist = lambda eps: torch.exp(-ebm(vae.decoder(eps))) * torch.exp(-0.5 * (torch.linalg.norm(eps,dim=1) ** 2))

    epsilon_hmc = hamiltorch.sample(log_prob_func=h_prob_dist.log_prob(epsilon), params_init=epsilon, num_samples=vae.latent_dim,
                               step_size=kwargs['sample_step_size'], num_steps_per_sample=kwargs['sample_steps'])
    
    return epsilon_hmc

    

def train_vaebm(vae, ebm, dataset, **kwargs):
    """
    Train the VAEBM model, with a pre-trained VAE.
    Parameters--->
        vae (torch.nn.module): VAE model used in the VAEBM
        ebm (torch.nn.module): EBM model used in the VAEBM
        dataset (torch.utils.DataLoader): dataset used for training
    Returns--->
        epoch_losses (list of ints): Losses in all epochs of training
    """

    vae.eval()    
    ebm.train()
    
    alpha_e = kwargs['l2_reg_weight']
    alpha_n = kwargs['spectral_norm_weight']

    data = load_data(dataset)
    if dataset == 'celeba':
        buffer = SampleBuffer()

    optimizer = Adam(params=ebm.parameters(),lr=kwargs['train_step_size'])
    
    for epoch in range(kwargs['train_steps']):
        
        for _ ,(pos_image, _) in tqdm(enumerate(data), total=len(data), leave=False):
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                pos_image = pos_image.to(device)
                
                epsilon = langevin_sample(
                    vae=vae,ebm=ebm,
                    batch_size=kwargs['batch_size'], 
                    sampling_steps=kwargs['sample_steps'],
                    step_size=kwargs['sample_step_size']
                )

                with torch.no_grad():
                    neg_image = vae.decoder(epsilon)

                pos_energy = ebm(pos_image)
                neg_energy = ebm(neg_image)
                energy_loss = pos_energy - neg_energy
                energy_reg_loss =  pos_energy ** 2 + neg_energy ** 2
                spectral_norm_loss = ebm.spec_norm()
                loss = (energy_loss + alpha_e * energy_reg_loss).mean() + alpha_n * spectral_norm_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pos_image = pos_image.detach()
            neg_image = neg_image.detach()
            pos_energy = pos_energy.detach()
            neg_energy = neg_energy.detach()
            energy_loss = energy_loss.detach()
            energy_reg_loss = energy_reg_loss.detach()
            spectral_norm_loss = spectral_norm_loss.detach()
            epsilon = epsilon.detach()
            loss = loss.detach()
            
            torch.cuda.empty_cache()
            
        torch.save(ebm.state_dict(),'./results/ebm_model_'+str(dataset)+"_"+str(epoch)+'.ckpt')
    
    return 0


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--num_workers',type=int, default=2)
    parser.add_argument('--dataset',type=str, default='mnist')
    parser.add_argument('--batch_size',type=int, default=32)

    parser.add_argument('--l2_reg_weight', type=float, default=1.0)
    parser.add_argument('--spectral_norm_weight', type=float, default=0.2)

    parser.add_argument('--sampling_type',type=str, default='langevin')
    parser.add_argument('--sample_step_size', type=float, default=8e-5)
    parser.add_argument('--sample_steps', type=int, default=10)

    parser.add_argument('--train_step_size', type=float, default=4e-5)
    parser.add_argument('--train_steps', type=int, default=15)
    
    args = parser.parse_args()

    vae_model_name = 'VAE_'+args.dataset      #Choose from VAE, beta-VAE, beta-TCVAE, factor-VAE 
    vae_model_dir = os.path.join(VAE_DIR,vae_model_name)

    vae = load_model(vae_model_dir).to(device)
    vae.eval()

    ebm = IGEBM().to(device)
    ebm.train()

    train_vaebm(
        vae=vae,ebm=ebm,dataset=args.dataset,
        alpha_e=args.l2_reg_weight, alpha_n=args.spectral_norm_weight,
        sample_batch_size=args.sample_batch_size, sample_steps=args.sample_steps, 
        sample_step_size=args.sample_step_size, train_steps=args.train_steps, 
        train_step_size=args.train_step_size, train_batch_size=args.train_batch_size
    )

#sample_batch_size,
#sample_steps, sample_step_size, train_steps,
#train_step_size, train_batch_size,
