#Code for training VAEBM 
import os 
import sys
import argparse

import torch
import torchvision
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, CenterCrop, Resize

from tqdm import tqdm
import matplotlib.pyplot as plt

from vae.disvae.training import Trainer
from vae.disvae.utils.modelIO import load_model
from Langevin_dynamics.langevin_sampling.SGLD import SGLD
from igebm.model import IGEBM

from datasets import Chairs, CelebA

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scaler = torch.cuda.amp.GradScaler()

VAE_DIR = './vae/results/'
ROOT_DIR = '/content/gdrive/MyDrive/results/'


def load_data(dataset, **kwargs):
    """
    Load the specified dataset for training.                #Need to train VAE on LSUN, CIFAR10, CIFAR100
    Parameters--->
        dataset (str): dataset specification ("CIFAR-10", "MNIST")
    Returns--->
        data_loader: torch DataLoader object for the dataset
    """
    
    dataset = dataset.upper().replace(" ","")
    transform = ToTensor()   #Define custom based on different datasets 
    
    if dataset in ['MNIST','CELEBA','CHAIRS']:
        
        if dataset == 'MNIST':
            trainset = MNIST(root='./data', transform=transform)
        if dataset == 'CELEBA':
            trainset = CelebA(root='./data/celeba')
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
    
    alpha_e = kwargs['alpha_e']
    alpha_n = kwargs['alpha_n']

    data = load_data(
        dataset, 
        batch_size=kwargs['batch_size'], 
        num_workers=kwargs['num_workers']
    )

    optimizer = Adam(params=ebm.parameters(),lr=kwargs['train_step_size'])
    
    for epoch in range(kwargs['train_steps']):
        iterator = tqdm(enumerate(data), total=len(data))
        for idx ,(pos_image, _) in iterator:
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                pos_image = pos_image.to(device)
                
                if kwargs['sample_type'] == 'lang':
                    epsilon = langevin_sample(
                        vae=vae,ebm=ebm,
                        batch_size=kwargs['batch_size'], 
                        sample_steps=kwargs['sample_steps'],
                        sample_step_size=kwargs['sample_step_size']
                    )
                else:
                    raise Exception('Please choose a valid option from lang')

                '''elif kwargs['sample_type'] == 'hmc' \
                        or kwargs['sample_type'] == 'rmhmc':
                    epsilon = hamiltonian_sample(
                        vae=vae,ebm=ebm,
                        sample_type=kwargs['sample_type'],
                        batch_size=kwargs['batch_size'], 
                        sample_steps=kwargs['sample_steps'],
                        sample_step_size=kwargs['sample_step_size']
                       )'''
                
                

                with torch.no_grad():
                    neg_image = vae.decoder(epsilon)

                pos_energy = ebm(pos_image)
                neg_energy = ebm(neg_image)
                energy_loss = pos_energy - neg_energy
                energy_reg_loss =  pos_energy * 2 + neg_energy * 2
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
            
            if dataset == 'chairs':
                if idx == 2697:
                    iterator.close()
                    break
            
            if dataset == 'celeba':
                if idx == 6330:
                    iterator.close()
                    break
        
        torch.save(
            ebm.state_dict(),
            ROOT_DIR+kwargs['vae_type']+'_'+str(dataset)+"_"+str(epoch)+'.ckpt'
        )
    
    return 0


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--vae_type',type=str, default='VAE', help='Choose from VAE, factor, btcvae')
    
    parser.add_argument('--num_workers',type=int, default=2)
    parser.add_argument('--dataset',type=str, default='mnist', help='Dataset: mnist, chairs, celeba')
    parser.add_argument('--batch_size',type=int, default=32)

    parser.add_argument('--l2_reg_weight', type=float, default=1.0)
    parser.add_argument('--spectral_norm_weight', type=float, default=0.2)

    parser.add_argument('--sample_type',type=str, default='lang', help='Type of sampling: lang, hmc, rmhmc')
    parser.add_argument('--sample_step_size', type=float, default=8e-5)
    parser.add_argument('--sample_steps', type=int, default=10)

    parser.add_argument('--train_step_size', type=float, default=4e-5)
    parser.add_argument('--train_steps', type=int, default=15)
    
    args = parser.parse_args()

    vae_model_name = args.vae_type + '_' + args.dataset      #Choose from VAE, factor-VAE 
    vae_model_dir = os.path.join(VAE_DIR,vae_model_name)

    vae = load_model(vae_model_dir).to(device)
    vae.eval()

    ebm = IGEBM(dataset=args.dataset).to(device)
    ebm.train()

    train_vaebm(
        vae=vae,ebm=ebm,
        dataset=args.dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        alpha_e=args.l2_reg_weight, alpha_n=args.spectral_norm_weight, 
        sample_type=args.sample_type, vae_type=args.vae_type,
        sample_steps=args.sample_steps, sample_step_size=args.sample_step_size, 
        train_steps=args.train_steps, train_step_size=args.train_step_size
    )
