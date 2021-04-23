import os
import argparse

import numpy as np
import torch 
import PIL 
import matplotlib.pyplot as plt 
import torchvision
from torchvision.datasets import MNIST, CIFAR10, CelebA, FashionMNIST

from igebm.model import IGEBM
from vae.disvae.utils.modelIO import load_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

VAE_DIR = './vae/results/'

DATASETS = {
            'mnist': MNIST,
            'chairs': Chairs,
            'celeba': CelebA
}

LATENT_DIM = {
            'mnist': 10,
            'chairs': 32,
            'celeba': 128
}

IMAGE_SHAPES = {
            'mnist': (1,32,32),
            'chairs': (1,64,64),
            'celeba': (3,64,64)
}

def langevin_sample_image(vae, ebm, batch_size, sampling_steps, step_size):
    """
    Sample output image using Langevin dynamics based MCMC, 
    from VAEBM.

    Parameters--> 
        vae (torch.nn.module) : VAE model used in VAEBM
        ebm (torch.nn.module) : EBM model used in VAEBM
        batch_size (int): batch size of images, default: BATCH_SIZE
        sampling_steps (int): number of sampling steps in MCMC, default: LD_N_STEPS
        step_size (int): step size in sampling 

    Returns-->
        image_out (torch.Tensor): image sample
    """
    epsilon = torch.randn(batch_size, vae.latent_dim, requires_grad=True, device=device)
    image_out = vae.decoder(epsilon)
    image_out.detach_()
    image_out_pil = torchvision.transforms.ToPILImage()(image_out[0])
    image_out_pil.save("initial.jpg")

    vae.eval()
    ebm.eval()
    log_h_prob = lambda epsilon: ebm(vae.decoder(epsilon)) + \
                                0.5 * torch.linalg.norm(vae.decoder(epsilon)-image_out,dim=1) ** 2

    for step in range(sampling_steps):
        noise = torch.randn_like(epsilon,device=device)
        loss = log_h_prob(epsilon)
        loss.sum().backward()

        epsilon.data.add_(noise, alpha=torch.sqrt(torch.tensor(step_size)))
        epsilon.grad.data.clamp_(-0.01,0.01)

        epsilon.data.add(epsilon.grad.data, alpha=-step_size / 2)
        epsilon.grad.detach_()
        epsilon.grad.zero_()
        # epsilon.data.clamp_(0, 1)
        
        # print(torch.linalg.norm(epsilon))
        # sample_img = vae.decoder(epsilon.data)
        # sample_img.detach_()
        # sample_pil = torchvision.transforms.ToPILImage()(image_out[0])
        # sample_pil.save("sample"+str(step+1)+".jpg")

        loss = loss.detach()
        noise = noise.detach()

    image_final = vae.decoder(epsilon)
    image_final.detach_()
    image_final_pil = torchvision.transforms.ToPILImage()(image_final[0])
    image_final_pil.save("final.jpg")

    return 0

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset',type=str, default='mnist')
    parser.add_argument('--batch_size',type=int, default=1)
    parser.add_argument('--step_size', type=float, default=8e-3)
    parser.add_argument('--steps', type=int, default=16)
    
    args = parser.parse_args()

    dataset = args.dataset
    batch_size = args.batch_size
    step_size = args.step_size
    steps = args.steps 

    ebm_model_file = '/content/results/no_clamp_MNIST_14.ckpt'
    
    vae_model_name = "VAE_"+dataset      #Choose from VAE, beta-VAE, beta-TCVAE, factor-VAE 
    vae_model_dir = os.path.join(VAE_DIR,vae_model_name)
    vae = load_model(vae_model_dir).to(device)
    vae.eval()

    ebm = IGEBM()
    ebm.load_state_dict(torch.load(os.path.join('./results',ebm_model_file)))
    ebm = ebm.to(device)
    ebm.eval()

    langevin_sample_image(vae, ebm, batch_size=batch_size, sampling_steps=steps, step_size=step_size)

if __name__ == '__main__':
    main()
        
