import os
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
 
LD_TEST_N_STEPS = 16
LD_TEST_STEP_SIZE = 8e-5

TEST_BATCH_SIZE = 1


def langevin_sample_image(vae, ebm, batch_size=TEST_BATCH_SIZE, sampling_steps=LD_TEST_N_STEPS, step_size=LD_TEST_STEP_SIZE):
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
        epsilon.data.clamp_(0, 1)
        
        # print(torch.linalg.norm(epsilon))
        # sample_img = vae.decoder(epsilon.data)
        # sample_img.detach_()
        # sample_pil = torchvision.transforms.ToPILImage()(image_out[0])
        # sample_pil.save("sample"+str(step+1)+".jpg")

        loss = loss.detach()
        noise = noise.detach()

    return vae.decoder(epsilon)

DATASETS = {
            'mnist': MNIST,
            'cifar10': CIFAR10,
            'celeba': CelebA,
            'fashion': FashionMNIST
}

LATENT_DIM = {
            'mnist': 10,
            'cifar10': 64,
            'celeba': 128,
            'fashion': 64
}

IMAGE_SHAPES = {
            'mnist': (1,32,32),
            'cifar10': (3,32,32),
            'celeba': (3,64,64),
            'fashion': (1,28,28)
}

if __name__ == '__main__':
        
    ebm_model_file = 'ebm_model9.ckpt'
    
    dataset = 'mnist'
    vae_model_name = "VAE_"+dataset      #Choose from VAE, beta-VAE, beta-TCVAE, factor-VAE 
    vae_model_dir = os.path.join(VAE_DIR,vae_model_name)
    vae = load_model(vae_model_dir).to(device)
    vae.eval()

    ebm = IGEBM()
    ebm.load_state_dict(torch.load(os.path.join('./results',ebm_model_file)))
    ebm = ebm.to(device)
    ebm.eval()

    image_out = langevin_sample_image(vae, ebm)
    langevin_sample_image(vae, ebm)

    image_out = torchvision.transforms.ToPILImage()(image_out[0])
    image_out = image_out.save("final.jpg")
