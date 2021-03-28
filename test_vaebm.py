import os
import numpy as np
import torch 
import PIL 
import matplotlib.pyplot as plt 
import torchvision

from igebm.model import IGEBM
from vae.disvae.utils.modelIO import load_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

VAE_DIR = './vae/results/'
 
LD_TEST_N_STEPS = 16
LD_TEST_STEP_SIZE = 8e-5

TEST_BATCH_SIZE = 32


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
    z = torch.randn(batch_size,vae.latent_dim,device=device,requires_grad=False)
    epsilon = torch.randn(batch_size,vae.latent_dim,device=device,requires_grad=True)
    z.detach_()
    image_out = vae.decoder(z)
    image_out.detach_()
    image_out_ref = torchvision.transforms.ToPILImage()(image_out[5])
    image_out_ref.save("geeks_ref.jpg")
    vae.eval()
    ebm.eval()
    h_prob_dist_test = lambda epsilon: torch.exp(-ebm(vae.decoder(epsilon))) * torch.exp(-0.5 * (torch.linalg.norm(vae.decoder(epsilon)-image_out,dim=1)) ** 2)    #Confirm second term

    for _ in range(sampling_steps):
        # print((batch_size,image_out.shape))
        noise = torch.randn(epsilon.shape,device=device)
        loss = h_prob_dist_test(epsilon)
        loss.sum().backward()

        epsilon.data.add_(noise, alpha=torch.sqrt(torch.tensor(step_size)))
        epsilon.grad.data.clamp_(-0.01,0.01)

        epsilon.data.add(epsilon.grad.data, alpha=-step_size / 2)
        epsilon.grad.detach_()
        epsilon.grad.zero_()
        epsilon.data.clamp_(0, 1)
        
        loss = loss.detach()
        noise = noise.detach()

    # image_out.requires_grad = False
    return vae.decoder(epsilon)


if __name__ == '__main__':
        
    # with open('./results/model_version.txt','r') as f:
    #     ebm_model_file = f.read()
    ebm_model_file = 'ebm_model9.ckpt'
    
    dataset = 'mnist'
    vae_model_name = 'VAE_'+dataset      #Choose from VAE, beta-VAE, beta-TCVAE, factor-VAE 
    vae_model_dir = os.path.join(VAE_DIR,vae_model_name)
    vae = load_model(vae_model_dir).to(device)
    vae.eval()

    ebm = IGEBM()
    ebm.load_state_dict(torch.load(os.path.join('./results',ebm_model_file)))
    ebm = ebm.to(device)
    ebm.eval()

    image_out = langevin_sample_image(vae, ebm)

    image_out = torchvision.transforms.ToPILImage()(image_out[5])
    image_out = image_out.save("geeks.jpg")
