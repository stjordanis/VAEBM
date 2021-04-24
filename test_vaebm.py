import os
import argparse

import numpy as np
import torch 
import PIL 
from PIL import Image
import matplotlib.pyplot as plt 
import torchvision
from torchvision.datasets import MNIST, CelebA

from igebm.model import IGEBM
from vae.disvae.utils.modelIO import load_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

VAE_DIR = './vae/results/'

DATASETS = {
            'mnist': MNIST,
            # 'chairs': Chairs,
            'celeba': CelebA
}

LATENT_DIM = {
            'mnist': 10,
            # 'chairs': 32,
            'celeba': 128
}

IMAGE_SHAPES = {
            'mnist': (1,32,32),
            # 'chairs': (1,64,64),
            'celeba': (3,64,64)
}

# def process_samples(samples):
#     '''
#     Creates and saves final traversal image for samples generated.

#     Input:
#         samples (list): list of Torch.tensor, samples from VAEBM

#     Returns:
#         None
#     '''
#     for sample in samples:
#         sample = torchvision.transforms.ToPILImage()(sample[0])
    
#     im_samples = [Image.open(sample) for sample in samples]
#     widths, heights = zip(i.size for i in im_samples)
#     W = sum(widths), L = max(heights)

#     final_im = Image.new('RGB', (W,L))

#     pos = 0
#     for im_sample in im_samples:
#         final_im.paste(im_sample, (pos,0))
#         pos += im_sample.size[0]

#     final_im.save('sample_traversal.png')

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

    vae.eval()
    ebm.eval()
    log_h_prob = lambda epsilon: ebm(vae.decoder(epsilon)) + \
                                0.5 * torch.linalg.norm(vae.decoder(epsilon)-image_out,dim=1) ** 2
    samples = []
    for step in range(sampling_steps):
        noise = torch.randn_like(epsilon,device=device)
        loss = log_h_prob(epsilon)
        loss.sum().backward()

        epsilon.data.add_(noise, alpha=torch.sqrt(torch.tensor(step_size)))
        epsilon.grad.data.clamp_(-0.01,0.01)

        epsilon.data.add(epsilon.grad.data, alpha=-step_size / 2)
        epsilon.grad.detach_()
        epsilon.grad.zero_()
        
        sample_img = vae.decoder(epsilon)
        sample_img = sample_img.detach().to('cpu')
        samples.append(sample_img)
        del sample_img
        
        loss = loss.detach()
        noise = noise.detach()

    for step, sample in enumerate(samples):
        sample_pil = torchvision.transforms.ToPILImage()(sample[0])
        sample_pil.save("sample"+str(step+1)+".jpg")
    
    return 0

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--vae_type',type=str, default='VAE')
    parser.add_argument('--dataset',type=str, default='mnist')
    parser.add_argument('--batch_size',type=int, default=1)
    parser.add_argument('--step_size', type=float, default=8e-3)
    parser.add_argument('--steps', type=int, default=16)
    
    args = parser.parse_args()

    vae_type = args.vae_type
    dataset = args.dataset
    batch_size = args.batch_size
    step_size = args.step_size
    steps = args.steps 

    ebm_model_file = '/content/results/no_clamp_MNIST_14.ckpt'
    
    vae_model_name = vae_type + '_' +dataset      #Choose from VAE, beta-VAE, beta-TCVAE, factor-VAE 
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
        
