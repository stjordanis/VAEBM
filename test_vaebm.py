import os
import argparse

import numpy as np
import torch 
from torch.nn.functional import mse_loss
import PIL 
from PIL import Image
import matplotlib.pyplot as plt 
import torchvision
from torchvision.datasets import MNIST, CelebA
from torchvision.transforms import ToPILImage
from igebm.model import IGEBM
from vae.disvae.utils.modelIO import load_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

VAE_DIR = './vae/results/'
RESULT_DIR = '/content/gdrive/MyDrive/cs698x_samples'
ROOT_DIR = '/content/gdrive/MyDrive/results/'

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

def generate_samples(samples, batch_size, **kwargs):
    final_samples = samples[-1]

    side_n = int(np.sqrt(batch_size))
    w, l = samples[0].shape[2], samples[0].shape[3]
    W, L = side_n * w, side_n * l
    
    final_im = Image.new('RGB', (W,L))

    for i in range(side_n):
        for j in range(side_n):
            final_im.paste(ToPILImage()(final_samples[side_n*i + j]), (i*w, j*l))

    final_im.save(RESULT_DIR+'/samples_'+kwargs['vae_type']+"_"+kwargs['dataset']+"_"+str(kwargs['step_size'])+'.png')

def traverse_samples(samples, batch_size, **kwargs):
    '''
    Creates and saves final traversal image for samples generated.

    Input:
        samples (list): list of Torch.tensor, samples from VAEBM

    Returns:
        None
    '''
    W = len(samples) * samples[0].shape[2]
    L = batch_size * samples[0].shape[3]
    # print(W,L)

    final_im = Image.new('RGB', (W,L))
    
    x_pos = 0

    for sample_step in samples:
        y_pos = 0
        for sample in sample_step:
            im_sample = ToPILImage()(sample)
            final_im.paste(im_sample, (x_pos,y_pos))
            y_pos += sample_step.shape[3]
        x_pos += sample_step.shape[2]

    final_im.save(RESULT_DIR+'/traversal_'+kwargs['vae_type']+"_"+kwargs['dataset']+"_"+str(kwargs['step_size'])+'.png')

def langevin_sample_image(vae, ebm, **kwargs):
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
    epsilon = torch.randn(kwargs['batch_size'], vae.latent_dim, requires_grad=True, device=device)
    image_out = vae.decoder(epsilon)
    image_out.detach_()

    vae.eval()
    ebm.eval()
    log_h_prob = lambda epsilon: ebm(vae.decoder(epsilon)) + \
                                 0.5 * mse_loss(vae.decoder(epsilon),image_out)
    # log_h_prob = lambda eps: ebm(vae.decoder(eps)) + 0.5 * (torch.linalg.norm(eps,dim=1) ** 2)
    
    samples = []
    step_size = kwargs['step_size']

    for step in range(kwargs['sampling_steps']):
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

    return samples
    
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

    ebm_model_file = ROOT_DIR + vae_type + '_' + dataset + '.ckpt'
    
    vae_model_name = vae_type + '_' +dataset      #Choose from VAE, beta-VAE, beta-TCVAE, factor-VAE 
    vae_model_dir = os.path.join(VAE_DIR,vae_model_name)
    vae = load_model(vae_model_dir).to(device)
    vae.eval()

    ebm = IGEBM(dataset=dataset)
    ebm.load_state_dict(torch.load(os.path.join('./results',ebm_model_file)))
    ebm = ebm.to(device)
    ebm.eval()

    samples = langevin_sample_image(vae, ebm, batch_size=batch_size, sampling_steps=steps, step_size=step_size)
    
    traverse_samples(
        samples, 
        batch_size=batch_size, 
        dataset=dataset,
        vae_type=vae_type,
        step_size=step_size
    )

    generate_samples(
        samples, 
        batch_size=batch_size, 
        dataset=dataset,
        vae_type=vae_type,
        step_size=step_size
    )

if __name__ == '__main__':
    main()