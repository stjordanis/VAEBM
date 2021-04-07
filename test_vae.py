import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST, CIFAR10, CelebA, FashionMNIST
from torchvision.transforms import Compose,Resize,ToTensor,ToPILImage
from tqdm import tqdm
import os
from vanilla_vae import VAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

DATA_ROOT = './data'
TEST_BATCH_SIZE = 128
NUM_WORKERS = 2

def main():
    dataset = 'fashion'

    vae_model_file = 'vae_model8.ckpt'

    vae = VAE(latent_dim=LATENT_DIM[dataset],img_shape=IMAGE_SHAPES[dataset])
    vae.load_state_dict(torch.load(os.path.join('./drive/Mydrive/fashion/result',vae_model_file)))
    vae = vae.to(device)
    vae.eval()
    for i in range(4):
        epsilon = torch.randn(TEST_BATCH_SIZE,vae.latent_dim,device=device)
        initial = vae.decoder(epsilon)
        image_out = torchvision.transforms.ToPILImage()(initial[26])
        image_out = image_out.save("initial" + str(i) + ".jpg")
    

if __name__=="__main__":
    main()
