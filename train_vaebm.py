#Code for training VAEBM 
import torch
from torch.utils.data import DataLoader
import numpy as np 
import os 

from disentangling_vae.disvae import * 
from igebm_pytorch.model import *
from igebm_pytorch.train import *


def preprocess_data(dataset):
    if dataset == 'mnist':
        pass
    elif dataset == 'cifar10':
        pass
    elif dataset == 'cifar100':
        pass

    return data

DATASET = 'mnist'        #MNIST/CIFAR10/CIFAR100
VAE = 'standard_vae'   #Standard VAE, beta-VAE, ...


def train_vaebm(vae=VAE,dataset=DATASET):
    data = preprocess_data(dataset=dataset)







