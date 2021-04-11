import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST, CIFAR10, CelebA, FashionMNIST
from torchvision.transforms import Compose,Resize,ToTensor,ToPILImage
from tqdm import tqdm

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
TRAIN_BATCH_SIZE = 128
NUM_WORKERS = 2
ADAM_LR = 5e-4

N_EPOCHS = 401

scaler = torch.cuda.amp.GradScaler()

def get_dataloader(dataset,train):
    transform = Compose([Resize((32,32)),ToTensor()])
    
    if dataset not in DATASETS.keys():
        raise Exception("Choose a valid dataset from mnist")
    else:
        data = DATASETS[dataset](root=DATA_ROOT,
                                 train=train,
                                 transform=transform,
                                 download=True
        )
        return DataLoader(dataset=data,
                          batch_size=TRAIN_BATCH_SIZE,
                          shuffle=True,
                          num_workers=NUM_WORKERS,
                          pin_memory=True
        )

def train_vae(vae,dataset):
    
    data = get_dataloader(dataset,True)
    optimizer = Adam(params=vae.parameters(),lr=ADAM_LR)
    
    epoch_losses = []
    continued_training = False
    if continued_training:
        vae.load_state_dict(torch.load("./results/vae_model_mnist_5.ckpt"))
     
    for epoch in range(N_EPOCHS):
        for idx ,(img, _) in tqdm(enumerate(data), total=len(data), leave=False):
            epoch_loss = 0.0
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                img = img.to(device)
                loss = vae.vae_loss(img)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss = epoch_loss + loss.item()

            loss = loss.detach()
            img = img.detach()
            
            if idx % 25 == 0:
                torch.cuda.empty_cache()

        epoch_losses.append(epoch_losses)

        plt.plot(epoch_losses)
        plt.savefig("VAE_loss_epoch_"+str(epoch)+".png")
        
        if epoch % 50 == 0:
            torch.save(vae.state_dict(),'./results/vae_model_'+str(dataset)+"_"+str(epoch // 50)+'.ckpt')

    return epoch_losses

def main():
    dataset = 'celeba'

    vae = VAE(latent_dim=LATENT_DIM[dataset],img_shape=IMAGE_SHAPES[dataset]).to(device)
    vae.train()

    train_vae(vae,dataset)

if __name__=="__main__":
    main()
