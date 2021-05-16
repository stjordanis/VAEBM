import os
import argparse
import torch
from train_vaebm import load_data
from torch.nn.functional import mse_loss
import PIL 
from PIL import Image
import torchvision
from torchvision.transforms import ToPILImage
from igebm.model import IGEBM
from pytorch_fid import fid_score
from vae.disvae.utils.modelIO import load_model

VAE_DIR = './vae/results/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_for_model(vae, ebm, sets, batch_size, sampling_steps, step_size,paths):
    for i in range(sets):
        epsilon = torch.randn(batch_size, vae.latent_dim, requires_grad=True, device=device)
        image_out = vae.decoder(epsilon)
        #print(torch.linalg.norm(vae.decoder(epsilon)-image_out,dim=1).shape)
        image_out.detach_()

        vae.eval()
        ebm.eval()
        log_h_prob = lambda epsilon: ebm(vae.decoder(epsilon)) + \
                                    0.5 * mse_loss(vae.decoder(epsilon),image_out)
        # log_h_prob = lambda eps: ebm(vae.decoder(eps)) + 0.5 * (torch.linalg.norm(eps,dim=1) ** 2)
        
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
            
        # print(samples[0][0].shape)
        j=0
        for sample in samples[-1]:
            temp = ToPILImage()(sample)
            temp.save(os.path.join(paths[0],str(i)+"/"+str(j)+".png"))
            j=j+1
        j=0
        for sample in samples[0]:
            temp = ToPILImage()(sample)
            temp.save(os.path.join(paths[1],str(i)+"/"+str(j)+".png"))
            j=j+1



def run_for_data(batch_size,sets,dataset,path):
    data = load_data(
        dataset, 
        batch_size=batch_size, 
        num_workers=2
    )
    i=0
    for (image,_) in data:
      j=0
      for img in image:
        temp = ToPILImage()(img)
        temp.save(os.path.join(path,str(i)+"/"+str(j)+".png"))
        j=j+1
      i = i + 1
      if(i==sets):
          break

if __name__=="__main__":

    parser = argparse.ArgumentParser()


    parser.add_argument('--vae_type',type=str, default='VAE')
    parser.add_argument('--dataset',type=str, default='mnist')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--step_size', type=float, default=8e-3)
    parser.add_argument('--steps', type=int, default=16)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--dims', type=int, default=2048)
    parser.add_argument('--sets',type=int,default=10)
    parser.add_argument('path', type=str, nargs=3)
    args = parser.parse_args()

    vae_type = args.vae_type
    dataset = args.dataset
    batch_size = args.batch_size
    step_size = args.step_size
    steps = args.steps
    sets = args.sets
    path = args.path

    file_path = path[0] #"./data_samples"
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    for folder in os.listdir(file_path):
        folder_path = os.path.join(file_path,folder)
        for file in os.listdir(folder_path):
            os.remove(os.path.join(folder_path,file))
        os.rmdir(folder_path)

    for i in range(sets):
        if not os.path.exists(os.path.join(file_path,str(i))):
            os.mkdir(os.path.join(file_path,str(i)))
    
    file_path=path[1] #"./generated_samples"
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    for folder in os.listdir(file_path):
        folder_path = os.path.join(file_path,folder)
        for file in os.listdir(folder_path):
            os.remove(os.path.join(folder_path,file))
        os.rmdir(folder_path)

    for i in range(sets):
        if not os.path.exists(os.path.join(file_path,str(i))):
            os.mkdir(os.path.join(file_path,str(i)))

    
    file_path=path[2] #"./vae_samples"
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    for folder in os.listdir(file_path):
        folder_path = os.path.join(file_path,folder)
        for file in os.listdir(folder_path):
            os.remove(os.path.join(folder_path,file))
        os.rmdir(folder_path)

    for i in range(sets):
        if not os.path.exists(os.path.join(file_path,str(i))):
            os.mkdir(os.path.join(file_path,str(i)))

    #improve this
    ebm_model_file = '/content/gdrive/MyDrive/results/'+vae_type+'_'+dataset+'.ckpt'
    
    vae_model_name = vae_type + '_' +dataset      #Choose from VAE, beta-VAE, beta-TCVAE, factor-VAE 
    vae_model_dir = os.path.join(VAE_DIR,vae_model_name)
    vae = load_model(vae_model_dir).to(device)
    vae.eval()

    ebm = IGEBM(dataset=dataset)
    ebm.load_state_dict(torch.load(ebm_model_file))
    ebm = ebm.to(device)
    ebm.eval()


    run_for_model(vae, ebm, sets, batch_size=batch_size, sampling_steps=steps, step_size=step_size,paths=[path[1],path[2]])

    run_for_data(batch_size,sets,dataset,path=path[0])

    fid_value = 0
    for i in range(sets):
        path0 = os.path.join(path[0],str(i))
        path1 = os.path.join(path[1],str(i))
        fid_value += fid_score.calculate_fid_given_paths([path0,path1], batch_size,device,args.dims)
    fid_value /= sets
    print("FID Score between EBM and Data: ", fid_value)

    fid_value = 0
    for i in range(sets):
        path0 = os.path.join(path[0],str(i))
        path2 = os.path.join(path[2],str(i))
        fid_value += fid_score.calculate_fid_given_paths([path0,path2], batch_size,device,args.dims)
    fid_value /= sets
    print("FID Score between VAE and Data: ", fid_value)
