import os
import argparse
from train_vaebm import load_data
from torchvision.transforms import ToPILImage
from igebm.model import IGEBM

def run_for_model(vae, ebm, batch_size, sampling_steps, step_size):
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
    i=0
    for sample in samples[-1]:
        temp = ToPILImage()(sample)
        temp.save("./generated_samples/"+str(i)+".png")
        i=i+1


def run_for_data():
    data = load_data(
        'mnist', 
        batch_size=50, 
        num_workers=2
    )
    i=0
    for (image,_) in data:
    for img in image:
        temp = ToPILImage()(img)
        temp.save("./data_samples/"+str(i)+".png")
        i=i+1
    break

if __name__=="__main__":
    i#check if folders are present
    file_path="./data_samples"
    for file in os.listdir(file_path):
        os.remove(os.path.join(file_path,file))

    file_path="./generated_samples"
    for file in os.listdir(file_path):
        os.remove(os.path.join(file_path,file))

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


    #improve this
    ebm_model_file = '/content/gdrive/MyDrive/results/ebm_model_MNIST_14.ckpt'
    
    vae_model_name = vae_type + '_' +dataset      #Choose from VAE, beta-VAE, beta-TCVAE, factor-VAE 
    vae_model_dir = os.path.join(VAE_DIR,vae_model_name)
    vae = load_model(vae_model_dir).to(device)
    vae.eval()

    ebm = IGEBM()
    ebm.load_state_dict(torch.load(os.path.join('./results',ebm_model_file)))
    ebm = ebm.to(device)
    ebm.eval()

    run_for_model(vae, ebm, batch_size=batch_size, sampling_steps=steps, step_size=step_size)

    run_for_data()