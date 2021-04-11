import torch 
import torchvision
import torch.nn as nn
from torch.nn.functional import mse_loss
from torch.distributions.kl import kl_divergence
from torch.distributions import MultivariateNormal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self,latent_dim,img_shape):
        super(Encoder,self).__init__()
        
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.kernel_size = 4
        self.channels = 32
        self.fc_layer_size = 256

        input_channels = img_shape[0]

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels,
                      out_channels=self.channels,
                      kernel_size=self.kernel_size,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.ReLU(inplace=True)
        )
        
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=self.channels,
                      out_channels=self.channels,
                      kernel_size=self.kernel_size,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.ReLU(inplace=True)
        )
        
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=self.channels,
                      out_channels=self.channels,
                      kernel_size=self.kernel_size,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.ReLU(inplace=True)
        )
        
        if img_shape[1] == 64 == img_shape[2]:
            self.conv_block_4 = nn.Sequential(
                nn.Conv2d(in_channels=self.channels,
                        out_channels=self.channels,
                        kernel_size=self.kernel_size,
                        stride=2,
                        padding=1,
                        bias=False),
                nn.ReLU(inplace=True)
            )
        
        self.linear_1 = nn.Sequential(
            nn.Linear(in_features=self.channels * self.kernel_size * self.kernel_size, 
                      out_features=self.fc_layer_size),
            nn.ReLU(inplace=True)
        )

        self.linear_2 = nn.Sequential(
            nn.Linear(in_features=self.fc_layer_size, 
                      out_features=self.fc_layer_size),
            nn.ReLU(inplace=True)
        )

        self.out_layer = nn.Linear(self.fc_layer_size, 2 * self.latent_dim)


    def forward(self,img):
        conv_out_1 = self.conv_block_1(img)
        conv_out_2 = self.conv_block_2(conv_out_1)
        conv_out_last = self.conv_block_3(conv_out_2)
            
        if self.img_shape[1] == 64:
            conv_out_last = self.conv_block_4(conv_out_last)
        
        conv_out_last = conv_out_last.view(-1, self.channels * self.kernel_size * self.kernel_size)

        linear_out_1 = self.linear_1(conv_out_last)
        linear_out_2 = self.linear_2(linear_out_1)

        out_params = self.out_layer(linear_out_2)
        mu, log_var = out_params.view(-1,self.latent_dim,2).unbind(-1)
        
        return mu, log_var

class Decoder(nn.Module):
    def __init__(self,latent_dim,img_shape):
        super(Decoder,self).__init__()

        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.kernel_size = 4
        self.channels = 32
        self.fc_layer_size = 256

        self.output_channels = img_shape[0]

        self.linear_1 = nn.Sequential(
            nn.Linear(in_features=latent_dim,
                      out_features=self.fc_layer_size
            ),
            nn.ReLU(inplace=True)
        )

        self.linear_2 = nn.Sequential(
            nn.Linear(in_features=self.fc_layer_size,
                      out_features=self.fc_layer_size
            ),
            nn.ReLU(inplace=True)
        )
        
        self.linear_out = nn.Sequential(
            nn.Linear(in_features=self.fc_layer_size,
                      out_features=self.channels * self.kernel_size * self.kernel_size
            ),
            nn.ReLU(inplace=True)
        )

        if img_shape[1] == 64:
            self.conv_t_block_0 = nn.Sequential(
                nn.ConvTranspose2d(in_channels=self.channels,
                                    out_channels=self.channels,
                                    kernel_size=self.kernel_size,
                                    stride=2,
                                    padding=1
                ),
            nn.ReLU(inplace=True)
            )

        self.conv_t_block_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.channels,
                               out_channels=self.channels,
                               kernel_size=self.kernel_size,
                               stride=2,
                               padding=1
            ),
            nn.ReLU(inplace=True)
        )

        self.conv_t_block_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.channels,
                               out_channels=self.channels,
                               kernel_size=self.kernel_size,
                               stride=2,
                               padding=1
            ),
            nn.ReLU(inplace=True)
        )

        self.conv_t_block_3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.channels,
                               out_channels=self.output_channels,
                               kernel_size=self.kernel_size,
                               stride=2,
                               padding=1
            ),
            nn.Sigmoid()
        )     

    def forward(self,z):

        batch_size = z.shape[0]

        x_1 = self.linear_1(z)
        x_2 = self.linear_2(x_1)
        x = self.linear_out(x_2)

        x = x.view(batch_size,self.channels,self.kernel_size,self.kernel_size)

        if self.img_shape[1] == 64:    
            x = self.conv_t_block_0(x)
        
        x_1 = self.conv_t_block_1(x)
        x_2 = self.conv_t_block_2(x_1)
        x_out = self.conv_t_block_3(x_2)
        
        return x_out

class VAE(nn.Module):
    def __init__(self, latent_dim, img_shape, batch_size):
        super(VAE,self).__init__()

        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.beta_kl = 0.5
        self.batch_size=batch_size
    
        self.encoder = Encoder(self.latent_dim,self.img_shape)
        self.decoder = Decoder(self.latent_dim,self.img_shape)

    def reparametrize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std,device=device)

        return mean + std * eps


    def sample(self):
        z = torch.randn(self.img_shape[0],self.latent_dim,device=device)
        return self.decoder(z)

    def forward(self, x):
        mean, logvar = self.encoder(x)
        latent = self.reparametrize(mean, logvar)
        reconstructed = self.decoder(latent)

        return latent, reconstructed

    def vae_loss(self,x):
        reconstructed = self.forward(x)[1]
        recon_loss = mse_loss(reconstructed,x)
        
        mean, logvar = self.encoder(x)

        latent_kl = 0.5 * (-1 - logvar + mean.pow(2) + logvar.exp()).mean(dim=0)
        total_kl = latent_kl.sum()

        loss = (recon_loss / self.batch_size) + self.beta_kl * total_kl
        return loss
