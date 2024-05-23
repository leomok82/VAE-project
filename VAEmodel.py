import torch
import torch.nn as nn
from torchsummary import summary

class VAE(nn.Module):
    def __init__(self, latent_dim = 64, channel_size = 9,latent_pixel_size = 16):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            
            nn.Conv3d(1, 8, kernel_size=(5,6,6), stride=(1,2,2), padding=(2,2,2)),  # Example for 3D conv layer
            nn.LeakyReLU(),
            nn.Conv3d(8, 4, kernel_size=(3,4,4), stride=(1,2,2), padding=1),
            nn.LeakyReLU(),
            nn.Conv3d(4,1, kernel_size=(3,4,4), stride=(1,2,2), padding=1),
        
        
            nn.LeakyReLU(),
    

            nn.Flatten(),
            nn.Linear(latent_pixel_size*latent_pixel_size*channel_size,256),
            nn.LeakyReLU(),
            nn.Dropout(),
            
        )
        ### Latent space transformations

        self.mu = nn.Linear(256,latent_dim)
        self.logvar = nn.Linear(256,latent_dim)
        
        
        self.decoder = nn.Sequential(
            
            nn.Linear(latent_dim, 256),

            nn.LeakyReLU(),
            nn.Linear(256,latent_pixel_size*latent_pixel_size*channel_size),
            nn.Unflatten(1,(1,channel_size,latent_pixel_size,latent_pixel_size)),

            nn.ConvTranspose3d(1,4, kernel_size = (3,4,4), stride= (1,2,2), padding = 1),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(4, 8, kernel_size=(3,4,4), stride=(1,2,2), padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(8, 1, kernel_size=(5,6,6), stride=(1,2,2), padding=2),

            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        
        return mu, logvar
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # Calculate standard deviation from log variance
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        z = self.decode(z)
        assert z.shape == x.shape
        return z, mu, logvar
