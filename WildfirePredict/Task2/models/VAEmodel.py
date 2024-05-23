import torch
import torch.nn as nn




class VAE(nn.Module):
    def __init__(self, latent_dim=64, channel_size=19, image_size=256):
        super(VAE, self).__init__()
        self.channel_size = channel_size
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channel_size * image_size * image_size, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Dropout(),
        )
        self.mu = nn.Linear(64, latent_dim)
        self.logvar = nn.Linear(64, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, channel_size * image_size * image_size),
            nn.Sigmoid(),
            nn.Unflatten(1, (channel_size, image_size, image_size)),
        )

    def encode(self, x):
        x = self.encoder(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x_hat = self.decoder(z)
        return x_hat.view(-1, self.channel_size, self.image_size, self.image_size)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar