import torch
import torch.nn as nn

__all__ = ['VAE']

class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) model.

    Args:
        latent_dim (int): The dimension of the latent space. Default is 64.
        channel_size (int): The number of channels in the input image. Default is 19.
        image_size (int): The size of the input image. Default is 256.

    Attributes:
        channel_size (int): The number of channels in the input image.
        latent_dim (int): The dimension of the latent space.
        image_size (int): The size of the input image.
        encoder (nn.Sequential): The encoder network.
        mu (nn.Linear): The linear layer for mean calculation.
        logvar (nn.Linear): The linear layer for log variance calculation.
        decoder (nn.Sequential): The decoder network.
    """

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
        """
        Encodes the input image into the latent space.

        Args:
            x (torch.Tensor): The input image tensor.

        Returns:
            tuple: A tuple containing the mean and log variance of the latent space.
        """
        x = self.encoder(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterizes the latent space using the mean and log variance.

        Args:
            mu (torch.Tensor): The mean of the latent space.
            logvar (torch.Tensor): The log variance of the latent space.

        Returns:
            torch.Tensor: The reparameterized latent space.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """
        Decodes the latent space into the reconstructed image.

        Args:
            z (torch.Tensor): The latent space tensor.

        Returns:
            torch.Tensor: The reconstructed image tensor.
        """
        x_hat = self.decoder(z)
        return x_hat.view(-1, self.channel_size, self.image_size, self.image_size)

    def forward(self, x):
        """
        Performs a forward pass of the VAE model.

        Args:
            x (torch.Tensor): The input image tensor.

        Returns:
            tuple: A tuple containing the reconstructed image, mean, and log variance of the latent space.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar