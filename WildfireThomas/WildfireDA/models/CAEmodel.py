import torch.nn as nn

__all__ = ['CAE']


class CAE(nn.Module):
    """
    Convolutional Autoencoder (CAE) model.

    This model consists of an encoder and a decoder. The encoder takes an input image and encodes it into a lower-dimensional representation.
    The decoder takes the encoded representation and reconstructs the original image.

    Attributes:
        encoder (nn.Sequential): The encoder network.
        decoder (nn.Sequential): The decoder network.
    """

    def __init__(self):
        super().__init__()
        padding = 1
        output_padding = 0

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU(True),

            nn.Conv2d(16, 1, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, 16, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.LeakyReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=1,
                               padding=1, output_padding=0),
            nn.ConvTranspose2d(8, 1, kernel_size=5, stride=1,
                               padding=2, output_padding=output_padding),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass of the CAE model.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Reconstructed image tensor.
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x.view(-1, 1, 256, 256)
