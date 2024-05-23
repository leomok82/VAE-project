import torch
import torch.nn as nn

__all__ = ['CAE']

class CAE(nn.Module):
    def __init__(self):
        super().__init__()
        padding = 1
        output_padding =0

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=1, padding= 2),
            nn.Conv2d(8, 16, kernel_size = 3, stride = 1, padding = 1),
            nn.MaxPool2d(2,2),
            nn.LeakyReLU(True),

            nn.Conv2d(16, 1, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(True),

        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.ConvTranspose2d(8, 1, kernel_size=5, stride=1, padding=2, output_padding=output_padding),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x.view(-1,1,256,256)