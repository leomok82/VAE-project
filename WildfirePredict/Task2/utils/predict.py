import numpy as np
import torch
import matplotlib.pyplot as plt

def predict_samples(model, n_samples, latent_dim, device='cpu'):
    with torch.no_grad():
        nums = torch.randn(n_samples, latent_dim).to(device)
        samples = model.decode(nums).cpu().detach().numpy()
    return samples

def display_samples(samples, channel_size, n_rows = 9, n_cols = 1):
    fig ,ax = plt.subplots(n_cols,n_rows, figsize = (20,10))
    for i in range(9):
        ax[i].imshow(samples[1].reshape(1,channel_size, 256, 256)[0][i*2])
        ax[i].axis('off')
    plt.show()