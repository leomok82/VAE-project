import torch
import matplotlib.pyplot as plt

__all__ = ['predict_samples', 'display_samples']


def predict_samples(model, n_samples, latent_dim, device='cpu'):
    """
    Generate samples from a given model.

    Args:
        model (torch.nn.Module): The trained model.
        n_samples (int): The number of samples to generate.
        latent_dim (int): The dimension of the latent space.
        device (str, optional): The device to use for computation. Defaults to 'cpu'.

    Returns:
        numpy.ndarray: The generated samples.
    """
    with torch.no_grad():
        nums = torch.randn(n_samples, latent_dim).to(device)
        samples = model.decoder(nums).cpu().detach().numpy()
    return samples


def display_samples(samples, channel_size, n_rows=9, n_cols=1):
    """
    Display the generated samples.

    Args:
        samples (numpy.ndarray): The generated samples.
        channel_size (int): The size of the channel.
        n_rows (int, optional): The number of rows in the plot. Defaults to 9.
        n_cols (int, optional): The number of columns in the plot. Defaults to 1.
    """
    fig, ax = plt.subplots(n_cols, n_rows, figsize=(20, 10))
    for i in range(9):
        ax[i].imshow(samples[1].reshape(1, channel_size, 256, 256)[0][i*2])
        ax[i].axis('off')
    plt.show()
