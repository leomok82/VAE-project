import pytest
import numpy as np

from WildfireThomas.WildfireGenerate.task2functions import predict
from WildfireThomas.WildfireGenerate.models import VAE


@pytest.fixture
def model():
    """
    Fixture that returns an instance of the VAE model.
    """
    return VAE()


@pytest.fixture
def n_samples():
    """
    Fixture that returns the number of samples to generate.
    """
    return 10


@pytest.fixture
def latent_dim():
    """
    Fixture that returns the dimension of the latent space.
    """
    return 64


@pytest.fixture
def device():
    """
    Fixture that returns the device to run the model on.
    """
    return 'cpu'


@pytest.fixture
def channel_size():
    """
    Fixture that returns the number of channels in the input data.
    """
    return 19


def test_predict_samples(model, n_samples, latent_dim, device):
    """
    Test case for the predict_samples function.

    It generates samples using the given model and asserts that the number of generated samples
    matches the specified number of samples.
    """
    samples = predict.predict_samples(model, n_samples, latent_dim, device)
    print(samples.shape)
    print(len(samples))
    assert samples.shape[0] == n_samples


def test_display_samples(model, n_samples, latent_dim, device, channel_size):
    """
    Test case for the display_samples function.

    It generates samples using the given model and displays the generated samples.
    """
    samples = predict.predict_samples(model, n_samples, latent_dim, device)
    print(samples.shape)
    print(len(samples))
    # Check if it runs without error
    predict.display_samples(samples, channel_size)
