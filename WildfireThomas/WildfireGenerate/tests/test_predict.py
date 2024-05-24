import pytest
import numpy as np

from WildfireThomas.WildfireGenerate.task2functions import predict
from WildfireThomas.WildfireGenerate.models import VAE

@pytest.fixture
def model():
    return VAE()

@pytest.fixture
def n_samples():
    return 10

@pytest.fixture
def latent_dim():
    return 64

@pytest.fixture
def device():
    return 'cpu'

@pytest.fixture
def channel_size():
    return 19

def test_predict_samples(model, n_samples, latent_dim, device):
    samples = predict.predict_samples(model, n_samples, latent_dim, device)
    print(samples.shape)
    print(len(samples))
    assert samples.shape[0] == n_samples

def test_display_samples(model, n_samples, latent_dim, device, channel_size):
    samples = predict.predict_samples(model, n_samples, latent_dim, device)
    print(samples.shape)
    print(len(samples))
    predict.display_samples(samples, channel_size)  # Check if it runs without error
