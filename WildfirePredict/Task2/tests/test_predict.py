import unittest
import torch
import numpy as np
from utils.predict import predict_samples, display_samples

class DummyModel:
    def decode(self, z):
        return z.view(-1, 1, 256, 256)

class TestPredict(unittest.TestCase):

    def setUp(self):
        self.model = DummyModel()
        self.n_samples = 10
        self.latent_dim = 100
        self.device = 'cpu'
        self.channel_size = 1

    def test_predict_samples(self):
        samples = predict_samples(self.model, self.n_samples, self.latent_dim, self.device)
        self.assertEqual(samples.shape, (self.n_samples, 1, 256, 256))

    def test_display_samples(self):
        samples = predict_samples(self.model, self.n_samples, self.latent_dim, self.device)
        display_samples(samples, self.channel_size)  # Check if it runs without error

if __name__ == '__main__':
    unittest.main()