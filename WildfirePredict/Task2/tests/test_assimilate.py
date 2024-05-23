import unittest
import numpy as np
import torch
from utils.assimilate import encode_data, compute_covariance_matrix, is_ill_conditioned, regularize_covariance, compute_kalman_gain, mse, update_state, run_assimilation, visualise

class DummyModel:
    def __init__(self):
        self.encoder = self.Encoder()

    class Encoder:
        def __call__(self, x):
            return x * 2

class TestAssimilate(unittest.TestCase):

    def setUp(self):
        self.model = DummyModel()
        self.obs_data = np.random.rand(10, 5)
        self.generated_data = np.random.rand(10, 5)
        self.device = 'cpu'

    def test_encode_data(self):
        encoded_sensor_data, encoded_model_data, sensors_tensor = encode_data(self.model, self.obs_data, self.generated_data, self.device)
        self.assertEqual(encoded_sensor_data.shape, encoded_model_data.shape)
        self.assertEqual(sensors_tensor.shape[0], 10)

    def test_compute_covariance_matrix(self):
        data = np.random.rand(10, 5)
        covariance_matrix = compute_covariance_matrix(data)
        self.assertEqual(covariance_matrix.shape, (5, 5))

    def test_is_ill_conditioned(self):
        matrix = np.random.rand(5, 5)
        is_ill_conditioned(matrix)  # Just check if it runs

    def test_regularize_covariance(self):
        matrix = np.random.rand(5, 5)
        regularized_matrix = regularize_covariance(matrix)
        self.assertEqual(regularized_matrix.shape, matrix.shape)

    def test_compute_kalman_gain(self):
        B = np.random.rand(5, 5)
        H = np.eye(5)
        R = np.random.rand(5, 5)
        K = compute_kalman_gain(B, H, R)
        self.assertEqual(K.shape, B.shape)

    def test_mse(self):
        y_obs = np.random.rand(10)
        y_pred = np.random.rand(10)
        error = mse(y_obs, y_pred)
        self.assertTrue(isinstance(error, float))

    def test_update_state(self):
        x = np.random.rand(5)
        K = np.random.rand(5, 5)
        H = np.eye(5)
        y = np.random.rand(5)
        updated_x = update_state(x, K, H, y)
        self.assertEqual(updated_x.shape, x.shape)

    def test_run_assimilation(self):
        flat_sensor = np.random.rand(10, 5)
        flat_model = np.random.rand(10, 5)
        latent_dim = 5
        encoded_shape = (10, 5)
        updated_state = run_assimilation(flat_sensor, flat_model, latent_dim, encoded_shape)
        self.assertEqual(updated_state.shape, encoded_shape)

    def test_visualise(self):
        sensor = np.random.rand(5, 1, 10, 10)
        generated_before = np.random.rand(5, 1, 10, 10)
        generated_after = np.random.rand(5, 1, 10, 10)
        visualise(sensor, generated_before, generated_after)  # Just check if it runs

if __name__ == '__main__':
    unittest.main()