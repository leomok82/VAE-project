import pytest
import numpy as np
import sys
from models.CAEmodel import CAE
import task2functions.assimilate as assimilate

# Adjust the path to include the parent directory of task2functions
sys.path.append('WildfirePredict/Task2')

# @pytest.fixture
# def model():
#     return CAE()

@pytest.fixture
def obs_data():
    return np.random.rand(10, 5)

@pytest.fixture
def generated_data():
    return np.random.rand(10, 5)

@pytest.fixture
def device():
    return 'cpu'

# def test_encode_data(model, obs_data, generated_data, device):
#     encoded_sensor_data, encoded_model_data, sensors_tensor = assimilate.encode_data(model, obs_data, generated_data, device)
#     assert encoded_sensor_data.shape == encoded_model_data.shape
#     assert sensors_tensor.shape[0] == 10

def test_compute_covariance_matrix():
    data = np.random.rand(10, 5)
    covariance_matrix = assimilate.compute_covariance_matrix(data)
    assert covariance_matrix.shape == (5, 5)

def test_is_ill_conditioned():
    matrix = np.random.rand(5, 5)
    assimilate.is_ill_conditioned(matrix)  # Just check if it runs

def test_regularize_covariance():
    matrix = np.random.rand(5, 5)
    regularized_matrix = assimilate.regularize_covariance(matrix)
    assert regularized_matrix.shape == matrix.shape

def test_compute_kalman_gain():
    B = np.random.rand(5, 5)
    H = np.eye(5)
    R = np.random.rand(5, 5)
    K = assimilate.compute_kalman_gain(B, H, R)
    assert K.shape == B.shape

def test_mse():
    y_obs = np.random.rand(10)
    y_pred = np.random.rand(10)
    error = assimilate.mse(y_obs, y_pred)
    assert isinstance(error, float)

def test_update_state():
    x = np.random.rand(5)
    K = np.random.rand(5, 5)
    H = np.eye(5)
    y = np.random.rand(5)
    updated_x = assimilate.update_state(x, K, H, y)
    assert updated_x.shape == x.shape

def test_run_assimilation():
    flat_sensor = np.random.rand(10, 5)
    flat_model = np.random.rand(10, 5)
    latent_dim = 5
    encoded_shape = (10, 5)
    updated_state = assimilate.run_assimilation(flat_sensor, flat_model, latent_dim, encoded_shape)
    assert updated_state.shape == encoded_shape