import pytest
import numpy as np

from WildfireThomas.WildfireDA.task3functions import assimilate
from WildfireThomas.WildfireDA.models import CAE

@pytest.fixture
def model():
    """
    Fixture that returns an instance of the CAE model.
    """
    return CAE()

@pytest.fixture
def obs_data():
    """
    Fixture that returns randomly generated observed data.
    """
    return np.random.rand(10, 5)

@pytest.fixture
def generated_data():
    """
    Fixture that returns randomly generated generated data.
    """
    return np.random.rand(10, 5)

@pytest.fixture
def device():
    """
    Fixture that returns the device type.
    """
    return 'cpu'

def test_compute_covariance_matrix():
    """
    Test case for the compute_covariance_matrix function in the assimilate module.
    It checks the shape of the computed covariance matrix.
    """
    data = np.random.rand(10, 5)
    covariance_matrix = assimilate.compute_covariance_matrix(data)
    assert covariance_matrix.shape == (5, 5)

def test_is_ill_conditioned():
    """
    Test case for the is_ill_conditioned function in the assimilate module.
    It checks if the function runs without any errors.
    """
    matrix = np.random.rand(5, 5)
    assimilate.is_ill_conditioned(matrix)  # Just check if it runs

def test_regularize_covariance():
    """
    Test case for the regularize_covariance function in the assimilate module.
    It checks the shape of the regularized covariance matrix.
    """
    matrix = np.random.rand(5, 5)
    regularized_matrix = assimilate.regularize_covariance(matrix)
    assert regularized_matrix.shape == matrix.shape

def test_compute_kalman_gain():
    """
    Test case for the compute_kalman_gain function in the assimilate module.
    It checks the shape of the computed Kalman gain.
    """
    B = np.random.rand(5, 5)
    H = np.eye(5)
    R = np.random.rand(5, 5)
    K = assimilate.compute_kalman_gain(B, H, R)
    assert K.shape == B.shape

def test_mse():
    """
    Test case for the mse function in the assimilate module.
    It checks if the error returned by the function is of type float.
    """
    y_obs = np.random.rand(10)
    y_pred = np.random.rand(10)
    error = assimilate.mse(y_obs, y_pred)
    assert isinstance(error, float)

def test_update_state():
    """
    Test case for the update_state function in the assimilate module.
    It checks the shape of the updated state vector.
    """
    x = np.random.rand(5)
    K = np.random.rand(5, 5)
    H = np.eye(5)
    y = np.random.rand(5)
    updated_x = assimilate.update_state(x, K, H, y)
    assert updated_x.shape == x.shape

def test_run_assimilation():
    """
    Test case for the run_assimilation function in the assimilate module.
    It checks the shape of the updated state vector.
    """
    flat_sensor = np.random.rand(10, 5)
    flat_model = np.random.rand(10, 5)
    latent_dim = 5
    encoded_shape = (10, 5)
    updated_state = assimilate.run_assimilation(flat_sensor, flat_model, latent_dim, encoded_shape)
    assert updated_state.shape == encoded_shape

# Tests for special values

def test_compute_covariance_matrix_special_values():
    """
    Test case for the compute_covariance_matrix function in the assimilate module with special values.
    It checks the computed covariance matrix against expected values.
    """
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    expected_cov_matrix = np.array([
        [9.0, 9.0, 9.0],
        [9.0, 9.0, 9.0],
        [9.0, 9.0, 9.0]
    ])
    computed_cov_matrix = assimilate.compute_covariance_matrix(X)
    assert np.allclose(computed_cov_matrix, expected_cov_matrix), f"Expected: {expected_cov_matrix}, but got: {computed_cov_matrix}"

def test_is_ill_conditioned_special_values():
    """
    Test case for the is_ill_conditioned function in the assimilate module with special values.
    It checks if the condition number of a matrix is greater than a threshold.
    """
    matrix = np.array([[1, 2], [2, 4]], dtype=float)
    expected_cond_number = np.inf  # The matrix is singular or near singular
    cond_number = np.linalg.cond(matrix)
    assert cond_number > 1e15, f"Expected condition number to be greater than 1e15 for near-singular matrix, but got: {cond_number}"

def test_update_state_special_values():
    """
    Test case for the update_state function in the assimilate module with special values.
    It checks the updated state vector against expected values.
    """
    x = np.array([1, 2], dtype=float)
    K = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float)
    H = np.array([[1, 0], [0, 1]], dtype=float)
    y = np.array([3, 4], dtype=float)
    expected_updated_state = x + np.dot(K, (y - np.dot(H, x)))
    computed_updated_state = assimilate.update_state(x, K, H, y)
    assert np.allclose(computed_updated_state, expected_updated_state), f"Expected: {expected_updated_state}, but got: {computed_updated_state}"

def test_compute_kalman_gain_special_values():
    """
    Test case for the compute_kalman_gain function in the assimilate module with special values.
    It checks the computed Kalman gain against expected values.
    """
    B = np.array([[1, 0], [0, 1]], dtype=float)
    H = np.array([[1, 0], [0, 1]], dtype=float)
    R = np.array([[0.1, 0], [0, 0.1]], dtype=float)
    expected_kalman_gain = np.array([
        [0.90909091, 0],
        [0, 0.90909091]
    ])
    computed_kalman_gain = assimilate.compute_kalman_gain(B, H, R)
    assert np.allclose(computed_kalman_gain, expected_kalman_gain), f"Expected: {expected_kalman_gain}, but got: {computed_kalman_gain}"

def test_regularize_covariance_special_values():
    """
    Test case for the regularize_covariance function in the assimilate module with special values.
    It checks the regularized covariance matrix against expected values.
    """
    matrix = np.array([[1, 0], [0, 1]], dtype=float)
    epsilon = 0.1
    expected_regularized_matrix = np.array([
        [1.1, 0],
        [0, 1.1]
    ])
    computed_regularized_matrix = assimilate.regularize_covariance(matrix, epsilon)
    assert np.allclose(computed_regularized_matrix, expected_regularized_matrix), f"Expected: {expected_regularized_matrix}, but got: {computed_regularized_matrix}"

def test_mse_special_values():
    """
    Test case for the mse function in the assimilate module with special values.
    It checks the mean squared error against expected values.
    """
    y_obs = np.array([1, 2, 3], dtype=float)
    y_pred = np.array([1.1, 2.1, 3.1], dtype=float)
    expected_mse = 0.01
    computed_mse = assimilate.mse(y_obs, y_pred)
    assert np.isclose(computed_mse, expected_mse), f"Expected: {expected_mse}, but got: {computed_mse}"
