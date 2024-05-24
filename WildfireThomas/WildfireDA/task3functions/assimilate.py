import torch
import numpy as np
import matplotlib.pyplot as plt

__all__ = ['compute_covariance_matrix', 'is_ill_conditioned', 'regularize_covariance', 'compute_kalman_gain', 'mse']

def compute_covariance_matrix(X):
    """
    Compute the covariance matrix of the input data.

    Args:
        X (numpy.ndarray): Input data matrix.

    Returns:
        numpy.ndarray: Covariance matrix of the input data.
    """
    means = np.mean(X, axis=0)
    centered_data = X - means
    covariance_matrix = np.dot(centered_data.T, centered_data) / (X.shape[0] - 1)

    return covariance_matrix

def is_ill_conditioned(matrix):
    """
    Check if the given matrix is ill-conditioned.

    Args:
        matrix (numpy.ndarray): Input matrix.

    Returns:
        None
    """
    cond_number = np.linalg.cond(matrix)
    print(f"Condition number: {cond_number}")

def regularize_covariance(matrix, epsilon=100):
    """
    Regularize the covariance matrix by adding a value to the diagonal elements.

    Args:
        matrix (numpy.ndarray): Input matrix.
        epsilon (float, optional): Regularization parameter. Defaults to 100.

    Returns:
        numpy.ndarray: Regularized covariance matrix.
    """
    regularized_matrix = matrix + epsilon * np.identity(matrix.shape[0])
    return regularized_matrix

def compute_kalman_gain(B, H, R):
    """
    Compute the Kalman gain.

    Args:
        B (numpy.ndarray): Covariance matrix of the model.
        H (numpy.ndarray): Observation matrix.
        R (numpy.ndarray): Covariance matrix of the sensor.

    Returns:
        numpy.ndarray: Kalman gain matrix.
    """
    temp_inv = np.linalg.inv(R + np.dot(H, np.dot(B, H.T)))
    K = np.dot(B, np.dot(H.T, temp_inv))
    return K

def mse(y_obs, y_pred):
    """
    Compute the mean squared error between the observed and predicted values.

    Args:
        y_obs (numpy.ndarray): Observed values.
        y_pred (numpy.ndarray): Predicted values.

    Returns:
        float: Mean squared error.
    """
    return np.square(np.subtract(y_obs, y_pred)).mean()

def update_state(x, K, H, y):
    """
    Update the state using the Kalman gain.

    Args:
        x (numpy.ndarray): Current state.
        K (numpy.ndarray): Kalman gain matrix.
        H (numpy.ndarray): Observation matrix.
        y (numpy.ndarray): Observed values.

    Returns:
        numpy.ndarray: Updated state.
    """
    return x + np.dot(K, (y - np.dot(H, x)))

def run_assimilation(flat_sensor, flat_model, latent_dim, encoded_shape, R_coeficient=0.001, epsilon=100):
    """
    Run the assimilation process.

    Args:
        flat_sensor (numpy.ndarray): Flattened sensor data.
        flat_model (numpy.ndarray): Flattened model data.
        latent_dim (int): Dimension of the latent space.
        encoded_shape (tuple): Shape of the encoded data.
        R_coeficient (float, optional): Coefficient for scaling the sensor covariance matrix. Defaults to 0.001.
        epsilon (float, optional): Regularization parameter for the covariance matrices. Defaults to 100.

    Returns:
        numpy.ndarray: Updated state after assimilation.
    """
    R = compute_covariance_matrix(flat_sensor)
    B = compute_covariance_matrix(flat_model)
    R_regularized = regularize_covariance(R)
    B_regularized = regularize_covariance(B, epsilon)
    R_regularized = R_regularized * R_coeficient

    H = np.eye(latent_dim)
    K = compute_kalman_gain(B_regularized, H, R_regularized)

    print("Kalman Gain shape:", K.shape)

    updated_state_flattened = update_state(flat_model.T, K, H, flat_sensor.T)
    updated_state = updated_state_flattened.T.reshape(encoded_shape)

    return updated_state

def visualise(sensor, generated_before, generated_after):
    """
    Visualize the observed, generated, and assimilated images.

    Args:
        sensor (numpy.ndarray): Observed sensor data.
        generated_before (numpy.ndarray): Generated data before assimilation.
        generated_after (numpy.ndarray): Generated data after assimilation.

    Returns:
        None
    """
    No = sensor.shape[0]
    fig, axes = plt.subplots(3, No, figsize=(15, 9))
    for i in range(5):
        # Plot observed images
        axes[0, i].imshow(sensor[i, 0], cmap='viridis')
        axes[0, i].set_title(f'Observed {i+1}')
        axes[0, i].axis('off')

        # Plot generated images
        axes[1, i].imshow(generated_before[i, 0], cmap='viridis')
        axes[1, i].set_title(f'Generated {i+1}')
        axes[1, i].axis('off')

        # Plot assimilated images
        axes[2, i].imshow(generated_after[i, 0], cmap='viridis')
        axes[2, i].set_title(f'Assimilated {i+1}')
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.show()