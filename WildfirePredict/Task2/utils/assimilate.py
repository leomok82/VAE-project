import torch
import numpy as np
import matplotlib.pyplot as plt

def encode_data(model, obs_data, generated_data, device):
    sensors_tensor = torch.tensor(obs_data, dtype=torch.float32).unsqueeze(1).to(device)
    generated_tensor = torch.tensor(generated_data, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        encoded_sensor_data = model.encoder(sensors_tensor)
        encoded_sensor_data = encoded_sensor_data.cpu().numpy()
        encoded_model_data = model.encoder(generated_tensor)
        encoded_model_data = encoded_model_data.cpu().numpy()

    return encoded_sensor_data, encoded_model_data, sensors_tensor

def compute_covariance_matrix(X):
    means = np.mean(X, axis=0)
    centered_data = X - means
    covariance_matrix = np.dot(centered_data.T, centered_data) / (X.shape[0] - 1)

    return covariance_matrix

def is_ill_conditioned(matrix):
    cond_number = np.linalg.cond(matrix)
    print(f"Condition number: {cond_number}")

def regularize_covariance(matrix, epsilon=100):
    #regularize the covariance matrix by a value to the diagonal elements.
    regularized_matrix = matrix + epsilon * np.identity(matrix.shape[0])
    return regularized_matrix

def compute_kalman_gain(B, H, R):
    temp_inv = np.linalg.inv(R + np.dot(H, np.dot(B, H.T)))
    K = np.dot(B, np.dot(H.T, temp_inv))
    return K

def mse(y_obs, y_pred):
    return np.square(np.subtract(y_obs, y_pred)).mean()

def update_state(x, K, H, y):
    return x + np.dot(K, (y - np.dot(H, x)))

def run_assimilation(flat_sensor, flat_model, latent_dim, encoded_shape, R_coeficient=0.001, epsilon=100):
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