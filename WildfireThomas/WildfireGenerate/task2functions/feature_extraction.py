import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

__all__ = ['extract_features', 'compute_pca']


def extract_features(images1, images2, mse_func, cos_sim_func, psnr_func):
    """
    Extracts features from pairs of images.

    Parameters:
    - images1 (list): List of images.
    - images2 (list): List of images.
    - mse_func (function): Function to calculate Mean Squared Error (MSE) score between two images.
    - cos_sim_func (function): Function to calculate Cosine Similarity score between two images.
    - psnr_func (function): Function to calculate Peak Signal-to-Noise Ratio (PSNR) score between two images.

    Returns:
    - features (list): List of extracted features, where each feature is a list of MSE score, 1 - Cosine Similarity score, and 1 / PSNR score.
    """
    features = []
    for img1 in images1:
        for img2 in images2:
            mse_score = mse_func(img1, img2)
            cos_sim_score = cos_sim_func(img1, img2)
            psnr_score = psnr_func(img1, img2)
            features.append([mse_score, 1 - cos_sim_score, 1 / psnr_score])
    return features


def compute_pca(features):
    """
    Computes Principal Component Analysis (PCA) on the given features.

    Parameters:
    - features (list): List of features.

    Returns:
    - w1 (float): Weight for MSE importance.
    - w2 (float): Weight for Cosine Similarity importance.
    - w3 (float): Weight for PSNR importance.
    """
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    pca = PCA(n_components=3)
    pca.fit(features_scaled)
    explained_variance = pca.explained_variance_ratio_

    components = pca.components_

    mse_importance = np.abs(components[0, 0])
    cos_sim_importance = np.abs(components[0, 1])
    psnr_importance = np.abs(components[0, 2])

    total_importance = mse_importance + cos_sim_importance + psnr_importance
    w1 = mse_importance / total_importance
    w2 = cos_sim_importance / total_importance
    w3 = psnr_importance / total_importance

    return w1, w2, w3
