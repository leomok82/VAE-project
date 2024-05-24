import pytest
import numpy as np

from WildfireThomas.WildfireGenerate.task2functions import feature_extraction, scoring


@pytest.fixture
def images():
    """
    Fixture function that returns two lists of images.

    Parameters:
    - None

    Returns:
    - A tuple containing two lists of images.
    """
    image1 = np.array([[1, 2], [3, 4]])
    image2 = np.array([[4, 3], [2, 1]])
    images1 = [image1]
    images2 = [image2]
    return images1, images2


def test_extract_features(images):
    """
    Test the extract_features function.

    Parameters:
    - images: A tuple containing two lists of images.

    Returns:
    - None
    """
    images1, images2 = images
    features = feature_extraction.extract_features(
        images1, images2, scoring.mse, scoring.cosine_sim, scoring.psnr)
    assert len(features) == 1
    assert len(features[0]) == 3


def test_compute_weights():
    """
    Test the compute_weights function.

    Parameters:
    - None

    Returns:
    - None
    """
    features = [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5]]
    w1, w2, w3 = feature_extraction.compute_pca(features)
    assert pytest.approx(w1 + w2 + w3, 0.00001) == 1.0


if __name__ == '__main__':
    pytest.main()
