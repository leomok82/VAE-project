import pytest
import numpy as np

from WildfireThomas.WildfireGenerate.task2functions import scoring


@pytest.fixture
def images():
    """
    Fixture that returns two sample images for testing.

    Returns:
        tuple: A tuple containing two numpy arrays representing images.
    """
    image1 = np.array([[1, 2], [3, 4]])
    image2 = np.array([[4, 3], [2, 1]])
    return image1, image2


@pytest.fixture
def simple_image_sets():
    """
    Fixture that returns two sets of generated and observed images for testing.

    Returns:
        tuple: A tuple containing two lists of numpy arrays representing generated and observed images.
    """
    generated_images = [
        np.array([[1, 2], [3, 4]]),
        np.array([[5, 6], [7, 8]]),
        np.array([[9, 10], [11, 12]])
    ]
    observed_images = [
        np.array([[1, 2], [3, 4]]),
        np.array([[5, 6], [7, 8]])
    ]
    return generated_images, observed_images


def test_mse(images):
    """
    Test the mean squared error (MSE) calculation.

    Args:
        images (tuple): A tuple containing two numpy arrays representing images.
    """
    image1, image2 = images
    result = scoring.mse(image1, image2)
    expected = np.mean((image1 - image2) ** 2)
    assert result == expected


def test_psnr(images):
    """
    Test the peak signal-to-noise ratio (PSNR) calculation.

    Args:
        images (tuple): A tuple containing two numpy arrays representing images.
    """
    image1, image2 = images
    result = scoring.psnr(image1, image2)
    assert result > 0


def test_cosine_sim(images):
    """
    Test the cosine similarity calculation.

    Args:
        images (tuple): A tuple containing two numpy arrays representing images.
    """
    image1, image2 = images
    result = scoring.cosine_sim(image1, image2)
    assert 0 <= result <= 1


def test_combined_similarity_score(images):
    """
    Test the combined similarity score calculation.

    Args:
        images (tuple): A tuple containing two numpy arrays representing images.
    """
    image1, image2 = images
    score = scoring.combined_similarity_score(image1, image2, 0.4, 0.3, 0.3)
    assert score >= 0


def test_compare_images(simple_image_sets):
    """
    Test the image comparison function.

    Args:
        simple_image_sets (tuple): A tuple containing two lists of numpy arrays representing generated and observed images.
    """
    generated_images, observed_images = simple_image_sets
    best_matches = scoring.compare_images(
        generated_images, observed_images, 0.48, 0.04, 0.48)
    assert len(best_matches) == len(observed_images)
    for idx, score in best_matches:
        assert idx >= 0
        assert score >= 0


if __name__ == '__main__':
    pytest.main()
