import pytest
import numpy as np

from WildfireThomas.WildfireGenerate.task2functions import scoring

@pytest.fixture

def images():
    image1 = np.array([[1, 2], [3, 4]])
    image2 = np.array([[4, 3], [2, 1]])
    return image1, image2

@pytest.fixture
def simple_image_sets():
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
    image1, image2 = images
    result = scoring.mse(image1, image2)
    expected = np.mean((image1 - image2) ** 2)
    assert result == expected
 
def test_psnr(images):
    image1, image2 = images
    result = scoring.psnr(image1, image2)
    assert result > 0
 
def test_cosine_sim(images):
    image1, image2 = images
    result = scoring.cosine_sim(image1, image2)
    assert 0 <= result <= 1

def test_combined_similarity_score(images):
    image1, image2 = images
    score = scoring.combined_similarity_score(image1, image2, 0.4, 0.3, 0.3)
    assert score >= 0
 
def test_compare_images(simple_image_sets):
    generated_images, observed_images = simple_image_sets
    best_matches = scoring.compare_images(generated_images, observed_images, 0.48, 0.04, 0.48)
    assert len(best_matches) == len(observed_images)
    for idx, score in best_matches:
        assert idx >= 0
        assert score >= 0

if __name__ == '__main__':
    pytest.main()
 